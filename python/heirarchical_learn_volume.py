#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# @file   mlp_learning_an_image_pytorch.py
# @author Thomas Müller, NVIDIA
# @brief  Replicates the behavior of the CUDA mlp_learning_an_image.cu sample
#         using tiny-cuda-nn's PyTorch extension. Runs ~2x slower than native.

import argparse
import commentjson as json
import numpy as np
import os
import sys
import torch
import intel_extension_for_pytorch
import time
import torch.nn as nn
from torch.func import stack_module_state, functional_call
from torch import vmap
import copy

try:
    import tiny_dpcpp_nn as tnn
except ImportError:
    print("This sample requires the tiny-dpcpp-nn extension for PyTorch.")
    print("You can install it by running:")
    print("============================================================")
    print("tiny-dpcpp-nn$ cd dpcpp_bindings")
    print("tiny-dpcpp-nn/dpcpp_bindings/$ pip install -e .")
    print("============================================================")
    sys.exit()

from common import read_image, write_image, ROOT_DIR, read_volume, write_volume
from encoder import HashEmbedderNative
from heirarchical_encoder import HeirarchicalHashEmbedderNative
from MLP_native import MLP_Native
import dvnr_sampler as spl

DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")


def get_args():
    parser = argparse.ArgumentParser(
        description="Image benchmark using PyTorch bindings."
    )

    # for dvnr volume sampler
    parser.add_argument(
        # '--filename', type=str, default="data/images/chameleon_1024x1024x1080_float32.raw", help="volume data file"
        '--filename', type=str, default="data/images/bonsai.raw", help="volume data file"
    )
    parser.add_argument(
        # "--dims", type=int, nargs=3, default=[1024, 1024, 1080], help="volume data dimensions"
        "--dims", type=int, nargs=3, default=[256, 256, 256], help="volume data dimensions"
    )
    parser.add_argument(
        # "--type", type=str, default="float32", help="volume data type"
        "--type", type=str, default="uint8", help="volume data type"
    )
    parser.add_argument(
        # "--max_val", type=float, default=1.0, help="volume data maximum value"
        "--max_val", type=float, default=255.0, help="volume data maximum value"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="data/config_hash.json",
        help="JSON config for tiny-dpcpp-nn",
    )
    parser.add_argument(
        "n_steps",
        nargs="?",
        type=int,
        default=1001,
        help="Number of training steps",
    )
    parser.add_argument(
        "result_filename",
        nargs="?",
        default="result.raw",
        help="Number of training steps",
    )

    args = parser.parse_args()
    return args

def accumulate_squared_errors_of_slice(output, targets):
    return ((output - targets) ** 2).sum()

def calculate_PSNR_from_squared_errors_sum(squared_errors_sum, resolution):
    temp = squared_errors_sum / (resolution[0] * resolution[1] * resolution[2])
    return 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(temp)))

def generate_batch_samples(batch_size: int, partition_size: int, n_pos_dims: int, device=torch.device("cpu")):
    # only support 3 dimensional inputs now
    assert (n_pos_dims == 3)
    mini_batch_size = int(batch_size / (partition_size ** n_pos_dims))
    interval_along_one_dim = 1.0 / partition_size
    mini_batches = []
    # generate batches of samples for different encoders
    for i in range(partition_size):
        for j in range(partition_size):
            for k in range(partition_size):
                # shape: [mini_batch_size, 3] = [mini_batch_size, 3] * [3] + [3] (broadcasting)
                mini_batch = torch.rand([mini_batch_size, 3], device=device, dtype=torch.float32) * torch.tensor(
                    [interval_along_one_dim, interval_along_one_dim, interval_along_one_dim], device=device) + torch.tensor([
                        interval_along_one_dim * i, interval_along_one_dim * j, interval_along_one_dim * k], device=device)
                mini_batches.append(mini_batch)
    mini_batches = torch.stack(mini_batches, dim=0)
    return mini_batches, mini_batch_size

def get_batch_encoder_idx(coords: torch.tensor, partition_size: int):
    coords = coords.contiguous().to(torch.float32)
    
    enc_idx = coords * torch.tensor([partition_size, partition_size, partition_size], device=coords.device).float()
    enc_idx = enc_idx.long()
    x_idx = enc_idx[:,0].clamp(min=0, max=partition_size - 1)
    y_idx = enc_idx[:,1].clamp(min=0, max=partition_size - 1)
    z_idx = enc_idx[:,2].clamp(min=0, max=partition_size - 1)
    flat_idx = x_idx + y_idx * partition_size + z_idx * partition_size ** 2 
    return flat_idx
    
def model_inference(coords: torch.tensor, enc_idx: torch.tensor, n_pos_dims: int, partition_size: int, encodings: any, network: any):
    encoded_results = torch.zeros(coords.shape[0], encodings[0].n_output_dims, device=coords.device).float()

    for i in range(partition_size ** n_pos_dims):
        group_idx = torch.nonzero(enc_idx == i).squeeze()
        encoded_results[group_idx] = encodings[i](coords[group_idx])
    
    return network(encoded_results)
    
def main():
    print("================================================================")
    print("This script replicates the behavior of the native SYCL example  ")
    print("mlp_learning_an_image.cu using tiny-dpcpp-nn's PyTorch extension.")
    print("================================================================")
    device_name = "xpu"
    # device_name = "cpu"
    device = torch.device(device_name)
    args = get_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    n_channels = 1
    sampler = spl.create_sampler("structuredRegular", "openvkl", filename=args.filename, dims=args.dims, dtype=args.type, n_channels=n_channels)

    # model = tnn.NetworkWithInputEncoding(
    #     n_input_dims=3,
    #     n_output_dims=n_channels,
    #     encoding_config=config["encoding"],
    #     network_config=config["network"],
    # ).to(device)

    # ===================================================================================================
    # The following is equivalent to the above, but slower. Only use "naked" tnn.Encoding and
    # tnn.Network when you don't want to combine them. Otherwise, use tnn.NetworkWithInputEncoding.
    # ===================================================================================================

    partition_size = 2
    n_pos_dims = 3
    # encoding = tnn.Encoding(
    #     n_input_dims=3,
    #     encoding_config=config["encoding"],
    #     dtype=torch.float,
    # )
    # encoding = HeirarchicalHashEmbedderNative(n_pos_dims=3, partition_size=2, individual_encoding_config=config["encoding"])
    encodings = nn.ModuleList([HashEmbedderNative(n_pos_dims=n_pos_dims, encoding_config=config["encoding"]).to(device)
                                            for _ in range(partition_size ** n_pos_dims)])
    # network = tnn.Network(
    #     n_input_dims=encoding.n_output_dims,
    #     n_output_dims=n_channels,
    #     network_config=config["network"],
    # )
    network = MLP_Native(n_input_dims=encodings[0].n_output_dims, n_output_dims=n_channels, network_config=config["network"]).to(device)
    # models = nn.ModuleList([torch.nn.Sequential(encoding, network).to(device) for encoding in encodings])

    # version with using vmap
    # params, buffers = stack_module_state(encodings)
    # base_model = copy.deepcopy(encodings[0])
    # base_model = base_model.to('meta')
    # def fmodel(params, buffers, x):
    #     return functional_call(base_model, (params, buffers), (x,))
    # enc_optimizer = torch.optim.Adam(params.values(), lr=1e-3)
    
    enc_optimizer = torch.optim.Adam(encodings.parameters(), lr=1e-3)
    net_optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    # Variables for saving/displaying image results
    resolution = args.dims
    
    # # Generate coordinates of regular grid
    # x = torch.arange(resolution[0], dtype=torch.float32, device=device) / (resolution[0] - 1)
    # y = torch.arange(resolution[1], dtype=torch.float32, device=device) / (resolution[1] - 1)
    # z = torch.arange(resolution[2], dtype=torch.float32, device=device) / (resolution[2] - 1)

    # # Create the grid using meshgrid
    # zv, yv, xv = torch.meshgrid([z, y, x])

    # # Stack the coordinates along the last dimension and reshape
    # zyx = torch.stack((zv.flatten() ,yv.flatten(), xv.flatten())).t()
    # xyz = zyx[:, [2, 1, 0]]
    # # print(zyx)
    
    prev_time = time.perf_counter()

    batch_size = 2**16
    interval = 10

    print(f"Beginning optimization with {args.n_steps} training steps.")


    for i in range(args.n_steps):
        # mini_batches, mini_batch_size = generate_batch_samples(batch_size=batch_size, partition_size=partition_size, n_pos_dims=n_pos_dims, device=device)
        # targets = traced_image(mini_batches.view(-1, 3))
        
        coords, targets = spl.sample(sampler, batch_size)
        coords = coords.to(device_name)
        targets = targets.to(device_name)
        
        # version with using vmap
        # shape of enc_output is [number of encoders, mini batch size, number of feature vec dims]
        # enc_output = vmap(fmodel)(params, buffers, mini_batches)
        # enc_output = enc_output.view(-1, encodings[0].n_output_dims)
        
        # version without using vmap
        # enc_output = []
        # for mini_batch, encoding in zip(mini_batches ,encodings):
        #     enc_output.append(encoding(mini_batch))
        # enc_output = torch.cat(enc_output, dim=0)
        
        # output = network(enc_output)

        # version complying with the coordinates generated by dvnr sampler
        enc_idx = get_batch_encoder_idx(coords=coords, partition_size=partition_size)
        output = model_inference(coords=coords, enc_idx=enc_idx, n_pos_dims=n_pos_dims, partition_size=partition_size, encodings=encodings, network=network)
        # adjust the output size to align with the target size
        # output = output.view(-1)
        relative_l2_error = (output - targets.to(output.dtype)) ** 2 / (
            output.detach() ** 2 + 0.01
        )

        loss = relative_l2_error.mean()

        enc_optimizer.zero_grad()
        net_optimizer.zero_grad()
        loss.backward()
        net_optimizer.step()
        enc_optimizer.step()

        if i % interval == 0:
            loss_val = loss.item()
            torch.xpu.synchronize()
            elapsed_time = time.perf_counter() - prev_time
            print(f"Step#{i}: loss={loss_val} time={int(elapsed_time*1000000)}[µs]")

            path = f"{i}.raw"
            print(f"Writing '{path}'... ", end="")
                        
            squared_errors_sum = 0
            # Generate coordinates of regular gird on yz slices
            with torch.no_grad():
                for z in range(resolution[2]):
                    x = torch.arange(resolution[0], dtype=torch.float32) / (resolution[0] - 1)
                    y = torch.arange(resolution[1], dtype=torch.float32) / (resolution[1] - 1)
                    z_coord = torch.full((resolution[0] * resolution[1], 1), z, dtype=torch.float32) / (resolution[2] - 1)

                    # Create the grid using meshgrid
                    yv, xv = torch.meshgrid([y, x])

                    # Stack the coordinates along the last dimension and reshape
                    yx = torch.stack((yv.flatten(), xv.flatten())).t()
                    zyx = torch.cat((z_coord, yx), dim=1)
                    xyz = zyx[:, [2, 1, 0]]
                    
                    # temporary solumtion for inferencing large dataset
                    # need to refactor for better structure and flexibility for different dataset
                    num_chunks = 2
                    assert (xyz.shape[0] % num_chunks) == 0 
                    chunk_size = int(xyz.shape[0] / num_chunks)
                    for chunk_idx in range(num_chunks):
                        chunk = xyz[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                        
                        targets = torch.zeros([chunk.shape[0], 1]).float()
                        spl.decode(sampler, chunk, targets)
                        chunk = chunk.to(device_name)
                        targets = targets.to(device_name)
                        enc_idx_chunk = get_batch_encoder_idx(coords=chunk, partition_size=partition_size)
                        output = model_inference(coords=chunk, enc_idx=enc_idx_chunk, n_pos_dims=n_pos_dims, 
                                                partition_size=partition_size, encodings=encodings, network=network).clamp(0.0, 1.0)
                        squared_errors_sum += accumulate_squared_errors_of_slice(output=output, targets=targets)
                        write_volume(
                            path, 
                            # output.reshape([resolution[0], resolution[1]]
                            #             ).detach().cpu().numpy() * args.max_val,
                            output.detach().cpu().numpy() * args.max_val,
                            dtype=args.type,
                            # calculate offset by the number of elements in xy plane and chunk offset
                            offset= z * resolution[0] * resolution[1] + chunk_idx * chunk_size 
                        )
            print("done.")
            PSNR = calculate_PSNR_from_squared_errors_sum(squared_errors_sum=squared_errors_sum, resolution=resolution)
            print("PSNR:", PSNR)

            # Ignore the time spent saving the image
            prev_time = time.perf_counter()

            if i > 0 and interval < 1000:
                interval *= 10

        # adjust encoders parameters after inference or train in this iteration
        # print("iter: ", i, end=" // ")
        # compare the loss between different encoders
        loss_of_encoders = torch.zeros(partition_size ** n_pos_dims, device=relative_l2_error.device)
        for j in range(partition_size ** n_pos_dims):
            group_idx = torch.nonzero(enc_idx == j).squeeze()
            loss_of_encoders[j] = relative_l2_error[group_idx].mean()
        # print(" max idx:", torch.argmax(loss_of_encoders), " min idx:", torch.argmin(loss_of_encoders))
        
        # print("==================================================")
    if args.result_filename:
        print(f"Writing '{args.result_filename}'... ", end="")
        
        squared_errors_sum = 0
        # Generate coordinates of regular gird on yz slices
        with torch.no_grad():
            for z in range(resolution[2]):
                x = torch.arange(resolution[0], dtype=torch.float32) / (resolution[0] - 1)
                y = torch.arange(resolution[1], dtype=torch.float32) / (resolution[1] - 1)
                z_coord = torch.full((resolution[0] * resolution[1], 1), z, dtype=torch.float32) / (resolution[2] - 1)

                # Create the grid using meshgrid
                yv, xv = torch.meshgrid([y, x])

                # Stack the coordinates along the last dimension and reshape
                yx = torch.stack((yv.flatten(), xv.flatten())).t()
                zyx = torch.cat((z_coord, yx), dim=1)
                xyz = zyx[:, [2, 1, 0]]
                
                # temporary solumtion for inferencing large dataset
                # need to refactor for better structure and flexibility for different dataset
                num_chunks = 2
                assert (xyz.shape[0] % num_chunks) == 0 
                chunk_size = int(xyz.shape[0] / num_chunks)
                for chunk_idx in range(num_chunks):
                    chunk = xyz[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                    
                    targets = torch.zeros([chunk.shape[0], 1]).float()
                    spl.decode(sampler, chunk, targets)
                    chunk = chunk.to(device_name)
                    targets = targets.to(device_name)
                    enc_idx_chunk = get_batch_encoder_idx(coords=chunk, partition_size=partition_size)
                    output = model_inference(coords=chunk, enc_idx=enc_idx_chunk, n_pos_dims=n_pos_dims, 
                                            partition_size=partition_size, encodings=encodings, network=network).clamp(0.0, 1.0)
                    squared_errors_sum += accumulate_squared_errors_of_slice(output=output, targets=targets)
                    write_volume(
                        args.result_filename, 
                        # output.reshape([resolution[0], resolution[1]]
                        #             ).detach().cpu().numpy() * args.max_val,
                        output.detach().cpu().numpy() * args.max_val,
                        dtype=args.type,
                        # calculate offset by the number of elements in xy plane and chunk offset
                        offset= z * resolution[0] * resolution[1] + chunk_idx * chunk_size 
                    )
        print("done.")
        PSNR = calculate_PSNR_from_squared_errors_sum(squared_errors_sum=squared_errors_sum, resolution=resolution)
        print("PSNR:", PSNR)

if __name__ == "__main__":
    main()