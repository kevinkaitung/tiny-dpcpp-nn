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
# import intel_extension_for_pytorch
import time
import torch.nn as nn
from torch.func import stack_module_state, functional_call
from torch import vmap
from torch.utils.tensorboard import SummaryWriter
import copy

from common import read_image, write_image, ROOT_DIR, read_volume, write_volume
from encoder import HashEmbedderNative
from heirarchical_encoder import HeirarchicalHashEmbedderNative
from MLP_native import MLP_Native
import dvnr_sampler as spl
from loss_monitor import loss_monitor

DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")


def get_args():
    parser = argparse.ArgumentParser(
        description="Image benchmark using PyTorch bindings."
    )

    # for dvnr volume sampler
    parser.add_argument(
        # '--filename', type=str, default="data/images/chameleon_1024x1024x1080_float32.raw", help="volume data file"
        # '--filename', type=str, default="data/images/bonsai.raw", help="volume data file"
        '--filename', type=str, default="data/images/1atm.H2O.3x.1152.320.853f32.bin", help="volume data file"
    )
    parser.add_argument(
        # "--dims", type=int, nargs=3, default=[1024, 1024, 1080], help="volume data dimensions"
        # "--dims", type=int, nargs=3, default=[256, 256, 256], help="volume data dimensions"
        "--dims", type=int, nargs=3, default=[1152, 320, 853], help="volume data dimensions"
    )
    parser.add_argument(
        # "--type", type=str, default="float32", help="volume data type"
        # "--type", type=str, default="uint8", help="volume data type"
        "--type", type=str, default="float32", help="volume data type"
    )
    parser.add_argument(
        # "--max_val", type=float, default=1.0, help="volume data maximum value"
        # "--max_val", type=float, default=255.0, help="volume data maximum value"
        "--max_val", type=float, default=1.0, help="volume data maximum value"
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
    # device_name = "xpu"
    # device_name = "cpu"
    device_name = "cuda"
    device = torch.device(device_name)
    args = get_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    n_channels = 1
    sampler = spl.create_sampler("structuredRegular", "openvkl", filename=args.filename, dims=args.dims, dtype=args.type, n_channels=n_channels)

    
    test_partition_size = 2
    try_runs = 1
    file_name = 'records_of_partition_size_'+str(test_partition_size)+'(loss_monitor_ver).json'
    records = dict()
    run_numbers = []
    all_losses = []
    all_PSNRs = []
    all_model_sizes = []
    all_hashmap_sizes = []
    for it in range(try_runs):
        print("run number:", it)
        one_loss, one_PSNR, one_model_size, one_hashmap_size = train_volume(device_name=device_name, 
            device=device, args=args, config=config, n_channels=n_channels, sampler=sampler, partition_size=test_partition_size)
        run_numbers.append(it)
        all_losses.append(one_loss)
        all_PSNRs.append(one_PSNR)
        all_model_sizes.append(one_model_size)
        all_hashmap_sizes.append(one_hashmap_size)
    
    records["run_number"] = run_numbers
    records["loss"] = all_losses
    records["PSNR"] = all_PSNRs
    records["model_size"] = all_model_sizes
    records["hashmap_size_distribution"] = all_hashmap_sizes
    # Write the data to a JSON file
    with open(file_name, 'w') as file:
        json.dump(records, file, indent=4) 

def train_volume(device_name, device, args, config, n_channels, sampler, partition_size):
    calculate_PSNR = True
    # partition_size = 2
    n_pos_dims = 3
    
    # # calculate the times of each encoder has the max loss in a iteration
    # encoders_max_loss_counts = torch.zeros(partition_size ** n_pos_dims, dtype=torch.int32, device=device)
    # frequency to increase hash map size for an encoder
    hashmap_size_incre_freq = 200
    error_bounds = 0.0
    
    encodings = nn.ModuleList([HashEmbedderNative(n_pos_dims=n_pos_dims, encoding_config=config["encoding"]).to(device)
                                            for _ in range(partition_size ** n_pos_dims)])
    network = MLP_Native(n_input_dims=encodings[0].n_output_dims, n_output_dims=n_channels, network_config=config["network"]).to(device)
    
    optimizer = torch.optim.Adam([{"params":encodings.parameters()}, {"params":network.parameters()}], lr=1e-3)
    enc_loss_monitors = [loss_monitor(encoder=x, optimizer=optimizer, mode="min", factor=1, patience=60, threshold=1e-4,
                                        threshold_mode="rel", cooldown=0, max_sz=25, eps=1e-8, enc_idx=j, error_bound=error_bounds) for j, x in enumerate(encodings)]
    # enc_optimizer = torch.optim.Adam(encodings.parameters(), lr=1e-3)
    # net_optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    # Variables for saving/displaying image results
    resolution = args.dims
            
    prev_time = time.perf_counter()

    batch_size = 2**16
    interval = 10

    
    # calculate model size
    param_size = 0
    for param in encodings.parameters():
        param_size += param.nelement() * param.element_size()
    for param in network.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in encodings.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    for buffer in network.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('before training, model size: {:.3f}MB'.format(size_all_mb))

    print(f"Beginning optimization with {args.n_steps} training steps.")

    for i in range(args.n_steps):
        
        coords, targets = spl.sample(sampler, batch_size)
        coords = coords.to(device_name)
        targets = targets.to(device_name)
        
        # version complying with the coordinates generated by dvnr sampler
        enc_idx = get_batch_encoder_idx(coords=coords, partition_size=partition_size)
        output = model_inference(coords=coords, enc_idx=enc_idx, n_pos_dims=n_pos_dims, partition_size=partition_size, encodings=encodings, network=network)
        # adjust the output size to align with the target size
        # output = output.view(-1)
        relative_l2_error = (output - targets.to(output.dtype)) ** 2 / (
            output.detach() ** 2 + 0.01
        )

        # net_optimizer.zero_grad()
        # calculate loss of each encoder
        loss_of_encoders = torch.zeros(partition_size ** n_pos_dims, device=relative_l2_error.device)
        for j in range(partition_size ** n_pos_dims):
            # enc_optimizer.zero_grad()
            group_idx = torch.nonzero(enc_idx == j).squeeze().to(torch.int64)
            loss_of_encoders[j] = relative_l2_error[group_idx].mean()
            # if j == partition_size ** n_pos_dims - 1:
            #     loss_of_encoders[j].backward(retain_graph=False)
            # else:
            #     loss_of_encoders[j].backward(retain_graph=True)
            # enc_optimizer.step()
        # net_optimizer.step()
        # get index of the encoder who has max loss
        

        # total loss
        loss = relative_l2_error.mean()
        # # find encoders whose losses are larger than average
        # large_loss_enc_idx = torch.nonzero(loss_of_encoders >= loss)
        # encoders_max_loss_counts[large_loss_enc_idx] += 1

        optimizer.zero_grad()
        # enc_optimizer.zero_grad()
        # net_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # enc_optimizer.step()
        # net_optimizer.step()

        if i % interval == 0 or i == (args.n_steps - 1):
            loss_val = loss.item()
            # torch.xpu.synchronize()
            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - prev_time
            print(f"Step#{i}: loss={loss_val} time={int(elapsed_time*1000000)}[µs]")
            
            if calculate_PSNR and i == (args.n_steps - 1):
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
                print("done.")
                PSNR = calculate_PSNR_from_squared_errors_sum(squared_errors_sum=squared_errors_sum, resolution=resolution)
                PSNR_val = PSNR.item()
                print("PSNR:", PSNR_val)
                

            # Ignore the time spent saving the image
            prev_time = time.perf_counter()

            if i > 0 and interval < args.n_steps:
                interval *= 10

        # if approaching the end of training, stop increase hashmap size
        # need to improve by checking the available memory size
        # if i < args.n_steps - 100:
        if i < 2300:
            for j in range(partition_size ** n_pos_dims):
                enc_loss_monitors[j].step(loss_of_encoders[j])
        # # adjust encoders parameters after inferencing or training in this iteration
        # # if a certain encoder reach the frequency, increase the hash map size of that encoder
        # for j in large_loss_enc_idx:
        #     if encoders_max_loss_counts[j] % hashmap_size_incre_freq == 0:
        #         encodings[j].increase_embedding_size_by_two()
        #         new_size = encodings[j].get_log2_hashmap_size()
        #         print("iter:", i, " encoder idx ", j, " increase hash map size to 2^", new_size)
        #         # add new parameters to encodings parameters in current optimizer
        #         optimizer.add_param_group({"params": encodings[j].parameters()})
        #         # recreate optimizer
        #         # optimizer = torch.optim.Adam([{"params":encodings.parameters()}, {"params":network.parameters()}], lr=1e-3)
        #         # enc_optimizer.add_param_group({"params": encodings[max_loss_enc_idx].parameters()})
        #         # enc_optimizer = torch.optim.Adam(encodings.parameters(), lr=1e-3)
        #         # net_optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    
    # calculate model size
    param_size = 0
    for param in encodings.parameters():
        param_size += param.nelement() * param.element_size()
    for param in network.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in encodings.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    for buffer in network.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
            
    # print("==================================================")
    
    print("hashmap sz of encoders at final iteration:")
    hashmap_sz_of_encoders = [x.get_log2_hashmap_size() for x in encodings]
    print(hashmap_sz_of_encoders)
    
    return loss_val, PSNR_val, size_all_mb, hashmap_sz_of_encoders

if __name__ == "__main__":
    main()