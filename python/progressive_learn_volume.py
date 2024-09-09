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
import subprocess

from common import read_image, write_image, ROOT_DIR, read_volume, write_volume
from encoder import HashEmbedderNative
from heirarchical_encoder import HeirarchicalHashEmbedderNative
from MLP_native import MLP_Native
import dvnr_sampler as spl
from loss_monitor import loss_monitor

if torch.cuda.is_available():
    import tinycudann as tcnn

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
        default=10001,
        # default=5001,
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

def accumulate_squared_errors_of_slice(diff_targets_output):
    return ((diff_targets_output) ** 2).sum()

def calculate_PSNR_from_squared_errors_sum(squared_errors_sum, resolution, max_value):
    # max value = 2 if value range is -1 ~ 1
    # max_value = 2.0
    temp = squared_errors_sum / (resolution[0] * resolution[1] * resolution[2])
    return 20 * torch.log10(max_value / torch.sqrt(torch.tensor(temp)))

sampler_max = 1.0
sampler_min = 0.0

def normalize_to_neg_one_and_one(data, original_max, original_min):
    original_range = original_max - original_min
    normalized_max = 1.0
    normalized_min = -1.0
    normalized_range = normalized_max - normalized_min
    return ((data - original_min) / original_range) * normalized_range + normalized_min

def denormalize_from_neg_one_and_one(data, original_max, original_min):
    original_range = original_max - original_min
    normalized_max = 1.0
    normalized_min = -1.0
    normalized_range = normalized_max - normalized_min
    return ((data - normalized_min) / normalized_range) * original_range + original_min

def decode_and_calculate_PSNR(device_name, resolution, sampler, model, path_name, datatype, old_diff_max, old_diff_min, prog_iter):
    squared_errors_sum = 0
    track_max = -10
    track_min = 10
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
            num_chunks = 1
            assert (xyz.shape[0] % num_chunks) == 0
            chunk_size = int(xyz.shape[0] / num_chunks)
            for chunk_idx in range(num_chunks):
                chunk = xyz[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                
                targets = torch.zeros([chunk.shape[0], 1]).float()
                spl.decode(sampler, chunk, targets)
                chunk = chunk.to(device_name)
                # targets and output are normalized now
                targets = targets.to(device_name)
                if prog_iter > 0:
                    # normalize targets from 0~1 to -1~1
                    targets = normalize_to_neg_one_and_one(targets, sampler_max, sampler_min)
                
                # output = model(chunk).clamp(0.0, 1.0)
                output = model(chunk)
                
                # for checking the max and min output values from the model
                if track_max < output.max():
                    track_max = output.max()
                if track_min > output.min():
                    track_min = output.min()
                diff_targets_output = targets - output
                squared_errors_sum += accumulate_squared_errors_of_slice(diff_targets_output=diff_targets_output)
                
                if prog_iter == 0:
                    output = normalize_to_neg_one_and_one(output, 1.0, 0.0)
                else:
                    # denormalized output and targets from -1~1 to original value range
                    output = denormalize_from_neg_one_and_one(output, old_diff_max, old_diff_min)
                
                write_volume(
                    path_name, 
                    # output.reshape([resolution[0], resolution[1]]
                    #             ).detach().cpu().numpy() * args.max_val,
                    # output.detach().cpu().numpy() * args.max_val,
                    # need to denormalize with current volume(sampler)'s value range
                    # denormalized_output.detach().cpu().numpy(),
                    output.detach().cpu().numpy(),
                    dtype=datatype,
                    # calculate offset by the number of elements in xy plane and chunk offset
                    offset= z * resolution[0] * resolution[1] + chunk_idx * chunk_size 
                )
    if prog_iter == 0:
        max_value = 1.0
    else:
        max_value = 2.0
    PSNR = calculate_PSNR_from_squared_errors_sum(squared_errors_sum=squared_errors_sum, resolution=resolution, max_value=max_value) # max value = 1 - (-1)
    PSNR_val = PSNR.item()
    print("PSNR:", PSNR_val)
    # import pdb; pdb.set_trace()
    return PSNR_val
    
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

    try_runs = 1
    for it in range(try_runs):
        print("run number:", it)
        train_volume_progressively(device_name=device_name, device=device, args=args)

def train_volume_progressively(device_name, device, args):
    
    with open(args.config) as config_file:
        config = json.load(config_file)
    resolution = args.dims
    training_steps = args.n_steps
    n_pos_dims = len(resolution)
    n_channels = 1
    
    track_loss = False
    if track_loss:
        writer = SummaryWriter()
    
    progressive_iter = 8
    neural_network_iter = 4
    
    # use our own pytorch model
    # encodings = [HashEmbedderNative(n_pos_dims=n_pos_dims, encoding_config=config["encoding"]) for _ in range(progressive_iter)]
    # networks = [MLP_Native(n_input_dims=encodings[0].n_output_dims, n_output_dims=n_channels, network_config=config["network"]) for _ in range(progressive_iter)]
    
    # use tiny-cuda-nn's model
    encodings = [tcnn.Encoding(n_pos_dims, config["encoding"]) for _ in range(progressive_iter)]
    
    networks = []
    for i in range(progressive_iter):
        current_config = config["network"].copy()
        if i == 0:
            current_config["activation"] = "ReLU"
        networks.append(tcnn.Network(encodings[0].n_output_dims, n_channels, current_config))
    
    models = nn.ModuleList([torch.nn.Sequential(encodings[i], networks[i]).to(device) for i in range(progressive_iter)])
    optimizers = [torch.optim.Adam([{"params":models[i].parameters()}], lr=1e-3) for i in range(progressive_iter)]
    # optimizer = torch.optim.Adam([{"params":models.parameters()}], lr=1e-3)
    
    # volume queried from sampler would be normalized to 0~1
    # but will intentionally normalize to -1~1 when querying sampler
    volume_max = 1.0
    volume_min = -1.0
    sampler = spl.create_sampler("structuredRegular", "openvkl", filename=args.filename, dims=args.dims, dtype=args.type, n_channels=n_channels)
    
    # just for convenience not to create grid again to query values in sampler
    # actually, no need to read original dataset and normalize it again
    # unnormalized form
    original_volume = read_volume(file=args.filename, shape=args.dims, dtype=args.type)
    # normalize to -1~1
    original_volume = normalize_to_neg_one_and_one(original_volume, original_volume.max(), original_volume.min())
    # original_volume_max = original_volume.max()
    # original_volume_min = original_volume.min()
    # # normalize original volume (0~1)
    # original_volume = (original_volume - original_volume_min) / (original_volume_max - original_volume_min) * 1.0
    
    # directly reading volume from file doesn't have normalization to -1~1
    residual_path_name = args.filename
    
    losses_all_progressive_iters = []
    for i in range(progressive_iter):
        print("Progressive iteration: ", i)
        decomp_path_name = "volume_progress_" + str(i) + "_decompressed.bin"
        
        # use neural network for early iterations
        if i < neural_network_iter:
            losses = train_volume_one_time(device_name=device_name, device=device, resolution=resolution, training_steps=training_steps,
                                sampler=sampler, model=models[i], optimizer=optimizers[i], prog_iter=i)
            losses_all_progressive_iters.append(losses)
            # write volume
            onePSNR = decode_and_calculate_PSNR(device_name=device_name, resolution=resolution,
                                    sampler=sampler, model=models[i], path_name=decomp_path_name, datatype=np.float32, 
                                    old_diff_max=volume_max, old_diff_min=volume_min, prog_iter=i)
        # use sperr for later iterations
        # be careful of value range
        else:
            temp_path = "temp_compr_output.bit"
            train_volume_with_SPERR(float_type=32, resolution=resolution, output_path=temp_path, target_PSNR=35, input_path=residual_path_name)
            decode_volume_with_SPERR(output_path=decomp_path_name, input_path=temp_path)
        
        if i == 0:
            tmp = read_volume(file=decomp_path_name, shape=args.dims, dtype="float32")
            accumulate_volume = tmp
        else:
            tmp = read_volume(file=decomp_path_name, shape=args.dims, dtype="float32")
            accumulate_volume += tmp
        # accumulate_volume = accumulate_volume.clip(0.0, 1.0)
        residual_volume = (original_volume - accumulate_volume)
        # import pdb; pdb.set_trace()
        # record the value range of residual_volume for later use (used for denormalizing in next iteration),
        # and store as binary file
        volume_max = residual_volume.max()
        volume_min = residual_volume.min()
        residual_path_name = "volume_progress_"+str(i)+"_residual.bin"
        write_volume(file=residual_path_name, volume=residual_volume, dtype="float32")
        
        # update sampler
        sampler = spl.create_sampler("structuredRegular", "openvkl", filename=residual_path_name,
                                     dims=args.dims, dtype="float32", n_channels=n_channels)
    
    # track losses with tensorboard
    if track_loss:
        for i in range(training_steps):
            losses_for_tensorboard = dict()
            for j in range(len(losses_all_progressive_iters)):
                losses_for_tensorboard["prog_iter_" + str(j)] = losses_all_progressive_iters[j][i]
            writer.add_scalars("Loss/train", losses_for_tensorboard, i)
    
        writer.flush()
        writer.close()
    
    
def train_volume_one_time(device_name, device, resolution, training_steps, sampler, model, optimizer, prog_iter):
            
    prev_time = time.perf_counter()

    batch_size = 2**16
    interval = 10
    
    losses = []

    print(f"Beginning optimization with {training_steps} training steps.")

    for i in range(training_steps):
        
        coords, targets = spl.sample(sampler, batch_size)
        coords = coords.to(device_name)
        targets = targets.to(device_name)
        if prog_iter > 0:
            # normalize targets from 0~1 to -1~1
            targets = normalize_to_neg_one_and_one(targets, sampler_max, sampler_min)
        
        # version complying with the coordinates generated by dvnr sampler
        output = model(coords)
        # adjust the output size to align with the target size
        # output = output.view(-1)
        relative_l2_error = (output - targets.to(output.dtype)) ** 2 / (
            output.detach() ** 2 + 0.01
        )

        # total loss
        loss = relative_l2_error.mean()
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % interval == 0 or i == (training_steps - 1) or i % 1000 == 0:
            loss_val = loss.item()
            # torch.xpu.synchronize()
            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - prev_time
            print(f"Step#{i}: loss={loss_val} time={int(elapsed_time*1000000)}[µs]")

            # Ignore the time spent saving the image
            prev_time = time.perf_counter()

            if i > 0 and interval < training_steps:
                interval *= 10
    
    # print("==================================================")
    return losses

def train_volume_with_SPERR(float_type, resolution, output_path, target_PSNR, input_path):
    
    try:
        result = subprocess.run(["/home/kctung/Projects/SPERR/build/bin/sperr3d", "-c", "--ftype", str(float_type),
                        "--dims", str(resolution[0]), str(resolution[1]), str(resolution[2]), "--bitstream", output_path,
                        "--print_stats", "--psnr", str(target_PSNR), input_path
                        ], capture_output=True, text=True, check=True)
        print("SPERR result:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def decode_volume_with_SPERR(output_path, input_path):
    
    try:
        result = subprocess.run(["/home/kctung/Projects/SPERR/build/bin/sperr3d", "-d",
                        "--decomp_f", output_path, input_path
                        ], capture_output=True, text=True, check=True)
        print("SPERR result:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    

if __name__ == "__main__":
    main()