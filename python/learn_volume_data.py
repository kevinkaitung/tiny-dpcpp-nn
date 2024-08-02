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
from MLP_native import MLP_Native

import dvnr_sampler as spl

DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

class Image(torch.nn.Module):
    def __init__(self, filename, shape, dtype, device):
        super(Image, self).__init__()
        self.data = read_volume(filename, shape, dtype=dtype)
        self.max_function_value = self.data.max()
        self.data /= self.max_function_value
        self.shape = self.data.shape
        self.data = torch.from_numpy(self.data).float().to(device)

    def forward(self, xs):
        with torch.no_grad():
            # Bilinearly filtered lookup from the image. Not super fast,
            # but less than ~20% of the overall runtime of this example.
            shape = self.shape

            xs = xs * torch.tensor([shape[0] - 1, shape[1] - 1, shape[2] - 1], device=xs.device).float()
            indices = xs.long()
            lerp_weights = xs - indices.float()

            x0 = indices[:, 0].clamp(min=0, max=shape[0] - 1)
            y0 = indices[:, 1].clamp(min=0, max=shape[1] - 1)
            z0 = indices[:, 2].clamp(min=0, max=shape[2] - 1)
            x1 = (x0 + 1).clamp(max=shape[0] - 1)
            y1 = (y0 + 1).clamp(max=shape[1] - 1)
            z1 = (z0 + 1).clamp(max=shape[2] - 1)

            c000 = self.data[x0, y0, z0]
            c010 = self.data[x0, y1, z0]
            c100 = self.data[x1, y0, z0]
            c110 = self.data[x1, y1, z0]
            c001 = self.data[x0, y0, z1]
            c011 = self.data[x0, y1, z1]
            c101 = self.data[x1, y0, z1]
            c111 = self.data[x1, y1, z1]

            # Trilinear interpolation
            return ((1 - lerp_weights[:,0]) * (1 - lerp_weights[:,1]) * (1 - lerp_weights[:,2]) * c000
                +   (1 - lerp_weights[:,0]) *      lerp_weights[:,1] *  (1 - lerp_weights[:,2]) * c010
                +        lerp_weights[:,0] *  (1 - lerp_weights[:,1]) * (1 - lerp_weights[:,2]) * c100
                +        lerp_weights[:,0] *       lerp_weights[:,1] *  (1 - lerp_weights[:,2]) * c110
                +   (1 - lerp_weights[:,0]) * (1 - lerp_weights[:,1]) *      lerp_weights[:,2] * c001
                +   (1 - lerp_weights[:,0]) *      lerp_weights[:,1] *       lerp_weights[:,2] * c011
                +        lerp_weights[:,0] *  (1 - lerp_weights[:,1]) *      lerp_weights[:,2] * c101
                +        lerp_weights[:,0] *       lerp_weights[:,1] *       lerp_weights[:,2] * c111)


def get_args():
    parser = argparse.ArgumentParser(
        description="Image benchmark using PyTorch bindings."
    )
    # for dvnr volume sampler
    parser.add_argument(
        '--filename', type=str, default="data/images/bonsai.raw", help="volume data file"
    )
    parser.add_argument(
        "--dims", type=int, nargs=3, default=[256, 256, 256], help="volume data dimensions"
    )
    parser.add_argument(
        "--type", type=str, default="uint8", help="volume data type"
    )
    # for original image sampler
    parser.add_argument(
        "image", nargs="?", default="data/images/bonsai.raw", help="Image to match"
    )
    parser.add_argument(
        "shape", nargs="?", default=(256, 256, 256), help="Image Shape"
    )
    parser.add_argument(
        "data_type", nargs="?", default=np.uint8, help="Image Data Type"
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

def calculate_PSNR(orginal_vol, compressed_vol, max_function_value):
    MSE = ((orginal_vol - compressed_vol) **2).mean()
    return 20 * torch.log10(max_function_value / torch.sqrt(MSE))

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

    image = Image(args.image, args.shape, args.data_type, device)
    n_channels = 1
    # sampler = spl.create_sampler("structuredRegular", "xpu", filename=args.filename, dims=args.dims, dtype=args.dtype, n_channels=n_channels)
    sampler = spl.create_sampler("structuredRegular", "openvkl", filename=args.filename, dims=args.dims, dtype=args.type, n_channels=n_channels)
    # n_channels = image.data.shape[3]

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

    encoding = tnn.Encoding(
        n_input_dims=3,
        encoding_config=config["encoding"],
        dtype=torch.float,
    )
    # encoding = HashEmbedderNative(n_pos_dims=2, encoding_config=config["encoding"])
    network = tnn.Network(
        n_input_dims=encoding.n_output_dims,
        n_output_dims=n_channels,
        network_config=config["network"],
    )
    # network = MLP_Native(n_input_dims=encoding.n_output_dims, n_output_dims=n_channels, network_config=config["network"])
    model = torch.nn.Sequential(encoding, network).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Variables for saving/displaying image results
    # resolution = args.dims
    resolution = image.data.shape[0:3]
    img_shape = resolution
    n_pixels = resolution[0] * resolution[1] * resolution[2]

    half_dx = 0.5 / resolution[0]
    half_dy = 0.5 / resolution[1]
    half_dz = 0.5 / resolution[2]
    xs = torch.linspace(half_dx, 1 - half_dx, resolution[0], device=device)
    ys = torch.linspace(half_dy, 1 - half_dy, resolution[1], device=device)
    zs = torch.linspace(half_dz, 1 - half_dz, resolution[2], device=device)
    xv, yv, zv = torch.meshgrid([xs, ys, zs])

    xyz = torch.stack((xv.flatten() ,yv.flatten(), zv.flatten())).t()
    
    print("xyz a:", xyz.shape)
    print(xyz)
    
    # # Generate coordinates of regular grid
    # x = torch.arange(256).to("xpu")
    # y = torch.arange(256).to("xpu")
    # z = torch.arange(256).to("xpu")

    # # Create the grid using meshgrid
    # xv, yv, zv = torch.meshgrid([x, y, z])

    # # Stack the coordinates along the last dimension and reshape
    # # xv, yv, zv / xv, zv, yv / yv, xv, zv / yv, zv, xv / zv, xv, yv / zv, yv, xv
    # xyz = torch.stack((xv.flatten() ,yv.flatten(), zv.flatten())).t()
    # xyz = xyz.float() / 255.0
    # print("xyz b:", xyz.shape)
    # print(xyz)
    
    prev_time = time.perf_counter()

    batch_size = 2**16
    interval = 10

    print(f"Beginning optimization with {args.n_steps} training steps.")

    # try:
    #     batch = torch.rand([batch_size, 3], device=device, dtype=torch.float32)
    #     traced_image = torch.jit.trace(image, batch)
    # except:
    #     # If tracing causes an error, fall back to regular execution
    #     print(
    #         f"WARNING: PyTorch JIT trace failed. Performance will be slightly worse than regular."
    #     )
    #     traced_image = image

    for i in range(args.n_steps):
        # batch = torch.rand([batch_size, 3], device=device, dtype=torch.float32).to(
        #     device_name
        # )
        # targets = traced_image(batch)
        # output = model(batch)
        coords, targets = spl.sample(sampler, batch_size)
        coords = coords.to(device_name)
        targets = targets.to(device_name)
        output = model(coords)

        # adjust the output size to align with the target size
        # targets = targets.view(-1)
        # output = output.view(-1)
        # print("targets shape: ", targets.shape, " output shape: ", output.shape)
        relative_l2_error = (output - targets.to(output.dtype)) ** 2 / (
            output.detach() ** 2 + 0.01
        )

        loss = relative_l2_error.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % interval == 0:
            loss_val = loss.item()
            torch.xpu.synchronize()
            elapsed_time = time.perf_counter() - prev_time
            print(f"Step#{i}: loss={loss_val} time={int(elapsed_time*1000000)}[µs]")

            path = f"{i}.raw"
            print(f"Writing '{path}'... ", end="")
            # with torch.no_grad():
            #     write_volume(
            #         path, 
            #         model(xyz).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy() * image.max_function_value,
            #         dtype=args.data_type
            #     )
            print("done.")

            # Ignore the time spent saving the image
            prev_time = time.perf_counter()

            if i > 0 and interval < 1000:
                interval *= 10
        # print("==================================================")
    if args.result_filename:
        print(f"Writing '{args.result_filename}'... ", end="")
        # with torch.no_grad():
        #     write_volume(
        #         args.result_filename,
        #         model(xyz).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy() * image.max_function_value,
        #         dtype=args.data_type
        #     )
        print("done.")
        
        # Generate coordinates of regular grid
        # x = torch.arange(args.dims[0]).to("xpu")
        for x in range(args.dims[0]):
            x_coord = torch.full((args.dims[1] * args.dims[2], 1), x, device=device, dtype=torch.float32) / (args.dims[0] - 1)
            y = torch.arange(args.dims[1], device=device, dtype=torch.float32) / (args.dims[1] - 1)
            z = torch.arange(args.dims[2], device=device, dtype=torch.float32) / (args.dims[2] - 1)

            # Create the grid using meshgrid
            yv, zv = torch.meshgrid([y, z])

            # Stack the coordinates along the last dimension and reshape
            # xv, yv, zv / xv, zv, yv / yv, xv, zv / yv, zv, xv / zv, xv, yv / zv, yv, xv
            yz = torch.stack((yv.flatten(), zv.flatten())).t()
            
            xyz = torch.cat((x_coord, yz), dim=1)
            
            
            with open("1000.raw", "ab") as f:
                model(xyz).reshape([args.dims[1], args.dims[2]]).clamp(0.0, 1.0).detach().cpu().numpy().astype(args.type).tofile(f)


if __name__ == "__main__":
    main()