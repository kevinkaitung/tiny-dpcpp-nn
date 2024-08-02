import argparse
import commentjson as json
import numpy as np
import os
import sys
import torch
import intel_extension_for_pytorch
import time
import torch.nn as nn
import dvnr_sampler as spl

def main():
    # Generate coordinates of regular grid
    x = torch.arange(256, dtype=torch.float32) / 255
    y = torch.arange(256, dtype=torch.float32) / 255
    z = torch.arange(256, dtype=torch.float32) / 255

    # Create the grid using meshgrid
    zv, yv, xv = torch.meshgrid([z, y, x])

    # Stack the coordinates along the last dimension and reshape
    # xv, yv, zv / xv, zv, yv / yv, xv, zv / yv, zv, xv / zv, xv, yv / zv, yv, xv
    zyx = torch.stack((zv.flatten() ,yv.flatten(), xv.flatten())).t()
    # print(zyx)
    # zyx works for dvnr sampler decode function
    zyx = zyx[:, [2, 1, 0]]
    print("a:", zyx)
    
    # xyz doesn't work for dvnr sampler decode function, although it has identical values as zyx
    zv, yv, xv = torch.meshgrid([z, y, x])
    xyz = torch.stack((xv.flatten() ,yv.flatten(), zv.flatten())).t()
    print("b:", xyz)
    print(xyz.allclose(zyx))
    
    
    # create sampler
    sampler = spl.create_sampler("structuredRegular", "openvkl", filename="data/images/bonsai.raw",
                                 dims=[256, 256, 256], dtype="uint8", n_channels=1)
    
    # query raw data
    decoded_values = torch.zeros([zyx.shape[0], 1]).float()
    spl.decode(sampler, zyx, decoded_values)
    # import pdb; pdb.set_trace()
    
    volume = decoded_values.flatten().reshape([256, 256, 256])
    volume *= 255.0
    print("decode:")
    print(volume[0][0])
    converted_volume = volume.numpy().astype(np.uint8)
    converted_volume.tofile("1000.raw")
    return
    # open original raw data
    with open("data/images/skull.raw", "rb") as f:
        volume_org = np.frombuffer(f.read(), dtype=np.uint8)
    volume_org = volume_org.astype(np.float32).reshape([256, 256, 256])
    print("origin:")
    print(volume_org[0][0])
    return
    # # count the number of each function value
    # frenquency = [0] * 256
    # for item in decoded_values:
    #     # print(item.item())
    #     frenquency[int(item.item()*255)] += 1
    # print("values count from sampler decode")
    # for i in range(256):
    #     print(frenquency[i], end=" ")
    # print("")
    
    
    # print("min max:", volume.min(), " ", volume.max())
    # # count the number of each function value
    # frenquency = [0] * 256
    # for item in volume:
    #     frenquency[int(item)] += 1
    # print("values count from raw data")
    # for i in range(256):
    #     print(frenquency[i], end=" ")
    # print("")
    
    # write volume
    decoded_values *= 255.0
    
    frenquency = [0] * 256
    for item in decoded_values:
        # print(item.item())
        frenquency[int(item.item())] += 1
    print("values count from sampler decode")
    for i in range(256):
        print(frenquency[i], end=" ")
    print("")
    
    volume = decoded_values.reshape([256, 256, 256])
    
    
    converted_volume = volume.numpy().astype(np.uint8)
    converted_volume.tofile("1000.raw")
    
    
if __name__ == "__main__":
    main()