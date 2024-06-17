import numpy as np
import os
import argparse
import commentjson as json
import numpy as np
import os
import sys
import torch
import intel_extension_for_pytorch
import time
import torch.nn as nn

def read_volume(file, shape, dtype=np.uint8):
    """
    Reads volume data from a .raw file.

    Args:
        file (str): Path to the .raw file.
        shape (tuple): Shape of the volume (depth, height, width).
        dtype (data-type): Desired data-type for the array.

    Returns:
        numpy.ndarray: The volume data.
    """
    with open(file, "rb") as f:
        volume = np.frombuffer(f.read(), dtype=dtype)
        # cast volume data into float32 and reshape
        volume = volume.astype(np.float32).reshape(shape)
    return volume

def write_volume(file, volume, dtype=np.uint8):
    if os.path.splitext(file)[1] == ".raw":
        # with open(file, "wb") as f:
            # Write dimensions (depth, height, width)
            # f.write(struct.pack("iii", volume.shape[0], volume.shape[1], volume.shape[2]))
            # Write the data
        converted_volume = volume.astype(dtype)
        converted_volume.tofile(file)
    else:
        raise ValueError("Unsupported file extension. Only .raw files are supported.")

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


if __name__ == "__main__":
    volume_oringin = read_volume("data/images/bonsai.raw", (256, 256, 256), np.uint8)
    volume_1000 = read_volume("1000.raw", (256, 256, 256), np.uint8)
    # print(volume_oringin.max())
    # test 1
    frenquency = [0] * 256
    for i in range(256):
        for j in range(256):
            for k in range(256):
                frenquency[int(volume_1000[i][j][k])] += 1
                
    for i in range(256):
        print(frenquency[i], end=" ")
    print("")
    
    # test 2
    # device = torch.device("xpu")
    # image = Image("data/images/bonsai.raw", (256, 256, 256), device)
    
    # # Variables for saving/displaying image results
    # # resolution = image.data.shape[0:3]
    # # img_shape = resolution
    # # n_pixels = resolution[0] * resolution[1] * resolution[2]

    # # half_dx = 0.5 / resolution[0]
    # # half_dy = 0.5 / resolution[1]
    # # half_dz = 0.5 / resolution[2]
    # # print("half begin: ", half_dx, " half end: ", 1 - half_dx)
    # # xs = torch.linspace(half_dx, 1 - half_dx, resolution[0], device=device)
    # # ys = torch.linspace(half_dy, 1 - half_dy, resolution[1], device=device)
    # # zs = torch.linspace(half_dz, 1 - half_dz, resolution[2], device=device)
    # # xv, yv, zv = torch.meshgrid([xs, ys, zs])

    # # xyz = torch.stack((xv.flatten() ,yv.flatten(), zv.flatten())).t()
    
    # # Generate coordinates
    # x = torch.arange(256).to("xpu")
    # y = torch.arange(256).to("xpu")
    # z = torch.arange(256).to("xpu")

    # # Create the grid using meshgrid
    # xv, yv, zv = torch.meshgrid([x, y, z])

    # # Stack the coordinates along the last dimension and reshape
    # # xv, yv, zv / xv, zv, yv / yv, xv, zv / yv, zv, xv / zv, xv, yv / zv, yv, xv
    # coordinates = torch.stack((xv.flatten() ,yv.flatten(), zv.flatten())).t()
    # # print(coordinates)
    # coordinates = coordinates.float() / 255.0
    # # print("after")
    # # print(coordinates)
    
    # print("coordinates")
    # print(coordinates)
    # img_inf = image(coordinates)
    
    # for item, compare in zip(img_inf[2000000:2002000], volume_oringin.flatten()[2000000:2002000]):
    #     print(item.item(), " ", compare, "/", end=" ")
    # print("")
    
    # # frenquency = [0] * 256
    # # for item in img_inf:
    # #     frenquency[round(item)] += 1
                
    # # for i in range(256):
    # #     print(frenquency[i], end=" ")
    # # print("")