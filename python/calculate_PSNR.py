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
from common import read_volume


def calculate_part_PSNR(orginal_vol, compressed_vol, max_val):
    temp = ((orginal_vol - compressed_vol) **2).sum()
    temp /= (orginal_vol.shape[0] * orginal_vol.shape[1] * orginal_vol.shape[2])
    return 20 * torch.log10(max_val / torch.sqrt(torch.tensor(temp)))



def main():
    # for bonsai dataset
    volume_oringin = read_volume("data/images/bonsai.raw", (256, 256, 256), np.uint8, 0)
    volume_1000 = read_volume("1000.raw", (256, 256, 256), np.uint8, 0)
    PSNR = calculate_part_PSNR(volume_oringin, volume_1000, 255.0)
    print(PSNR)
    
    # for chameleon dataset
    # TBD
    
if __name__ == "__main__":
    main()