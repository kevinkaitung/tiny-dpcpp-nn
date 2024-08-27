from common import read_image, write_image, ROOT_DIR, read_volume, write_volume
import numpy as np
import torch
import torch.nn as nn
from torch.func import stack_module_state, functional_call
from torch import vmap
from torch.utils.tensorboard import SummaryWriter


def accumulate_squared_errors_of_slice(diff_targets_output):
    return ((diff_targets_output) ** 2).sum()

def calculate_PSNR_from_squared_errors_sum(squared_errors_sum, resolution):
    temp = squared_errors_sum / (resolution[0] * resolution[1] * resolution[2])
    return 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(temp)))

def main():
    resolution = [1152, 320, 853]
    # resolution = [256, 256, 256]
    original_volume = read_volume("data/images/1atm.H2O.3x.1152.320.853f32.bin", resolution, np.float32)
    # original_volume = read_volume("data/images/bonsai.raw", resolution, np.uint8)
    original_volume_max = original_volume.max()
    original_volume_min = original_volume.min()
    # normalize original volume
    original_volume = (original_volume - original_volume_min) / (original_volume_max - original_volume_min) * 1.0
    progressive_iters = 8
    for i in range(progressive_iters):
        if i == 0:
            whole_volume = read_volume("volume_progress_" + str(i) + "_decompressed.bin", resolution, np.float32)
        else:
            whole_volume += read_volume("volume_progress_" + str(i) + "_decompressed.bin", resolution, np.float32)
        # whole_volume = whole_volume.clip(0.0, 1.0)
        # import pdb; pdb.set_trace()
        # current PSNR calculation requires normalization between 0~1
        # whole_volume is already normalized
        squared_errors_sum = accumulate_squared_errors_of_slice(original_volume - whole_volume)
        print("iter:", i, " PSNR:", calculate_PSNR_from_squared_errors_sum(squared_errors_sum, resolution))
        
        write_volume("whole_volume_" + str(i) + ".bin", whole_volume, np.float32)
        


if __name__ == "__main__":
    main()