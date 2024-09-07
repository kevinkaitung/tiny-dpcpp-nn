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
    # max value = 2 if value range is -1 ~ 1
    max_value = 2.0
    temp = squared_errors_sum / (resolution[0] * resolution[1] * resolution[2])
    return 20 * torch.log10(max_value / torch.sqrt(torch.tensor(temp)))

def normalize_to_neg_one_and_one(data, original_max, original_min):
    original_range = original_max - original_min
    normalized_max = 1.0
    normalized_min = -1.0
    normalized_range = normalized_max - normalized_min
    return ((data - original_min) / original_range) * normalized_range + normalized_min

def main():
    resolution = [1152, 320, 853]
    # resolution = [256, 256, 256]
    original_volume = read_volume("data/images/1atm.H2O.3x.1152.320.853f32.bin", resolution, np.float32)
    # original_volume = read_volume("data/images/bonsai.raw", resolution, np.uint8)
    original_volume = normalize_to_neg_one_and_one(original_volume, original_volume.max(), original_volume.min())
    # original_volume_max = original_volume.max()
    # original_volume_min = original_volume.min()
    # # normalize original volume
    # original_volume = (original_volume - original_volume_min) / (original_volume_max - original_volume_min) * 1.0
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