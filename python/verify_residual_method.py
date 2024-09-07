import numpy as np
import torch
from common import read_image, write_image, ROOT_DIR, read_volume, write_volume


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
    original_volume = read_volume("data/images/1atm.H2O.3x.1152.320.853f32.bin", resolution, np.float32)
    original_volume = normalize_to_neg_one_and_one(original_volume, original_volume.max(), original_volume.min())
    # original_volume_max = original_volume.max()
    # original_volume_min = original_volume.min()
    # # normalize original volume
    # original_volume = (original_volume - original_volume_min) / (original_volume_max - original_volume_min) * 1.0
    
    progressive_iter = 8
    for i in range(progressive_iter):
        print("iter: ", i)
        vol_whole = read_volume("whole_volume_"+str(i)+".bin", resolution, np.float32)
        vol_decompressed = read_volume("volume_progress_"+str(i)+"_decompressed.bin", resolution, np.float32)
        # import pdb; pdb.set_trace()
        if i > 0:
            # normalize vol_decompressed and vol_residual for calculating PSNR and loss
            # vol_decompressed = (vol_decompressed - vol_residual.min()) / (vol_residual.max() - vol_residual.min())
            # vol_residual = (vol_residual - vol_residual.min()) / (vol_residual.max() - vol_residual.min())
            vol_decompressed = normalize_to_neg_one_and_one(vol_decompressed, vol_residual.max(), vol_residual.min())
            vol_residual = normalize_to_neg_one_and_one(vol_residual, vol_residual.max(), vol_residual.min())
            print("residual (ground truth) max and min value: ", vol_residual.max(), " ", vol_residual.min())
            print("trained model's inference output max and min value:", vol_decompressed.max(), " ", vol_decompressed.min())
            PSNR = calculate_PSNR_from_squared_errors_sum(
                accumulate_squared_errors_of_slice(vol_decompressed - vol_residual), resolution)
            relative_l2_error = (vol_decompressed - vol_residual) ** 2 / (
                vol_decompressed ** 2 + 0.01
            )
            print("PSNR and loss between residual errors and trained results: PSNR: ",
                  PSNR.item(), " loss: ", relative_l2_error.mean())
        # vol_residual would be the ground truth of next level's vol_decompressed
        vol_residual = read_volume("volume_progress_"+str(i)+"_residual.bin", resolution, np.float32)
        verify_vol = vol_whole + vol_residual
        PSNR= calculate_PSNR_from_squared_errors_sum(
            accumulate_squared_errors_of_slice(verify_vol - original_volume), resolution
        )
        relative_l2_error = (verify_vol - original_volume) ** 2 / (
                verify_vol ** 2 + 0.01
        )
        # verify whether vol_whole + vol_residual equals to original_volume (ground truth)
        print("PSNR and loss between original volume and accumulated volume + residual volume: PSNR",
              PSNR.item(), " loss: ", relative_l2_error.mean())
if __name__ == "__main__":
    main()