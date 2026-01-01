import torch
import torch.nn.functional as F
import torch.fft
import torchvision.transforms.v2 as v2
from torchvision.transforms import InterpolationMode


def psnr_batch(img_out, img_gt):
    mse = F.mse_loss(img_out, img_gt, reduction="none").mean(dim=[1, 2, 3])
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.mean().item()


def downsample_via_fft(images: torch.Tensor, scale_factor: int = 4) -> torch.Tensor:
    B, C, H, W = images.shape
    new_W = W // scale_factor
    new_H = H // scale_factor

    fft_images = torch.fft.fft2(images, dim=(-2, -1), norm="forward")
    shifted_fft = torch.fft.fftshift(fft_images, dim=(-2, -1))

    start_H = (H // 2) - (new_H // 2)
    end_H = start_H + new_H
    start_W = (W // 2) - (new_W // 2)
    end_W = start_W + new_W

    cropped_shifted_fft = shifted_fft[..., start_H:end_H, start_W:end_W]
    cropped_fft = torch.fft.ifftshift(cropped_shifted_fft, dim=(-2, -1))

    ifft_result = torch.fft.ifft2(cropped_fft, dim=(-2, -1), norm="forward")
    low_res_images = ifft_result.abs()
    return low_res_images


def augmentation_create():
    interp_mode = InterpolationMode.BILINEAR

    mri_augmentation_pipeline = v2.Compose([
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply(
            [v2.RandomChoice([
                v2.RandomRotation(degrees=(90, 90), interpolation=interp_mode, fill=0),
                v2.RandomRotation(degrees=(180, 180), interpolation=interp_mode, fill=0),
                v2.RandomRotation(degrees=(270, 270), interpolation=interp_mode, fill=0),
            ])],
            p=0.75
        ),
    ])
    return mri_augmentation_pipeline
