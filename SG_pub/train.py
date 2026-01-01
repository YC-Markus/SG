import os
import sys
import time
import argparse
import logging
import hashlib
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms.v2.functional as vf
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import utils


from fused_ssim import fused_ssim
from lpips import LPIPS

from models.SG import SG
from IXI_dataset import IXIDataset


NUM_WORKERS     = 8
BATCH_SIZE      = 4
MAX_EPOCH       = 50
DEVICE_ID       = 5
DEVICE          = torch.device(f"cuda:{DEVICE_ID}")

LR              = 1e-4
LR_STEP_SIZE    = 40
LR_GAMMA        = 0.1
WEIGHT_DECAY    = 0.0
UPSCALE         = 4

LOG_DIR         = Path("/data/SG/result_IXI")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE        = LOG_DIR / "test.log"
MODEL_DIR       = LOG_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("train-log")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE, mode="a")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



@torch.no_grad()
def validate(model, val_loader, lpips_metric):
    model.eval()
    psnr_total, ssim_total, lpips_total = 0.0, 0.0, 0.0
    n_batch = 0

    for batch_idx, batch in enumerate(val_loader):
        batch_tmp = torch.cat([batch['T2'], batch['T1']], dim=1).to(DEVICE)
        hr, ref = batch_tmp[:,0:1,:,:], batch_tmp[:,1:2,:,:]
        lr = utils.downsample_via_fft(batch_tmp[:,0:1,:,:], scale_factor=4)
        pred = model(lr, ref)[0]
        img_pred  = pred.clamp_(0, 1).to(DEVICE)
        img_gt    = hr.clamp_(0, 1).to(DEVICE)

        psnr_val  = utils.psnr_batch(img_pred, img_gt)
        ssim_val  = fused_ssim(img_pred, img_gt).squeeze().mean().item()
        lpips_val = lpips_metric(img_pred, img_gt).squeeze().mean().item()

        psnr_total  += psnr_val
        ssim_total  += ssim_val
        lpips_total += lpips_val
        n_batch     += 1
    return psnr_total / n_batch, ssim_total / n_batch, lpips_total / n_batch


def get_dataloaders():
    train_set = IXIDataset(h5_paths=['mri_data_part_00.h5', 'mri_data_part_01.h5', 'mri_data_part_02.h5',
                                 'mri_data_part_03.h5', 'mri_data_part_04.h5', 'mri_data_part_05.h5',
                                 'mri_data_part_06.h5', 'mri_data_part_07.h5'], slice_percentage=0.85)
    val_set = IXIDataset(h5_paths=['mri_data_part_08.h5'], slice_percentage=0.85)
    test_set = IXIDataset(h5_paths=['mri_data_part_09.h5'], slice_percentage=0.85)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_every_epoch",
        action="store_true",
        default=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(DEVICE_ID)
    logger.info(f"Using device: {DEVICE}")
    train_loader, val_loader = get_dataloaders()
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = SG(
        colors=1, 
        dim=[128], 
        block_num=[4], 
        heads=[4], 
        qk_dim=[128],
        mlp_ratio=[2], 
        local_size=[[6, 8, 8, 6]],
        n_segments=[[256, 256, 256, 256]],
        upscale=4,
    ).to(DEVICE)
    model.to(DEVICE)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True



    optimizer  = torch.optim.AdamW(params=model.parameters(), lr=LR)
    scheduler  = MultiStepLR(optimizer, milestones=[LR_STEP_SIZE], gamma=LR_GAMMA)
    lpips_metric = LPIPS(net='alex').to(DEVICE).eval()

    best_ssim = -1.0
    global_step = 0
    augmentation_pipeline = utils.augmentation_create()
    loss_func = torch.nn.L1Loss()
    for epoch in range(1, MAX_EPOCH + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(enumerate(train_loader, start=1),
                    total=len(train_loader),
                    desc=f"Epoch {epoch}/{MAX_EPOCH}",
                    ncols=100)

        for batch_idx, batch in pbar:
            optimizer.zero_grad()

            with torch.no_grad():
                batch_tmp = torch.cat([batch['T2'], batch['T1']], dim=1).to(DEVICE)
                batch_augmented = augmentation_pipeline(batch_tmp)

                lr = utils.downsample_via_fft(batch_augmented[:,0:1,:,:], scale_factor=4)
                hr, ref = batch_augmented[:,0:1,:,:], batch_augmented[:,1:2,:,:]

            pred = model(lr, ref)
            loss = loss_func(pred, hr)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1


        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"[TRAIN] Epoch {epoch} | Avg L1 Loss: {avg_loss:.6f}")

        psnr_mean, ssim_mean, lpips_mean = validate(model, val_loader, lpips_metric)
        val_msg = f"[VAL] Epoch {epoch} | PSNR: {psnr_mean:.4f}, SSIM: {ssim_mean:.4f}, LPIPS: {lpips_mean:.4f}"
        logger.info(val_msg)

        save_path_best = MODEL_DIR / "best.pth"
        if ssim_mean > best_ssim:
            best_ssim = ssim_mean
            torch.save(model.state_dict(), save_path_best)
            logger.info(f"★ New best SSIM ({best_ssim:.4f}) → save to {save_path_best}")

        if args.save_every_epoch:
            save_path_epoch = MODEL_DIR / f"epoch_{epoch:03d}.pth"
            torch.save(model.state_dict(), save_path_epoch)

    logger.info("Training finished.")


if __name__ == "__main__":
    main()