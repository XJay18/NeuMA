import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torchvision.utils import save_image
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from modules.tune.utils import save_video_mediapy, AverageMeter

def arg_parser():
    parser = argparse.ArgumentParser(description='Calculate image metrics')
    parser.add_argument('--pred_dir', '-p', type=str, help='Path to the directory containing the predicted images')
    parser.add_argument('--gt_dir', '-g', type=str, help='Path to the directory containing the ground truth images')
    parser.add_argument('--start', '-s', type=int, default=0, help='Start index')
    parser.add_argument('--skip', '-k', type=int, default=1, help='Skip index')
    parser.add_argument('--num', '-n', type=int, default=10, help='Number of images to calculate')
    parser.add_argument('--device', '-d', type=str, default='cuda', help='Device to use')
    parser.add_argument('--view', type=int, required=True)
    return parser.parse_args()


def calculate_synthetic_image_metrics(pred_dir, gt_dir, start, skip, num, device='cuda'):
    idx = [start + i * skip for i in range(num + 1)]
    
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = structural_similarity_index_measure
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

    crop_params_x = [220, 580]
    crop_params_y = [220, 580]

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    lpips_meter = AverageMeter()

    os.system('rm -rf results/debug')
    os.system('mkdir -p results/debug')

    print(f'current pred_dir: {pred_dir}, skip: {skip}, from {start} to {start + num * skip}')

    p_bar = tqdm(idx, leave=False, desc="Calculating image metrics for synthetic data ...")
    for i in p_bar:
        pred_path = os.path.join(pred_dir, f'e_{args.view}_{i:03d}.png')
        assert os.path.exists(pred_path), f"File not exist for {pred_path}"

        gt_path = os.path.join(gt_dir, f'e_{args.view}_{i:03d}.png')
        assert os.path.exists(gt_path), f"File not exist for {gt_path}"

        pred_img = Image.open(pred_path).convert('RGB')
        pred_img = np.array(pred_img)
        
        gt_img = Image.open(gt_path)
        gt_img = np.array(gt_img.convert("RGBA"))

        norm_data = gt_img / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + np.array([1, 1, 1]) * (1 - norm_data[:, :, 3:4])
        gt_img = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        gt_img = np.array(gt_img)

        pred_img = pred_img[crop_params_x[0]:crop_params_x[1], crop_params_y[0]:crop_params_y[1], :]
        gt_img = gt_img[crop_params_x[0]:crop_params_x[1], crop_params_y[0]:crop_params_y[1], :]
        
        pred_tensor = transforms.ToTensor()(pred_img).unsqueeze(0).to(device)

        gt_tensor = transforms.ToTensor()(gt_img).unsqueeze(0).to(device)

        psnr_val = float(psnr(gt_tensor, pred_tensor).item())
        ssim_val = float(ssim(gt_tensor, pred_tensor))
        lpips_val = float(lpips(gt_tensor, pred_tensor))

        psnr_meter.update(psnr_val)
        ssim_meter.update(ssim_val)
        lpips_meter.update(lpips_val)

        p_bar.set_description(
            f"pr: {pred_path.split('/')[-1]} [{pred_img.shape}] | "
            f"gt: {gt_path.split('/')[-1]} [{gt_img.shape}] | "
            f"PSNR: {psnr_val:.2f} | "
            f"SSIM: {ssim_val:.2f} | "
            f"LPIPS: {lpips_val:.2f}"
        )

        save_image(torch.cat([pred_tensor, gt_tensor], dim=0), f'results/debug/{i}.png', nrow=2)
        
    save_path = os.path.join(args.pred_dir, '..', f'{pred_dir.split("/")[-1]}_metrics.txt')
    video_path = os.path.join(args.pred_dir, '..', f'{pred_dir.split("/")[-1]}_pred-gt.mp4')
    
    with open(save_path, 'w') as f:
        f.write(f"PSNR: {psnr_meter.avg:.2f}\n")
        f.write(f"SSIM: {ssim_meter.avg:.2f}\n")
        f.write(f"LPIPS: {lpips_meter.avg:.4f}\n")

    save_video_mediapy(Path("results/debug"), "*.png", Path(video_path), skip_frame=5, fps=30, white_bg=True)

if __name__ == "__main__":
    args = arg_parser()
    calculate_synthetic_image_metrics(args.pred_dir, args.gt_dir, args.start, args.skip, args.num, args.device)
