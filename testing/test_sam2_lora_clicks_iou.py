#!/usr/bin/env python3
"""
Test SAM2 with LoRA and plot Clicks vs mIoU curve
Plots the relationship between number of clicks and mean IoU across validation set
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm

from sam2.build_sam import build_sam2_with_lora
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2_simpleclick_adapter import SAM2ClickerAdapter
from hydra.core.global_hydra import GlobalHydra


def register_omegaconf_resolvers():
    """Register custom OmegaConf resolvers"""
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    OmegaConf.register_new_resolver("negate", lambda x: -x, replace=True)
    OmegaConf.register_new_resolver("round", lambda x: int(round(x)), replace=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Test SAM2 and plot Clicks vs mIoU curve")

    # Model arguments
    parser.add_argument('--config', type=str, default='configs/sam2.1/sam2.1_hiera_l.yaml',
                        help='Path to SAM2 config file')
    parser.add_argument('--checkpoint', type=str,
                        default='output_1112_sam2.1_hiera_l_hels_finetune+lora/checkpoints/checkpoint_400.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on')

    # LoRA arguments
    parser.add_argument('--use-lora', action='store_true', default=True,
                        help='Use LoRA model')
    parser.add_argument('--lora-rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                        help='LoRA dropout')
    parser.add_argument('--lora-target-modules', type=str, default='qkv,proj',
                        help='LoRA target modules (comma separated)')

    # Test data arguments
    parser.add_argument('--image-dir', type=str,
                        default="/opt/data/private/lls/HLES-SAM/data/CVPR2026/image",
                        help='Directory containing test images')
    parser.add_argument('--mask-dir', type=str,
                        default="/opt/data/private/lls/HLES-SAM/data/CVPR2026/ground_truth",
                        help='Directory containing ground truth masks')
    parser.add_argument('--image-ext', type=str, default='.png,.jpg,.jpeg',
                        help='Image file extensions (comma separated)')
    parser.add_argument('--mask-ext', type=str, default='.png',
                        help='Mask file extension')

    # Evaluation arguments
    parser.add_argument('--max-clicks', type=int, default=20,
                        help='Maximum number of clicks (default: 20)')
    parser.add_argument('--pred-threshold', type=float, default=0.49,
                        help='Prediction threshold for binary mask')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='testing/output_clicks_iou',
                        help='Output directory for results')
    parser.add_argument('--save-data', action='store_true', default=True,
                        help='Save raw data to JSON')

    args = parser.parse_args()
    return args


def load_sam2_model(args):
    """Load SAM2 model with optional LoRA"""
    device = torch.device(args.device)

    # Clear any existing Hydra instance
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Register custom resolvers
    register_omegaconf_resolvers()

    if args.use_lora:
        print("Loading SAM2 model with LoRA...")
        lora_target_modules = tuple(args.lora_target_modules.split(','))

        sam2_model = build_sam2_with_lora(
            config_file=args.config,
            ckpt_path=args.checkpoint,
            device=device,
            mode="eval",
            apply_postprocessing=True,
            lora_rank=args.lora_rank,
            lora_dropout=args.lora_dropout,
            lora_target_modules=lora_target_modules,
        )

        print(f" SAM2 model loaded with LoRA (rank={args.lora_rank}, "
              f"dropout={args.lora_dropout}, target_modules={lora_target_modules})")

    return sam2_model


def load_image_and_mask(image_path, mask_path):
    """Load test image and ground truth mask"""
    # Load image
    image = Image.open(image_path)
    if image.mode == 'L':  # Grayscale
        image = image.convert('RGB')
    image = np.array(image)

    # Load mask
    mask = Image.open(mask_path)
    # Convert to grayscale if needed (ensures single channel)
    if mask.mode != 'L':
        mask = mask.convert('L')
    mask = np.array(mask)

    # Convert to binary mask (0 or 1)
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)

    return image, mask


def get_image_mask_pairs(args):
    """Get list of (image_path, mask_path) pairs"""
    pairs = []

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)

    # Get all image files
    image_exts = [ext.strip() for ext in args.image_ext.split(',')]
    image_files = []
    for ext in image_exts:
        image_files.extend(list(image_dir.glob(f'*{ext}')))

    # Match with masks
    for image_path in sorted(image_files):
        mask_path = mask_dir / f"{image_path.stem}{args.mask_ext}"
        if mask_path.exists():
            pairs.append((image_path, mask_path))
        else:
            print(f"Warning: No mask found for {image_path.name}")

    return pairs


def evaluate_all_images(args, predictor):
    """
    Evaluate all images and collect IoU at each click

    Returns:
        all_ious: List of IoU lists, one per image. Each inner list contains IoU at each click.
    """
    # Get image-mask pairs
    pairs = get_image_mask_pairs(args)
    print(f"\nFound {len(pairs)} image-mask pairs")

    # Create adapter
    adapter = SAM2ClickerAdapter(predictor, pred_threshold=args.pred_threshold)

    # Store all IoU trajectories
    all_ious = []
    failed_images = []

    print("\n" + "=" * 80)
    print("Running Evaluation...")
    print("=" * 80)

    # Process each image
    for image_path, mask_path in tqdm(pairs, desc="Processing images"):
        try:
            # Load data
            image, gt_mask = load_image_and_mask(image_path, mask_path)

            # Run iterative prediction
            ious_list, final_mask, clicks_list = adapter.iterative_predict(
                image=image,
                gt_mask=gt_mask,
                max_clicks=args.max_clicks,
                target_iou=1.0,  # No early stopping
                multimask_output=True
            )

            # Store IoU trajectory
            all_ious.append(ious_list)

        except Exception as e:
            print(f"\nError processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            failed_images.append(image_path.name)
            continue

    if failed_images:
        print(f"\n{len(failed_images)} images failed: {failed_images}")

    return all_ious


def compute_mean_iou_per_click(all_ious, max_clicks):
    """
    Compute mean IoU at each click number across all images

    Args:
        all_ious: List of IoU lists (one per image)
        max_clicks: Maximum number of clicks

    Returns:
        clicks: Array of click numbers [1, 2, 3, ..., max_clicks]
        mean_ious: Array of mean IoU at each click
        std_ious: Array of std IoU at each click
    """
    # Initialize arrays to store IoUs at each click
    ious_per_click = [[] for _ in range(max_clicks)]

    # Collect IoUs at each click number
    for iou_list in all_ious:
        for click_idx, iou in enumerate(iou_list):
            if click_idx < max_clicks:
                ious_per_click[click_idx].append(iou)

    # Compute mean and std at each click
    clicks = np.arange(1, max_clicks + 1)
    mean_ious = []
    std_ious = []

    for click_idx in range(max_clicks):
        if len(ious_per_click[click_idx]) > 0:
            mean_ious.append(np.mean(ious_per_click[click_idx]))
            std_ious.append(np.std(ious_per_click[click_idx]))
        else:
            mean_ious.append(np.nan)
            std_ious.append(np.nan)

    return clicks, np.array(mean_ious), np.array(std_ious)


def plot_clicks_vs_iou(clicks, mean_ious, std_ious, output_path):
    """
    Plot Clicks vs mIoU curve with error bars

    Args:
        clicks: Array of click numbers
        mean_ious: Array of mean IoU at each click
        std_ious: Array of std IoU at each click
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    # Plot mean IoU with error bars (std)
    plt.plot(clicks, mean_ious, 'b-o', linewidth=2, markersize=6, label='Mean IoU')
    plt.fill_between(clicks,
                     mean_ious - std_ious,
                     mean_ious + std_ious,
                     alpha=0.2,
                     color='blue',
                     label='± 1 std')

    # Styling
    plt.xlabel('Number of Clicks', fontsize=14)
    plt.ylabel('Mean IoU', fontsize=14)
    plt.title('Interactive Segmentation Performance: Clicks vs mIoU', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12)

    # Set y-axis limits
    plt.ylim([0, 1.0])

    # Add text box with final statistics
    final_miou = mean_ious[-1]
    final_std = std_ious[-1]
    textstr = f'Final mIoU@{len(clicks)}: {final_miou:.4f} ± {final_std:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n Plot saved to: {output_path}")
    plt.close()


def save_data(clicks, mean_ious, std_ious, all_ious, output_path):
    """Save raw data to JSON"""
    data = {
        'clicks': clicks.tolist(),
        'mean_ious': mean_ious.tolist(),
        'std_ious': std_ious.tolist(),
        'num_images': len(all_ious),
        'all_ious': [[float(iou) for iou in iou_list] for iou_list in all_ious]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f" Data saved to: {output_path}")


def print_statistics(clicks, mean_ious, std_ious):
    """Print statistics table"""
    print("\n" + "=" * 80)
    print("Clicks vs mIoU Statistics")
    print("=" * 80)
    print(f"{'Clicks':<10} {'Mean IoU':<15} {'Std IoU':<15} {'Improvement':<15}")
    print("-" * 80)

    for i, (click, miou, std) in enumerate(zip(clicks, mean_ious, std_ious)):
        if i == 0:
            improvement = "-"
        else:
            improvement = f"+{(miou - mean_ious[i-1]):.4f}"

        print(f"{int(click):<10} {miou:.6f}{'':>8} {std:.6f}{'':>8} {improvement:<15}")

    print("=" * 80)


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("SAM2 Interactive Segmentation: Clicks vs mIoU Analysis")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Max clicks: {args.max_clicks}")
    print(f"Output: {args.output_dir}")

    # Load model
    sam2_model = load_sam2_model(args)
    predictor = SAM2ImagePredictor(sam2_model)

    # Evaluate all images
    all_ious = evaluate_all_images(args, predictor)

    if len(all_ious) == 0:
        print("\nError: No images were successfully processed!")
        return

    print(f"\n Successfully processed {len(all_ious)} images")

    # Compute mean IoU at each click
    clicks, mean_ious, std_ious = compute_mean_iou_per_click(all_ious, args.max_clicks)

    # Print statistics
    print_statistics(clicks, mean_ious, std_ious)

    # Plot results
    plot_path = output_dir / 'clicks_vs_miou.png'
    plot_clicks_vs_iou(clicks, mean_ious, std_ious, plot_path)

    # Save raw data
    if args.save_data:
        data_path = output_dir / 'clicks_vs_miou_data.json'
        save_data(clicks, mean_ious, std_ious, all_ious, data_path)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
