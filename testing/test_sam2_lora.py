"""
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from tqdm import tqdm
import json

from sam2.build_sam import build_sam2_with_lora
from sam2.sam2_image_predictor import SAM2ImagePredictor
from training.utils.train_utils import register_omegaconf_resolvers
from sam2_simpleclick_adapter import SAM2ClickerAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Test SAM2 with SimpleClick iterative strategy")

    # Model arguments
    parser.add_argument('--config', type=str, default='configs/sam2.1/sam2.1_hiera_l.yaml',
                        help='Path to SAM2 config file')
    parser.add_argument('--checkpoint', type=str,
                        default='output_1112_sam2.1_hiera_l_hels_finetune+lora/checkpoints/checkpoint_290.pt',
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

    # Test data arguments (single image or batch)
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single test image')
    parser.add_argument('--mask', type=str, default=None,
                        help='Path to single ground truth mask')
    parser.add_argument('--image-dir', type=str, default="/opt/data/private/lls/HLES-SAM/data/CVPR2026/image",
                        help='Directory containing test images')
    parser.add_argument('--mask-dir', type=str, default="/opt/data/private/lls/HLES-SAM/data/CVPR2026/ground_truth",
                        help='Directory containing ground truth masks')
    parser.add_argument('--image-ext', type=str, default='.png,.jpg,.jpeg',
                        help='Image file extensions (comma separated)')
    parser.add_argument('--mask-ext', type=str, default='.png',
                        help='Mask file extension')

    parser.add_argument('--output-dir', type=str, default='testing/output',
                        help='Output directory for results')

    # Evaluation arguments
    parser.add_argument('--max-clicks', type=int, default=25,
                        help='Maximum number of clicks (default: 25)')
    parser.add_argument('--target-iou', type=float, default=0.99,
                        help='Target IoU threshold')
    parser.add_argument('--pred-threshold', type=float, default=0.49,
                        help='Prediction threshold for binary mask')

    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization results')
    parser.add_argument('--visualize-all', action='store_true',
                        help='Save visualization for all images (not just failed ones)')

    args = parser.parse_args()

    # Validate arguments
    if args.image is None and args.image_dir is None:
        parser.error("Either --image or --image-dir must be specified")
    if args.image is not None and args.mask is None:
        parser.error("--mask must be specified when using --image")
    if args.image_dir is not None and args.mask_dir is None:
        parser.error("--mask-dir must be specified when using --image-dir")

    return args


def load_sam2_model(args):
    """Load SAM2 model with optional LoRA"""
    device = torch.device(args.device)

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

        print(f" SAM2 model loaded with LoRA (rank={args.lora_rank}, "
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

    if args.image is not None:
        # Single image mode
        pairs.append((Path(args.image), Path(args.mask)))
    else:
        # Batch mode
        image_dir = Path(args.image_dir)
        mask_dir = Path(args.mask_dir)

        # Get all image files
        image_exts = args.image_ext.split(',')
        image_files = []
        
        # test split
        with open("/opt/data/private/lls/HLES-SAM/data/CVPR2026/patients_test_6_2_2.txt","r") as f:
            lines = f.readlines()
            for line in lines:
                img_path = image_dir / line.strip()
                image_files.append(img_path)

        image_files = sorted(image_files)

        # Find corresponding masks
        for image_path in image_files:
            # Try to find mask with same name but different extension
            mask_path = mask_dir / f"{image_path.stem}{args.mask_ext}"

            if mask_path.exists():
                pairs.append((image_path, mask_path))
            else:
                print(f"Warning: Mask not found for {image_path.name}, skipping...")

    return pairs


def visualize_result(image, gt_mask, pred_mask, clicks_list, iou, save_path):
    """Visualize prediction result"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Original image with clicks
    axes[0].imshow(image)
    axes[0].set_title(f'Image with Clicks ({len(clicks_list)})', fontsize=12)
    for idx, click in enumerate(clicks_list):
        color = 'green' if click['is_positive'] else 'red'
        marker = '*'
        coords = click['coords']
        axes[0].scatter(coords[1], coords[0], c=color, marker=marker, s=200,
                       edgecolors='white', linewidths=2, zorder=100-idx)
    axes[0].axis('off')

    # 2. Ground truth
    axes[1].imshow(image)
    axes[1].imshow(gt_mask, alpha=0.5, cmap='jet')
    axes[1].set_title('Ground Truth', fontsize=12)
    axes[1].axis('off')

    # 3. Prediction
    axes[2].imshow(image)
    axes[2].imshow(pred_mask, alpha=0.5, cmap='jet')
    axes[2].set_title(f'Prediction (IoU={iou:.4f})', fontsize=12)
    axes[2].axis('off')

    # 4. Error map
    fn_mask = np.logical_and(gt_mask, np.logical_not(pred_mask))
    fp_mask = np.logical_and(np.logical_not(gt_mask), pred_mask)
    error_map = np.zeros((*gt_mask.shape, 3))
    error_map[fn_mask] = [1, 0, 0]  # False Negative: Red
    error_map[fp_mask] = [0, 0, 1]  # False Positive: Blue
    axes[3].imshow(image)
    axes[3].imshow(error_map, alpha=0.5)
    axes[3].set_title(f'Error Map (Red=FN, Blue=FP)', fontsize=12)
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_noc_metric(all_ious, iou_thrs=[0.80, 0.85, 0.90], max_clicks=25):

    noc_dict = {}
    noc_std_dict = {}
    over_max_dict = {}

    for iou_thr in iou_thrs:
        noc_values = []
        over_max_count = 0

        for sample_ious in all_ious:
            found = False
            for click_idx, iou in enumerate(sample_ious):
                if iou >= iou_thr:
                    noc_values.append(click_idx + 1)
                    found = True
                    break

            if not found:
                noc_values.append(max_clicks)
                over_max_count += 1

        noc_dict[iou_thr] = np.mean(noc_values)
        noc_std_dict[iou_thr] = np.std(noc_values)
        over_max_dict[iou_thr] = over_max_count / len(all_ious)

    return noc_dict, noc_std_dict, over_max_dict


def compute_miou_at_k(all_ious, k_values=[1, 3, 5, 10, 15, 20, 25]):
    miou_dict = {}

    for k in k_values:
        iou_at_k = []
        for sample_ious in all_ious:
            if len(sample_ious) >= k:
                iou_at_k.append(sample_ious[k - 1])

        if iou_at_k:
            miou_dict[k] = np.mean(iou_at_k)

    return miou_dict


def plot_average_iou_curve(all_ious, save_path, iou_thrs=[0.80, 0.85, 0.90]):
    """Plot average IoU progression curve"""
    # Compute average IoU at each click
    max_len = max(len(ious) for ious in all_ious)
    avg_ious = []
    std_ious = []

    for k in range(max_len):
        iou_at_k = [ious[k] for ious in all_ious if len(ious) > k]
        if iou_at_k:
            avg_ious.append(np.mean(iou_at_k))
            std_ious.append(np.std(iou_at_k))
        else:
            avg_ious.append(np.nan)
            std_ious.append(np.nan)

    avg_ious = np.array(avg_ious)
    std_ious = np.array(std_ious)
    clicks = np.arange(1, len(avg_ious) + 1)

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(clicks, avg_ious, 'b-o', linewidth=2, markersize=6, label='Average IoU')
    plt.fill_between(clicks, avg_ious - std_ious, avg_ious + std_ious,
                     alpha=0.2, color='blue', label='?1 std')

    # Add threshold lines
    colors = ['green', 'orange', 'red']
    for iou_thr, color in zip(iou_thrs, colors):
        plt.axhline(y=iou_thr, color=color, linestyle='--', linewidth=2,
                   label=f'IoU={iou_thr:.2f}')

    plt.xlabel('Number of Clicks', fontsize=12)
    plt.ylabel('IoU', fontsize=12)
    plt.title(f'Average IoU Progression ({len(all_ious)} samples)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim([0, 1.0])
    plt.xlim([1, len(avg_ious)])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def test_multiple_images(args):
    """Test on multiple images"""
    print("=" * 80)
    print("SAM2 + SimpleClick Batch Evaluation")
    print("=" * 80)

    # Get image-mask pairs
    pairs = get_image_mask_pairs(args)
    print(f"\n Found {len(pairs)} image-mask pairs")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    sam2_model = load_sam2_model(args)
    predictor = SAM2ImagePredictor(sam2_model)

    # Create adapter
    adapter = SAM2ClickerAdapter(predictor, pred_threshold=args.pred_threshold)

    # Store results
    all_ious = []
    all_results = []

    print("\n" + "=" * 80)
    print("Running Evaluation...")
    print("=" * 80)

    # Process each image
    for idx, (image_path, mask_path) in enumerate(tqdm(pairs, desc="Processing images")):
        try:
            # Load data
            image, gt_mask = load_image_and_mask(image_path, mask_path)

            # Run iterative prediction
            ious_list, final_mask, clicks_list = adapter.iterative_predict(
                image=image,
                gt_mask=gt_mask,
                max_clicks=args.max_clicks,
                target_iou=args.target_iou,
                multimask_output=True
            )

            # Store results
            all_ious.append(ious_list)
            result = {
                'image_name': image_path.name,
                'ious': [float(iou) for iou in ious_list],  # Convert to native Python float
                'num_clicks': len(clicks_list),
                'final_iou': float(ious_list[-1]),
                'max_iou': float(max(ious_list)),
                'reached_target': bool(ious_list[-1] >= args.target_iou)  # Convert to native Python bool
            }
            all_results.append(result)

            # Visualize if needed
            if args.visualize and (args.visualize_all or not result['reached_target']):
                viz_dir = output_dir / 'visualizations'
                viz_dir.mkdir(exist_ok=True)
                viz_path = viz_dir / f"{image_path.stem}_result.png"
                visualize_result(image, gt_mask, final_mask, clicks_list,
                               ious_list[-1], viz_path)

        except Exception as e:
            print(f"\nError processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute metrics
    print("\n" + "=" * 80)
    print("Computing Metrics...")
    print("=" * 80)

    # NoC metrics
    iou_thrs = [0.80, 0.85, 0.90]
    noc_dict, noc_std_dict, over_max_dict = compute_noc_metric(
        all_ious, iou_thrs=iou_thrs, max_clicks=args.max_clicks
    )

    # mIoU@k metrics
    k_values = [1, 3, 5, 10, 15, 20, 25]
    miou_dict = compute_miou_at_k(all_ious, k_values=k_values)

    # Overall statistics
    final_ious = [result['final_iou'] for result in all_results]
    num_clicks_list = [result['num_clicks'] for result in all_results]
    reached_target_count = sum(result['reached_target'] for result in all_results)

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Total samples: {len(all_ious)}")
    print(f"Reached target (IoU>={args.target_iou:.2f}): {reached_target_count}/{len(all_ious)} "
          f"({reached_target_count/len(all_ious)*100:.1f}%)")

    print("\n" + "-" * 80)
    print("NoC (Number of Clicks) Metrics:")
    print("-" * 80)
    for iou_thr in iou_thrs:
        print(f"  NoC@{iou_thr:.0%}: {noc_dict[iou_thr]:.2f} / {noc_std_dict[iou_thr]:.2f}")
        print(f"  >={args.max_clicks}@{iou_thr:.0%}: {over_max_dict[iou_thr]*100:.1f}%")

    print("\n" + "-" * 80)
    print("mIoU@k Metrics:")
    print("-" * 80)
    for k in k_values:
        if k in miou_dict:
            print(f"  mIoU@{k:2d}: {miou_dict[k]:.4f}")

    print("\n" + "-" * 80)
    print("Overall Statistics:")
    print("-" * 80)
    print(f"  Average final IoU: {np.mean(final_ious):.4f} / {np.std(final_ious):.4f}")
    print(f"  Average clicks: {np.mean(num_clicks_list):.2f} / {np.std(num_clicks_list):.2f}")
    print(f"  Min/Max final IoU: {np.min(final_ious):.4f} / {np.max(final_ious):.4f}")
    print(f"  Min/Max clicks: {np.min(num_clicks_list)} / {np.max(num_clicks_list)}")

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results...")
    print("=" * 80)

    # Save detailed results
    results_path = output_dir / 'detailed_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f" Detailed results saved to: {results_path}")

    # Save summary metrics
    summary = {
        'num_samples': len(all_ious),
        'max_clicks': args.max_clicks,
        'target_iou': args.target_iou,
        'reached_target_count': reached_target_count,
        'reached_target_ratio': reached_target_count / len(all_ious),
        'noc_metrics': {f'NoC@{k:.0%}': v for k, v in noc_dict.items()},
        'noc_std': {f'NoC@{k:.0%}_std': v for k, v in noc_std_dict.items()},
        'over_max': {f'>={args.max_clicks}@{k:.0%}': v for k, v in over_max_dict.items()},
        'miou_metrics': {f'mIoU@{k}': v for k, v in miou_dict.items()},
        'avg_final_iou': float(np.mean(final_ious)),
        'std_final_iou': float(np.std(final_ious)),
        'avg_clicks': float(np.mean(num_clicks_list)),
        'std_clicks': float(np.std(num_clicks_list)),
    }

    summary_path = output_dir / 'summary_metrics.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f" Summary metrics saved to: {summary_path}")

    # Save IoU curve
    curve_path = output_dir / 'average_iou_curve.png'
    plot_average_iou_curve(all_ious, curve_path, iou_thrs=iou_thrs)
    print(f" Average IoU curve saved to: {curve_path}")

    # Save text report
    report_path = output_dir / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SAM2 + SimpleClick Evaluation Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Total samples: {len(all_ious)}\n")
        f.write(f"Max clicks: {args.max_clicks}\n")
        f.write(f"Target IoU: {args.target_iou:.2f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("NoC Metrics:\n")
        f.write("-" * 80 + "\n")
        for iou_thr in iou_thrs:
            f.write(f"NoC@{iou_thr:.0%}: {noc_dict[iou_thr]:.2f} ? {noc_std_dict[iou_thr]:.2f}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("mIoU@k Metrics:\n")
        f.write("-" * 80 + "\n")
        for k in k_values:
            if k in miou_dict:
                f.write(f"mIoU@{k:2d}: {miou_dict[k]:.4f}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("Per-Image Results:\n")
        f.write("-" * 80 + "\n")
        for result in all_results:
            f.write(f"{result['image_name']:30s}: IoU={result['final_iou']:.4f}, "
                   f"Clicks={result['num_clicks']:2d}, "
                   f"Target={'' if result['reached_target'] else ''}\n")

    print(f" Text report saved to: {report_path}")

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)

    return summary, all_results


def main():
    args = parse_args()

    # Run evaluation
    summary, all_results = test_multiple_images(args)

    return summary


if __name__ == '__main__':
    main()
