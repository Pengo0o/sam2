import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from sam2.build_sam import build_sam2_video_predictor_with_lora


def select_device() -> torch.device:
    """Select computation device (prefer CUDA)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        # Optional CUDA settings similar to the notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nWarning: MPS support is preliminary. Results may differ from CUDA.\n"
        )
    return device


def build_predictor(device: torch.device):
    """Build the SAM2 video predictor with LoRA, using the same config/checkpoint as the notebook."""
    # Use absolute paths to avoid issues with different working directories
    sam2_checkpoint = (
        "/opt/data/private/hyp/sam2/output_1112_sam2.1_hiera_l_hels_finetune+lora/"
        "checkpoints/checkpoint_300.pt"
    )
    model_cfg = "/opt/data/private/hyp/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

    predictor = build_sam2_video_predictor_with_lora(
        model_cfg,
        sam2_checkpoint,
        device=device,
        lora_rank=8,
        lora_dropout=0.1,
        lora_target_modules=("qkv", "proj"),
    )
    return predictor


def parse_frame_idx(name: str) -> int:
    """
    Extract the integer index from a filename like 'LI WAN QIANG.Ser7.Img1234.png'
    or 'LI WAN QIANG.Ser7.Img1234_mask.png'.
    """
    stem = Path(name).stem.replace("_mask", "")
    return int(stem.split("Img")[-1])


def main():
    # -------------------------
    # Paths and basic settings
    # -------------------------
    long_video_dir = "/opt/data/private/hyp/sam2/data_LiWanQiang1125/images"
    long_mask_dir = "/opt/data/private/hyp/sam2/data_LiWanQiang1125/mask_181"
    output_dir_long_chunked = (
        "/opt/data/private/hyp/sam2/data_LiWanQiang1125/pred_masks_sam2_chunked"
    )
    chunk_jpg_dir = (
        "/opt/data/private/hyp/sam2/data_LiWanQiang1125/images_jpg_chunk"
    )

    os.makedirs(output_dir_long_chunked, exist_ok=True)
    os.makedirs(chunk_jpg_dir, exist_ok=True)

    valid_frame_exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    long_obj_id = 1
    mask_threshold = 127  # grayscale threshold to binarize masks
    chunk_size = 300

    # -------------------------
    # Collect frames and masks
    # -------------------------
    print("Scanning long video frames and sparse masks...")
    long_frame_paths = sorted(
        [
            p
            for p in Path(long_video_dir).iterdir()
            if p.suffix in valid_frame_exts
        ],
        key=lambda p: parse_frame_idx(p.name),
    )
    long_frame_names = [p.name for p in long_frame_paths]
    frame_name_to_idx = {name: idx for idx, name in enumerate(long_frame_names)}

    long_mask_paths = sorted(
        [
            p
            for p in Path(long_mask_dir).iterdir()
            if p.suffix in valid_frame_exts
        ],
        key=lambda p: parse_frame_idx(p.name),
    )

    print(f"Total frames: {len(long_frame_names)}")
    print(f"Sparse masks: {len(long_mask_paths)}")
    if not long_frame_paths:
        raise RuntimeError(f"No frames found in {long_video_dir}")
    if not long_mask_paths:
        print(f"Warning: no sparse masks found in {long_mask_dir}")

    # Map global frame index -> list of mask paths
    global_idx_to_masks = defaultdict(list)
    for mp in long_mask_paths:
        frame_stem = mp.stem.replace("_mask", "")
        frame_name = f"{frame_stem}.png"
        gi = frame_name_to_idx.get(frame_name, None)
        if gi is not None:
            global_idx_to_masks[gi].append(mp)

    # -------------------------
    # Build predictor
    # -------------------------
    device = select_device()
    predictor = build_predictor(device)

    # -------------------------
    # Chunked processing loop
    # -------------------------
    for chunk_start in range(0, len(long_frame_paths), chunk_size):
        chunk_end = min(len(long_frame_paths), chunk_start + chunk_size)
        print(f"\n=== Processing chunk {chunk_start}–{chunk_end - 1} ===")

        # Clear old JPGs in the chunk folder
        for f in os.listdir(chunk_jpg_dir):
            os.remove(os.path.join(chunk_jpg_dir, f))

        # Export this chunk's PNGs to sequential JPGs 00000.jpg, 00001.jpg, ...
        print("Exporting PNG frames to temporary JPGs...")
        for local_idx, global_idx in enumerate(
            tqdm(range(chunk_start, chunk_end), desc="Export PNG->JPG")
        ):
            img_path = long_frame_paths[global_idx]
            img = Image.open(img_path).convert("RGB")
            out_path = os.path.join(chunk_jpg_dir, f"{local_idx:05d}.jpg")
            img.save(out_path, quality=95)

        # Initialize SAM2 state for this chunk
        print("Initializing SAM2 state for this chunk...")
        chunk_state = predictor.init_state(
            video_path=chunk_jpg_dir, offload_video_to_cpu=True
        )

        # Add sparse masks that fall inside this chunk
        print("Adding sparse mask prompts for this chunk...")
        for global_idx in range(chunk_start, chunk_end):
            if global_idx not in global_idx_to_masks:
                continue
            local_idx = global_idx - chunk_start
            for mp in global_idx_to_masks[global_idx]:
                mask_arr = np.array(Image.open(mp).convert("L"))
                mask_bool = mask_arr > mask_threshold
                predictor.add_new_mask(
                    inference_state=chunk_state,
                    frame_idx=local_idx,
                    obj_id=long_obj_id,
                    mask=mask_bool,
                )

        # Propagate within this chunk
        print("Propagating masks in this chunk...")
        chunk_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            chunk_state
        ):
            chunk_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Save masks for this chunk
        print("Saving predicted masks for this chunk...")
        for local_idx in range(chunk_end - chunk_start):
            global_idx = chunk_start + local_idx
            if local_idx not in chunk_segments:
                continue
            if long_obj_id not in chunk_segments[local_idx]:
                continue
            mask = chunk_segments[local_idx][long_obj_id]
            mask_uint8 = (mask.squeeze() * 255).astype(np.uint8)
            out_name = Path(long_frame_names[global_idx]).with_suffix(".png").name
            out_path = os.path.join(output_dir_long_chunked, out_name)
            cv2.imwrite(out_path, mask_uint8)

        print(
            f"Saved masks for frames {chunk_start}–{chunk_end - 1} "
            f"into {output_dir_long_chunked}"
        )

    print("\nAll chunks processed. Dense predictions written to:")
    print(output_dir_long_chunked)


if __name__ == "__main__":
    main()


