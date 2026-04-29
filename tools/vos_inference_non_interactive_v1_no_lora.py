# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor


# ---------------------------------------------------------------------------
# DAVIS palette helpers
# ---------------------------------------------------------------------------

DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)


# ---------------------------------------------------------------------------
# Core: single-frame non-interactive segmentation
# ---------------------------------------------------------------------------

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def segment_frame_non_interactive(
    predictor, inference_state, frame_idx=0, score_thresh=0.0
):
    """Run prompt-free segmentation on a single frame via the no_mem_embed path."""
    video_H = inference_state["video_height"]
    video_W = inference_state["video_width"]

    compact_out, pred_masks_gpu = predictor._run_single_frame_inference(
        inference_state=inference_state,
        output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        frame_idx=frame_idx,
        batch_size=1,
        is_init_cond_frame=True,  # triggers the no_mem_embed path
        point_inputs=None,
        mask_inputs=None,
        reverse=False,
        run_mem_encoder=False,
    )

    # Upsample to video resolution (model output is typically 256×256)
    raw_logit = F.interpolate(
        pred_masks_gpu.float(),
        size=(video_H, video_W),
        mode="bilinear",
        align_corners=False,
    )  # [1, 1, H, W]  float32, on GPU

    binary_mask = (raw_logit.squeeze() > score_thresh).cpu().numpy()

    iou = compact_out.get("pred_ious")
    if iou is None:
        iou = torch.tensor([0.5])
    else:
        iou = iou.cpu().float()

    return binary_mask, iou, raw_logit


# ---------------------------------------------------------------------------
# Standard (non-fusion) pipeline
# ---------------------------------------------------------------------------

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_non_interactive_inference(
    predictor,
    base_video_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    per_obj_png_file=False,
):
    """Non-interactive VOS inference: image-level first frame → propagate."""
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = sorted(
        [
            os.path.splitext(p)[0]
            for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in (".jpg", ".jpeg", ".JPG", ".JPEG")
        ],
        key=lambda p: int(p),
    )

    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]

    # Step 1: segment the first frame without any prompt
    first_frame_mask, _, _ = segment_frame_non_interactive(
        predictor, inference_state, frame_idx=0, score_thresh=score_thresh
    )

    # Step 2: inject the predicted mask as a conditioning prompt
    predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        mask=first_frame_mask,
    )

    # Step 3: propagate across the whole video
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        per_obj_output_mask = {
            out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_segments[out_frame_idx] = per_obj_output_mask

    for out_frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[out_frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=per_obj_png_file,
            output_palette=list(DAVIS_PALETTE),
        )


# ---------------------------------------------------------------------------
# Fusion (three-path) pipeline v1
# Backward anchor = last frame directly (no IoU-based search)
# ---------------------------------------------------------------------------

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_non_interactive_fusion_inference(
    predictor,
    base_video_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    per_obj_png_file=False,
):
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = sorted(
        [
            os.path.splitext(p)[0]
            for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in (".jpg", ".jpeg", ".JPG", ".JPEG")
        ],
        key=lambda p: int(p),
    )
    num_frames = len(frame_names)

    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]

    # ── Step 1: Forward propagation ──────────────────────────────────────────
    fwd_mask_0, fwd_iou_0, fwd_raw_logit_0 = segment_frame_non_interactive(
        predictor, inference_state, frame_idx=0, score_thresh=score_thresh
    )
    predictor.add_new_mask(inference_state, 0, 1, fwd_mask_0)

    fwd_results = {}  # frame_idx → (logit [1,1,H,W], iou [1])
    for out_frame_idx, out_obj_ids, out_mask_logits, out_ious in (
        predictor.propagate_in_video(inference_state, return_ious=True)
    ):
        fwd_results[out_frame_idx] = (
            out_mask_logits[0:1].clone(),  # [1,1,H,W] at video resolution
            out_ious[0].cpu().float(),     # [1]
        )

    fwd_results[0] = (fwd_raw_logit_0, fwd_iou_0)

    # ── Step 2: Backward propagation (v1: anchor = last frame directly) ───────
    predictor.reset_state(inference_state)

    last_frame_idx = num_frames - 1
    bwd_mask_last, bwd_iou_last, bwd_raw_logit_last = segment_frame_non_interactive(
        predictor, inference_state, frame_idx=last_frame_idx, score_thresh=score_thresh
    )

    bwd_results = {}
    if bwd_mask_last.any():
        predictor.add_new_mask(inference_state, last_frame_idx, 1, bwd_mask_last)
        for out_frame_idx, out_obj_ids, out_mask_logits, out_ious in (
            predictor.propagate_in_video(
                inference_state,
                start_frame_idx=last_frame_idx,
                reverse=True,
                return_ious=True,
            )
        ):
            bwd_results[out_frame_idx] = (
                out_mask_logits[0:1].clone(),
                out_ious[0].cpu().float(),
            )

        # Overwrite last frame entry with the raw logit from image-level seg
        bwd_results[last_frame_idx] = (bwd_raw_logit_last, bwd_iou_last)

    # ── Step 3: Per-frame image-level segmentation ────────────────────────────
    predictor.reset_state(inference_state)
    img_results = {}  # frame_idx → (raw_logit [1,1,H,W], iou [1])
    for i in range(num_frames):
        _, img_iou, img_raw_logit = segment_frame_non_interactive(
            predictor, inference_state, frame_idx=i, score_thresh=score_thresh
        )
        img_results[i] = (img_raw_logit, img_iou)

    # ── Step 4: IoU-weighted softmax fusion ───────────────────────────────────
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    zeros = torch.zeros(1, 1, height, width)

    for i in range(num_frames):
        fwd_logit, fwd_iou = fwd_results.get(i, (zeros, torch.tensor([0.0])))
        bwd_logit, bwd_iou = bwd_results.get(i, (zeros, torch.tensor([0.0])))
        img_logit, img_iou = img_results[i]

        device = fwd_logit.device
        bwd_logit = bwd_logit.to(device)
        img_logit = img_logit.to(device)

        ious = torch.cat([fwd_iou, bwd_iou, img_iou]).to(device)  # [3]
        weights = torch.softmax(ious.float(), dim=0)               # [3]

        fused = (
            weights[0] * fwd_logit.float()
            + weights[1] * bwd_logit.float()
            + weights[2] * img_logit.float()
        )  # [1,1,H,W]

        binary = (fused.squeeze() > score_thresh).cpu().numpy()

        per_obj_output_mask = {0: binary}
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[i],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=per_obj_png_file,
            output_palette=list(DAVIS_PALETTE),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Non-interactive VOS inference v1 (last-frame backward anchor)"
    )
    parser.add_argument("--sam2_cfg", type=str, required=True)
    parser.add_argument("--sam2_checkpoint", type=str, required=True)
    parser.add_argument("--base_video_dir", type=str, required=True)
    parser.add_argument("--output_mask_dir", type=str, required=True)
    parser.add_argument("--video_list_file", type=str, default=None)
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument("--per_obj_png_file", action="store_true")
    parser.add_argument("--apply_postprocessing", action="store_true")
    parser.add_argument("--use_fusion", action="store_true")
    # parser.add_argument("--lora_rank", type=int, default=8)
    # parser.add_argument("--lora_dropout", type=float, default=0.1)
    args = parser.parse_args()

    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("false" if args.per_obj_png_file else "true")
    ]

    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
    )

    if args.video_list_file is not None:
        with open(args.video_list_file) as f:
            video_names = [v.strip() for v in f.readlines()]
    else:
        video_names = sorted(
            p
            for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
        )

    mode = "fusion (v1)" if args.use_fusion else "standard"
    print(
        f"Running non-interactive VOS inference ({mode}) "
        f"on {len(video_names)} video(s)."
    )

    for n_video, video_name in enumerate(video_names):
        print(f"\n{n_video + 1}/{len(video_names)} - {video_name}")
        if args.use_fusion:
            vos_non_interactive_fusion_inference(
                predictor=predictor,
                base_video_dir=args.base_video_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                score_thresh=args.score_thresh,
                per_obj_png_file=args.per_obj_png_file,
            )
        else:
            vos_non_interactive_inference(
                predictor=predictor,
                base_video_dir=args.base_video_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                score_thresh=args.score_thresh,
                per_obj_png_file=args.per_obj_png_file,
            )

    print(f"\nDone. Output masks saved to {args.output_mask_dir}")


if __name__ == "__main__":
    main()
