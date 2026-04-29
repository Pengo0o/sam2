# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Debug 版推理脚本：将四路分割结果分别保存到独立文件夹。

推理逻辑与 vos_inference_non_interactive_v2.py 完全一致（Bug 1/2/3 修复），
只新增将中间结果分别写入：

  {output_mask_dir}/
    img/    {video_name}/*.png  — 逐帧 image-level 分割（no_mem_embed，无时序）
    fwd/    {video_name}/*.png  — 前向传播结果
    bwd/    {video_name}/*.png  — 后向传播结果（anchor 之后的帧为空掩码）
    fused/  {video_name}/*.png  — 三路 IoU 加权 softmax 融合结果
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor_with_lora


# ---------------------------------------------------------------------------
# DAVIS palette helpers
# ---------------------------------------------------------------------------

DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def save_ann_png(path, mask, palette):
    assert mask.dtype == np.uint8 and mask.ndim == 2
    img = Image.fromarray(mask)
    img.putpalette(palette)
    img.save(path)


def save_single_mask(out_dir, video_name, frame_name, binary_mask, height, width):
    """binary_mask: bool numpy [H,W]，前景=1，背景=0。"""
    os.makedirs(os.path.join(out_dir, video_name), exist_ok=True)
    mask_u8 = binary_mask.reshape(height, width).astype(np.uint8)
    save_ann_png(
        os.path.join(out_dir, video_name, f"{frame_name}.png"),
        mask_u8,
        list(DAVIS_PALETTE),
    )


def logit_to_mask(logit, score_thresh=0.0):
    """float tensor [1,1,H,W] → bool numpy [H,W]"""
    return (logit.squeeze() > score_thresh).cpu().numpy()


# ---------------------------------------------------------------------------
# 核心：单帧 no_mem_embed 分割（与 v2 完全相同）
# ---------------------------------------------------------------------------

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def segment_frame_non_interactive(predictor, inference_state, frame_idx=0, score_thresh=0.0):
    """单帧 no_mem_embed 分割，不修改 inference_state 全局 memory。

    返回:
        binary_mask : bool numpy [H, W]
        iou         : float32 tensor [1]
        raw_logit   : float32 tensor [1,1,H,W]，视频分辨率软 logit
    """
    video_H = inference_state["video_height"]
    video_W = inference_state["video_width"]

    compact_out, pred_masks_gpu = predictor._run_single_frame_inference(
        inference_state=inference_state,
        output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        frame_idx=frame_idx,
        batch_size=1,
        is_init_cond_frame=True,
        point_inputs=None,
        mask_inputs=None,
        reverse=False,
        run_mem_encoder=False,
    )

    raw_logit = F.interpolate(
        pred_masks_gpu.float(), size=(video_H, video_W),
        mode="bilinear", align_corners=False,
    )
    binary_mask = (raw_logit.squeeze() > score_thresh).cpu().numpy()

    iou = compact_out.get("pred_ious")
    iou = torch.tensor([0.5]) if iou is None else iou.cpu().float()

    return binary_mask, iou, raw_logit


# ---------------------------------------------------------------------------
# Debug 推理主函数（推理逻辑与 v2 完全一致）
# ---------------------------------------------------------------------------

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_fusion_debug_inference(
    predictor,
    base_video_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
):
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = sorted(
        [os.path.splitext(p)[0] for p in os.listdir(video_dir)
         if os.path.splitext(p)[-1] in (".jpg", ".jpeg", ".JPG", ".JPEG")],
        key=lambda p: int(p),
    )
    num_frames = len(frame_names)

    inference_state = predictor.init_state(video_path=video_dir, async_loading_frames=False)
    height = inference_state["video_height"]
    width  = inference_state["video_width"]

    dir_img   = os.path.join(output_mask_dir, "img")
    dir_fwd   = os.path.join(output_mask_dir, "fwd")
    dir_bwd   = os.path.join(output_mask_dir, "bwd")
    dir_fused = os.path.join(output_mask_dir, "fused")

    zeros = torch.zeros(1, 1, height, width)

    # ── Step 1: 前向传播 ──────────────────────────────────────────────────────
    print(f"  [fwd]  前向传播 frame 0 → {num_frames - 1}")
    fwd_mask_0, fwd_iou_0, fwd_raw_logit_0 = segment_frame_non_interactive(
        predictor, inference_state, frame_idx=0, score_thresh=score_thresh
    )
    predictor.add_new_mask(inference_state, 0, 1, fwd_mask_0)

    fwd_results = {}
    for out_frame_idx, out_obj_ids, out_mask_logits, out_ious in (
        predictor.propagate_in_video(inference_state, return_ious=True)
    ):
        fwd_results[out_frame_idx] = (
            out_mask_logits[0:1].clone(),
            out_ious[0].cpu().float(),
        )
    # Bug 2+3 fix：用干净的 no_mem_embed 结果覆盖 frame 0
    fwd_results[0] = (fwd_raw_logit_0, fwd_iou_0)

    # ── Step 2: 后向传播 ──────────────────────────────────────────────────────
    predictor.reset_state(inference_state)
    bwd_iou_thresh = 0.5
    bwd_anchor_frame = None
    bwd_anchor_mask  = None
    bwd_anchor_iou   = None
    bwd_anchor_raw_logit = None
    bwd_fallback_frame = None
    bwd_fallback_mask  = None
    bwd_fallback_iou   = None
    bwd_fallback_raw_logit = None

    print(f"  [bwd]  扫描后向锚帧（IoU 阈值={bwd_iou_thresh}）...")
    for scan_idx in range(num_frames - 1, -1, -1):
        mask_scan, iou_scan, raw_logit_scan = segment_frame_non_interactive(
            predictor, inference_state, frame_idx=scan_idx, score_thresh=score_thresh
        )
        if mask_scan.any():
            if bwd_fallback_frame is None:
                bwd_fallback_frame     = scan_idx
                bwd_fallback_mask      = mask_scan
                bwd_fallback_iou       = iou_scan
                bwd_fallback_raw_logit = raw_logit_scan
            if iou_scan.item() >= bwd_iou_thresh:
                bwd_anchor_frame     = scan_idx
                bwd_anchor_mask      = mask_scan
                bwd_anchor_iou       = iou_scan
                bwd_anchor_raw_logit = raw_logit_scan
                break

    if bwd_anchor_frame is None:
        bwd_anchor_frame     = bwd_fallback_frame
        bwd_anchor_mask      = bwd_fallback_mask
        bwd_anchor_iou       = bwd_fallback_iou
        bwd_anchor_raw_logit = bwd_fallback_raw_logit

    bwd_results = {}
    if bwd_anchor_frame is not None:
        print(f"  [bwd]  后向传播 frame {bwd_anchor_frame} → 0")
        predictor.add_new_mask(inference_state, bwd_anchor_frame, 1, bwd_anchor_mask)
        for out_frame_idx, out_obj_ids, out_mask_logits, out_ious in (
            predictor.propagate_in_video(
                inference_state,
                start_frame_idx=bwd_anchor_frame,
                reverse=True,
                return_ious=True,
            )
        ):
            bwd_results[out_frame_idx] = (
                out_mask_logits[0:1].clone(),
                out_ious[0].cpu().float(),
            )
        # Bug 2+3 fix：用干净的 no_mem_embed 结果覆盖锚帧
        bwd_results[bwd_anchor_frame] = (bwd_anchor_raw_logit, bwd_anchor_iou)
    else:
        print(f"  [bwd]  未找到有效锚帧，后向结果全为空掩码")

    # ── Step 3: 逐帧 image-level 分割 ────────────────────────────────────────
    predictor.reset_state(inference_state)
    print(f"  [img]  逐帧 image-level 分割（共 {num_frames} 帧）")
    img_results = {}
    for i in range(num_frames):
        _, img_iou, img_raw_logit = segment_frame_non_interactive(
            predictor, inference_state, frame_idx=i, score_thresh=score_thresh
        )
        img_results[i] = (img_raw_logit, img_iou)

    # ── Step 4: 融合 + 保存四路结果 ──────────────────────────────────────────
    print(f"  [save] 保存四路结果...")
    for i in range(num_frames):
        fname = frame_names[i]

        fwd_logit, fwd_iou = fwd_results.get(i, (zeros, torch.tensor([0.0])))
        bwd_logit, bwd_iou = bwd_results.get(i, (zeros, torch.tensor([0.0])))
        img_logit, img_iou = img_results[i]

        device = fwd_logit.device
        bwd_logit = bwd_logit.to(device)
        img_logit = img_logit.to(device)

        ious    = torch.cat([fwd_iou, bwd_iou, img_iou]).to(device)
        weights = torch.softmax(ious.float(), dim=0)
        fused   = (
            weights[0] * fwd_logit.float()
            + weights[1] * bwd_logit.float()
            + weights[2] * img_logit.float()
        )

        save_single_mask(dir_img,   video_name, fname,
                         logit_to_mask(img_logit, score_thresh), height, width)
        save_single_mask(dir_fwd,   video_name, fname,
                         logit_to_mask(fwd_logit, score_thresh), height, width)
        save_single_mask(dir_bwd,   video_name, fname,
                         logit_to_mask(bwd_logit, score_thresh), height, width)
        save_single_mask(dir_fused, video_name, fname,
                         logit_to_mask(fused,     score_thresh), height, width)

    print(f"  [done] bwd 锚帧={bwd_anchor_frame}")
    print(f"         img   → {dir_img}/{video_name}/")
    print(f"         fwd   → {dir_fwd}/{video_name}/")
    print(f"         bwd   → {dir_bwd}/{video_name}/")
    print(f"         fused → {dir_fused}/{video_name}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Debug：保存 img / fwd / bwd / fused 四路分割结果"
    )
    parser.add_argument("--sam2_cfg",        type=str, required=True)
    parser.add_argument("--sam2_checkpoint", type=str, required=True)
    parser.add_argument("--base_video_dir",  type=str, required=True)
    parser.add_argument("--output_mask_dir", type=str, required=True)
    parser.add_argument("--video_list_file", type=str, default=None)
    parser.add_argument("--score_thresh",    type=float, default=0.0)
    parser.add_argument("--apply_postprocessing", action="store_true")
    parser.add_argument("--lora_rank",    type=int,   default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    args = parser.parse_args()

    predictor = build_sam2_video_predictor_with_lora(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=["++model.non_overlap_masks=true"],
        lora_rank=args.lora_rank,
        lora_dropout=args.lora_dropout,
        lora_target_modules=("qkv", "proj"),
    )

    if args.video_list_file is not None:
        with open(args.video_list_file) as f:
            video_names = [v.strip() for v in f.readlines()]
    else:
        video_names = sorted(
            p for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
        )

    print(f"Debug 推理：{len(video_names)} 个视频")
    print(f"输出根目录：{args.output_mask_dir}/{{img,fwd,bwd,fused}}/\n")

    for n, video_name in enumerate(video_names):
        print(f"[{n + 1}/{len(video_names)}] {video_name}")
        vos_fusion_debug_inference(
            predictor=predictor,
            base_video_dir=args.base_video_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            score_thresh=args.score_thresh,
        )

    print(f"\n全部完成。结果保存至 {args.output_mask_dir}/")


if __name__ == "__main__":
    main()
