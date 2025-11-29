# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import cv2
import numpy as np
import torch
import torch.distributed
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import (
    get_1d_sine_pe,
    get_next_point,
    sample_box_points,
    select_closest_cond_frames,
)

from sam2.utils.misc import concat_points

from training.utils.data_utils import BatchedVideoDatapoint


class SAM2Train(SAM2Base):
    def __init__(
        self,
        image_encoder,
        memory_attention=None,
        memory_encoder=None,
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
        num_frames_to_correct_for_train=1,  # default: only iteratively sample on first frame
        num_frames_to_correct_for_eval=1,  # default: only iteratively sample on first frame
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
        # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
        # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
        # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
        # these are initial conditioning frames because as we track the video, more conditioning frames might be added
        # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
        num_init_cond_frames_for_train=1,  # default: only use the first frame as initial conditioning frame
        num_init_cond_frames_for_eval=1,  # default: only use the first frame as initial conditioning frame
        rand_init_cond_frames_for_train=True,  # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        rand_init_cond_frames_for_eval=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        # how many additional correction points to sample (on each frame selected to be corrected)
        # note that the first frame receives an initial input click (in addition to any correction clicks)
        num_correction_pt_per_frame=7,
        # method for point sampling during evaluation
        # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
        # default to "center" to be consistent with evaluation in the SAM paper
        pt_sampling_for_eval="center",
        # During training, we optionally allow sampling the correction points from GT regions
        # instead of the prediction error regions with a small probability. This might allow the
        # model to overfit less to the error regions in training datasets
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        forward_backbone_per_frame_for_eval=False,
        freeze_image_encoder=False,
        # Visualization parameters
        visualize_interval=100,  # Visualize every N iterations
        visualize_dir="visualization_output",  # Directory to save visualizations
        **kwargs,
    ):
        super().__init__(image_encoder, memory_attention, memory_encoder, **kwargs)
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval

        # Visualization settings
        self.visualize_interval = visualize_interval
        self.visualize_dir = visualize_dir
        self.iter_count = 0

        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def forward(self, input: BatchedVideoDatapoint):
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(input.flat_img_batch)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)

        # Visualization (only on rank 0 to avoid duplicate outputs in distributed training)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                self.visualize_segmentation(input, backbone_out, previous_stages_out)
        else:
            self.visualize_segmentation(input, backbone_out, previous_stages_out)

        return previous_stages_out

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        """
        # Load the ground-truth masks on all frames (so that we can later
        # sample correction points from them)
        # gt_masks_per_frame = {
        #     stage_id: targets.segments.unsqueeze(1)  # [B, 1, H_im, W_im]
        #     for stage_id, targets in enumerate(input.find_targets)
        # }
        gt_masks_per_frame = {
            stage_id: masks.unsqueeze(1)  # [B, 1, H_im, W_im]
            for stage_id, masks in enumerate(input.masks)
        }
        # gt_masks_per_frame = input.masks.unsqueeze(2) # [T,B,1,H_im,W_im] keep everything in tensor form
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = input.num_frames
        backbone_out["num_frames"] = num_frames

        # Randomly decide whether to use point inputs or mask inputs
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        if num_frames == 1:
            # here we handle a special case for mixing video + SAM on image training,
            # where we force using point input for the SAM task on static images
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        assert num_init_cond_frames >= 1
        # (here `self.rng.random()` returns value in range 0.0 <= X < 1.0)
        use_pt_input = self.rng.random() < prob_to_use_pt_input
        if rand_init_cond_frames and num_init_cond_frames > 1:
            # randomly select 1 to `num_init_cond_frames` frames as initial conditioning frames
            num_init_cond_frames = self.rng.integers(
                1, num_init_cond_frames, endpoint=True
            )
        if (
            use_pt_input
            and rand_frames_to_correct
            and num_frames_to_correct > num_init_cond_frames
        ):
            # randomly select `num_init_cond_frames` to `num_frames_to_correct` frames to sample
            # correction clicks (only for the case of point input)
            num_frames_to_correct = self.rng.integers(
                num_init_cond_frames, num_frames_to_correct, endpoint=True
            )
        backbone_out["use_pt_input"] = use_pt_input

        # Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]  # starting frame
        else:
            # starting frame + randomly selected remaining frames (without replacement)
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        # Prepare mask or point inputs on initial conditioning frames
        backbone_out["mask_inputs_per_frame"] = {}  # {frame_idx: <input_masks>}
        backbone_out["point_inputs_per_frame"] = {}  # {frame_idx: <input_points>}
        for t in init_cond_frames:
            if not use_pt_input:
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
            else:
                # During training # P(box) = prob_to_use_pt_input * prob_to_use_box_input
                use_box_input = self.rng.random() < prob_to_use_box_input
                if use_box_input:
                    points, labels = sample_box_points(
                        gt_masks_per_frame[t],
                    )
                else:
                    # (here we only sample **one initial point** on initial conditioning frames from the
                    # ground-truth mask; we may sample more correction points on the fly)
                    points, labels = get_next_point(
                        gt_masks=gt_masks_per_frame[t],
                        pred_masks=None,
                        method=(
                            "uniform" if self.training else self.pt_sampling_for_eval
                        ),
                    )

                point_inputs = {"point_coords": points, "point_labels": labels}
                backbone_out["point_inputs_per_frame"][t] = point_inputs

        # Sample frames where we will add correction clicks on the fly
        # based on the error between prediction and ground-truth masks
        if not use_pt_input:
            # no correction points will be sampled when using mask inputs
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            assert num_frames_to_correct > num_init_cond_frames
            # initial cond frame + randomly selected remaining frames (without replacement)
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                init_cond_frames
                + self.rng.choice(
                    backbone_out["frames_not_in_init_cond"], extra_num, replace=False
                ).tolist()
            )
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt

        return backbone_out

    def forward_tracking(
        self, backbone_out, input: BatchedVideoDatapoint, return_dict=False
    ):
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)

        # Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        for stage_id in processing_order:
            # Get the image features for the current frames
            # img_ids = input.find_inputs[stage_id].img_ids
            img_ids = input.flat_obj_to_img_idx[stage_id]
            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                (
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )

            # Get output masks based on this frame's prompts and previous memory
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
            )
            # Append the output, depending on whether it's a conditioning frame
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        if return_dict:
            return output_dict
        # turn `output_dict` into a list for loss function
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        frames_to_add_correction_pt=None,
        gt_masks=None,
    ):
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        # Optionally, sample correction points iteratively to correct the mask
        if frame_idx in frames_to_add_correction_pt:
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame,
                point_inputs,
                gt_masks,
                high_res_features,
                pix_feat,
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                object_score_logits,
                current_out,
            )
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = final_sam_outputs

        # Use the final prediction (after all correction steps for output and eval)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out

    def _iter_correct_pt_sampling(
        self,
        is_init_cond_frame,
        point_inputs,
        gt_masks,
        high_res_features,
        pix_feat_with_mem,
        low_res_multimasks,
        high_res_multimasks,
        ious,
        low_res_masks,
        high_res_masks,
        object_score_logits,
        current_out,
    ):

        assert gt_masks is not None
        all_pred_masks = [low_res_masks]
        all_pred_high_res_masks = [high_res_masks]
        all_pred_multimasks = [low_res_multimasks]
        all_pred_high_res_multimasks = [high_res_multimasks]
        all_pred_ious = [ious]
        all_point_inputs = [point_inputs]
        all_object_score_logits = [object_score_logits]
        for _ in range(self.num_correction_pt_per_frame):
            # sample a new point from the error between prediction and ground-truth
            # (with a small probability, directly sample from GT masks instead of errors)
            if self.training and self.prob_to_sample_from_gt_for_train > 0:
                sample_from_gt = (
                    self.rng.random() < self.prob_to_sample_from_gt_for_train
                )
            else:
                sample_from_gt = False
            # if `pred_for_new_pt` is None, only GT masks will be used for point sampling
            pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
            new_points, new_labels = get_next_point(
                gt_masks=gt_masks,
                pred_masks=pred_for_new_pt,
                method="uniform" if self.training else self.pt_sampling_for_eval,
            )
            point_inputs = concat_points(point_inputs, new_points, new_labels)
            # Feed the mask logits of the previous SAM outputs in the next SAM decoder step.
            # For tracking, this means that when the user adds a correction click, we also feed
            # the tracking output mask logits along with the click as input to the SAM decoder.
            mask_inputs = low_res_masks
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            if self.use_act_ckpt_iterative_pt_sampling and not multimask_output:
                sam_outputs = torch.utils.checkpoint.checkpoint(
                    self._forward_sam_heads,
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    use_reentrant=False,
                )
            else:
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                )
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                _,
                object_score_logits,
            ) = sam_outputs
            all_pred_masks.append(low_res_masks)
            all_pred_high_res_masks.append(high_res_masks)
            all_pred_multimasks.append(low_res_multimasks)
            all_pred_high_res_multimasks.append(high_res_multimasks)
            all_pred_ious.append(ious)
            all_point_inputs.append(point_inputs)
            all_object_score_logits.append(object_score_logits)

        # Concatenate the masks along channel (to compute losses on all of them,
        # using `MultiStepIteractiveMasks`)
        current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
        current_out["multistep_pred_masks_high_res"] = torch.cat(
            all_pred_high_res_masks, dim=1
        )
        current_out["multistep_pred_multimasks"] = all_pred_multimasks
        current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
        current_out["multistep_pred_ious"] = all_pred_ious
        current_out["multistep_point_inputs"] = all_point_inputs
        current_out["multistep_object_score_logits"] = all_object_score_logits

        return point_inputs, sam_outputs

    def visualize_segmentation(self, input, backbone_out, pred):
        """
        Visualize medical image segmentation results.
        Saves visualization every N iterations (specified by visualize_interval).
        """
        # Update iteration counter
        self.iter_count += 1

        # Only visualize at specified intervals
        if self.iter_count % self.visualize_interval != 0:
            return

        try:
            # Create visualization directory if it doesn't exist
            os.makedirs(self.visualize_dir, exist_ok=True)

            # Get the first frame for visualization
            frame_idx = 0
            img = input.flat_img_batch[frame_idx]  # Shape: [C, H, W]

            # Get prediction mask (high resolution)
            pred_mask = pred[frame_idx]['pred_masks_high_res'][0, 0]  # Shape: [H, W]

            # Get ground truth mask
            gt_mask = input.masks[frame_idx][0]  # Shape: [H, W]

            # Denormalize image (ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
            img_denorm = img * std + mean
            img_denorm = torch.clamp(img_denorm * 255, 0, 255)
            img_np = img_denorm.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            img_np = np.ascontiguousarray(img_np)

            # Convert to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Convert prediction mask to uint8
            pred_mask_np = (torch.sigmoid(pred_mask) > 0.5).float()
            pred_mask_np = (pred_mask_np * 255).cpu().numpy().astype(np.uint8)

            # Convert ground truth mask to uint8
            gt_mask_np = (gt_mask * 255).cpu().numpy().astype(np.uint8)

            # Create colored overlays
            # Green overlay for prediction
            pred_overlay = img_bgr.copy()
            pred_mask_colored = cv2.applyColorMap(pred_mask_np, cv2.COLORMAP_JET)
            pred_overlay = cv2.addWeighted(pred_overlay, 0.6, pred_mask_colored, 0.4, 0)

            # Blue overlay for ground truth
            gt_overlay = img_bgr.copy()
            gt_mask_colored = cv2.applyColorMap(gt_mask_np, cv2.COLORMAP_JET)
            gt_overlay = cv2.addWeighted(gt_overlay, 0.6, gt_mask_colored, 0.4, 0)

            # Draw points/box if they exist
            # Try to get all iterative points from pred first (includes all correction clicks)
            point_inputs = None
            if 'multistep_point_inputs' in pred[frame_idx]:
                # Get the final step's points (which includes all accumulated points)
                multistep_points = pred[frame_idx]['multistep_point_inputs']
                if isinstance(multistep_points, list) and len(multistep_points) > 0:
                    point_inputs = multistep_points[-1]  # Last step has all accumulated points

            # Fallback to initial points if multistep not available
            if point_inputs is None and frame_idx in backbone_out.get('point_inputs_per_frame', {}):
                point_inputs = backbone_out['point_inputs_per_frame'][frame_idx]

            if point_inputs is not None and 'point_coords' in point_inputs:
                points = point_inputs['point_coords'][0]  # [N, 2]
                labels = point_inputs['point_labels'][0]  # [N]

                # Check if this is a box input (labels 2 and 3 for the first 2 points)
                initial_labels = labels[:2].cpu().numpy() if labels.shape[0] >= 2 else labels.cpu().numpy()
                is_box_input = any(l in [2, 3] for l in initial_labels)

                if is_box_input:
                    # Draw bounding box from first two points (label 2 and 3)
                    box_points = []
                    for i in range(min(2, points.shape[0])):
                        label = labels[i].item()
                        if label in [2, 3]:
                            x, y = int(points[i, 0].item()), int(points[i, 1].item())
                            box_points.append((x, y, label))

                    if len(box_points) >= 2:
                        # Sort to get top-left and bottom-right
                        box_points.sort(key=lambda p: p[2])  # sort by label
                        pt1 = (box_points[0][0], box_points[0][1])  # top-left
                        pt2 = (box_points[1][0], box_points[1][1])  # bottom-right

                        # Draw rectangle (box)
                        cv2.rectangle(img_bgr, pt1, pt2, (0, 255, 0), 2)
                        cv2.rectangle(pred_overlay, pt1, pt2, (0, 255, 0), 2)

                        # Draw corner points
                        cv2.circle(img_bgr, pt1, 5, (255, 0, 0), -1)  # blue for top-left
                        cv2.circle(img_bgr, pt2, 5, (255, 0, 255), -1)  # magenta for bottom-right
                        cv2.circle(pred_overlay, pt1, 5, (255, 0, 0), -1)
                        cv2.circle(pred_overlay, pt2, 5, (255, 0, 255), -1)

                    # Draw correction points (points after the first 2 box corners)
                    if points.shape[0] > 2:
                        for i in range(2, points.shape[0]):
                            x, y = int(points[i, 0].item()), int(points[i, 1].item())
                            label = labels[i].item()
                            # Positive correction points: green, negative: red
                            color = (0, 255, 0) if label == 1 else (0, 0, 255)
                            # Draw with a different style (larger circle with border)
                            cv2.circle(img_bgr, (x, y), 7, color, 2)  # hollow circle
                            cv2.circle(pred_overlay, (x, y), 7, color, 2)
                else:
                    # Draw regular points (no box)
                    for i in range(points.shape[0]):
                        x, y = int(points[i, 0].item()), int(points[i, 1].item())
                        label = labels[i].item()
                        # Positive points (label=1): green circle
                        # Negative points (label=0): red circle
                        color = (0, 255, 0) if label == 1 else (0, 0, 255)
                        # First point: filled circle, correction points: hollow circle
                        if i == 0:
                            cv2.circle(img_bgr, (x, y), 5, color, -1)
                            cv2.circle(pred_overlay, (x, y), 5, color, -1)
                        else:
                            cv2.circle(img_bgr, (x, y), 7, color, 2)
                            cv2.circle(pred_overlay, (x, y), 7, color, 2)

            # Add text labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            # Add point count information
            point_count_text = ''
            if point_inputs is not None and 'point_coords' in point_inputs:
                num_points = point_inputs['point_coords'][0].shape[0]
                point_count_text = f' ({num_points} points)'

            cv2.putText(img_bgr, f'Input Image{point_count_text}', (10, 30), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(pred_overlay, 'Prediction', (10, 30), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(gt_overlay, 'Ground Truth', (10, 30), font, font_scale, (255, 255, 255), thickness)

            combined_image = np.concatenate([img_bgr, pred_overlay, gt_overlay], axis=1)

            os.makedirs(os.path.join(self.visualize_dir, "visualization_images_with_prompt"), exist_ok=True)
            save_path = os.path.join(self.visualize_dir, "visualization_images_with_prompt",f'iter_{self.iter_count:06d}_basic.jpg')
            cv2.imwrite(save_path, combined_image)
            logging.info(f"Basic visualization saved to {save_path}")

            # Loss weight mask visualization
            print('loss_weight_masks' in pred[frame_idx])
            if 'loss_weight_masks' in pred[frame_idx]:
                loss_weight_masks = pred[frame_idx]['loss_weight_masks']
                if loss_weight_masks and loss_weight_masks[0] is not None:
                    loss_weight_mask = loss_weight_masks[-1][0, 0]
                    loss_weight_mask_np = (loss_weight_mask * 255).cpu().numpy().astype(np.uint8)
                    edge_mask_np = ((1.0 - loss_weight_mask) * 255).cpu().numpy().astype(np.uint8)

                    loss_weight_colored = cv2.applyColorMap(loss_weight_mask_np, cv2.COLORMAP_VIRIDIS)
                    edge_colored = cv2.applyColorMap(edge_mask_np, cv2.COLORMAP_HOT)
                    img_with_loss_weight = cv2.addWeighted(img_bgr, 0.6, loss_weight_colored, 0.4, 0)
                    img_with_edge = cv2.addWeighted(img_bgr, 0.6, edge_colored, 0.4, 0)
                    pred_with_loss_weight = cv2.addWeighted(pred_overlay.copy(), 0.6, loss_weight_colored, 0.4, 0)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    cv2.putText(loss_weight_colored, 'Loss Weight Mask', (10, 30), font, font_scale, (255, 255, 255), thickness)
                    cv2.putText(edge_colored, 'Edge Region (No Grad)', (10, 30), font, font_scale, (255, 255, 255), thickness)
                    cv2.putText(img_with_loss_weight, 'Image + Loss Mask', (10, 30), font, font_scale, (255, 255, 255), thickness)
                    cv2.putText(img_with_edge, 'Image + Edge Mask', (10, 30), font, font_scale, (255, 255, 255), thickness)
                    cv2.putText(pred_with_loss_weight, 'Pred + Loss Mask', (10, 30), font, font_scale, (255, 255, 255), thickness)

                    row1 = np.concatenate([loss_weight_colored, edge_colored, img_with_loss_weight], axis=1)
                    row2 = np.concatenate([img_with_edge, pred_with_loss_weight, gt_overlay], axis=1)
                    combined_loss_vis = np.concatenate([row1, row2], axis=0)

                    os.makedirs(os.path.join(self.visualize_dir, "visualizations_loss_masks"), exist_ok=True)
                    save_path_loss = os.path.join(self.visualize_dir, "visualizations_loss_masks",f'iter_{self.iter_count:06d}_loss_mask.jpg')
                    cv2.imwrite(save_path_loss, combined_loss_vis)
                    logging.info(f"Loss mask visualization saved to {save_path_loss}")

        except Exception as e:
            logging.warning(f"Visualization failed: {e}")
