# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List
import os
import logging

import cv2
import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_opening, distance_transform_edt
from skimage.measure import label

from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized


def extract_edge_mask(masks, edge_width=3):
    """
    Extract edge/boundary mask from ground truth masks.
    Args:
        masks: Binary masks of shape [N, C, H, W]
        edge_width: Width of the edge region in pixels
    Returns:
        Edge mask of same shape, where 1=edge region, 0=non-edge region
    """
    import torch.nn.functional as F

    # Use max pooling (dilation) to expand the mask
    kernel_size = 2 * edge_width + 1
    dilated = F.max_pool2d(
        masks,
        kernel_size=kernel_size,
        stride=1,
        padding=edge_width
    )

    # Use -max_pool on inverted mask (equivalent to erosion)
    eroded = -F.max_pool2d(
        -masks,
        kernel_size=kernel_size,
        stride=1,
        padding=edge_width
    )

    # Edge is the difference between dilated and eroded
    edge_mask = (dilated - eroded) > 0

    return edge_mask.float()


# def unet_weight_map(y, wc=None, w0=10, sigma=5):
#     """
#     Generate weight maps for background regions only.
#     Foreground regions get weight 1.0, background regions get calculated weights.

#     Args:
#         y: Binary mask (numpy array) of shape [H, W]
#         wc: Class weights dictionary (optional)
#         w0: Weight parameter for boundary regions
#         sigma: Sigma parameter for Gaussian weighting

#     Returns:
#         Weight map of shape [H, W]
#     """
#     y_separated = binary_opening(y, iterations=2)

#     labels = label(y_separated)

#     no_labels = labels == 0
#     label_ids = sorted(np.unique(labels))[1:]

#     # Initialize weight map with 1.0 for all pixels
#     w = np.ones_like(y, dtype=np.float64)

#     # Only calculate weights for background regions
#     if len(label_ids) > 1:
#         distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

#         for i, label_id in enumerate(label_ids):
#             # 计算到特定标签边界的距离
#             distances[:, :, i] = distance_transform_edt(labels != label_id)

#         distances = np.sort(distances, axis=2)
#         d1 = distances[:, :, 0]
#         d2 = distances[:, :, 1]
#         boundary_weight = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels

#         # Apply boundary weight only to background regions
#         w[no_labels] = 1.0 + boundary_weight[no_labels]

#     # Apply class weights only to background if provided
#     if wc and 0 in wc:
#         w[y == 0] = w[y == 0] * wc[0]

#     return w


# def extract_small_area_mask(masks, w0=10, sigma=5):
#     """
#     Extract foreground regions from ground truth masks using unet_weight_map.

#     Args:
#         masks: Binary masks of shape [N, C, H, W] (PyTorch tensor)
#         w0: Weight parameter for boundary regions
#         sigma: Sigma parameter for Gaussian weighting

#     Returns:
#         Foreground mask of same shape, where 1=foreground region, 0=background
#     """
#     device = masks.device
#     dtype = masks.dtype
#     N, C, H, W = masks.shape

#     small_area_masks = torch.zeros_like(masks)

#     # Process each sample in the batch
#     for n in range(N):
#         for c in range(C):
#             # Convert to numpy
#             mask_np = masks[n, c].cpu().numpy()

#             # Apply unet_weight_map to get weight map
#             weight_map = unet_weight_map(mask_np, wc=None, w0=w0, sigma=sigma)

#             # Extract foreground regions (where mask > 0)
#             small_area = (weight_map > 0).astype(np.float32)

#             # Convert back to torch tensor
#             small_area_masks[n, c] = torch.from_numpy(small_area).to(device=device, dtype=dtype)

#     return small_area_masks


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False, loss_weight_mask=None):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        loss_weight_mask: Optional mask to weight loss spatially [N, 1, H, W]
                          (e.g., to ignore edge regions)
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs_flat = inputs.flatten(2)
        targets_flat = targets.flatten(2)

        if loss_weight_mask is not None:
            # Apply spatial weighting (e.g., zero out edge regions)
            # loss_weight_mask: [N, 1, H, W] -> [N, 1, H*W]
            weight_flat = loss_weight_mask.flatten(2)
            # Expand to match multimask dimension [N, M, H*W]
            weight_flat = weight_flat.expand_as(inputs_flat)
            inputs_flat = inputs_flat * weight_flat
            targets_flat = targets_flat * weight_flat

        numerator = 2 * (inputs_flat * targets_flat).sum(-1)
        denominator = inputs_flat.sum(-1) + targets_flat.sum(-1)
    else:
        if loss_weight_mask is not None:
            inputs = inputs * loss_weight_mask
            targets = targets * loss_weight_mask
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)

    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
    loss_weight_mask=None,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
        loss_weight_mask: Optional mask to weight loss spatially [N, 1, H, W]
                          (e.g., to ignore edge regions)
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_weight_mask is not None:
        # Apply spatial weighting (e.g., zero out edge regions)
        # loss_weight_mask: [N, 1, H, W]
        # Expand to match loss shape
        loss = loss * loss_weight_mask.expand_as(loss)

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects


def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
        ignore_edge_loss=False,
        edge_width=3,
        focal_edge_weight = None,
        focal_edge_smallarea_weight=None, # add small area penalty
        dice_edge_weight=0, # ignore the propagation
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
            ignore_edge_loss: if True, prevent gradient backprop on edge regions (legacy, sets both to 0.0)
            edge_width: width of edge region in pixels
            focal_edge_weight: weight multiplier for edge regions in focal loss
                             - None: no edge weighting (normal training)
                             - 0.0: ignore edges (no gradient on edges)
                             - 2.0: emphasize edges (2x weight on edges)
                             - any float: custom edge weight
            dice_edge_weight: weight multiplier for edge regions in dice loss
                            - None: no edge weighting (normal training)
                            - 0.0: ignore edges (no gradient on edges)
                            - 2.0: emphasize edges (2x weight on edges)
                            - any float: custom edge weight
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.edge_width = edge_width

        # Handle edge weight parameters
        if ignore_edge_loss:
            # Legacy parameter: set both to 0.0
            self.focal_edge_weight = 0.0
            self.dice_edge_weight = 0.0
        else:
            self.focal_edge_weight = focal_edge_weight
            self.dice_edge_weight = dice_edge_weight
        
        self.focal_edge_smallarea_weight = focal_edge_smallarea_weight

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor, weight_maps_batch: torch.Tensor = None):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        if weight_maps_batch is not None:
            for outs, targets, weight_maps in zip(outs_batch, targets_batch, weight_maps_batch):
                cur_losses = self._forward(outs, targets, num_objects, weight_maps)
                for k, v in cur_losses.items():
                    losses[k] += v
        else:
            for outs, targets in zip(outs_batch, targets_batch):
                cur_losses = self._forward(outs, targets, num_objects)
                for k, v in cur_losses.items():
                    losses[k] += v

        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects, weight_maps: torch.Tensor = None):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """

        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]

        # Prepare weight maps if provided
        if weight_maps is not None:
            weight_maps = weight_maps.unsqueeze(1).float()  # [N, 1, H, W]
            assert weight_maps.dim() == 4

        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        last_loss_weight_mask = None  # Store only the last loss weight mask for visualization
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            loss_weight_mask = self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits, weight_maps
            )
            last_loss_weight_mask = loss_weight_mask  # Keep only the last one

        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)

        # Store only the last loss_weight_mask in outputs for visualization
        # Using a list with single element to maintain compatibility with visualization code
        outputs["loss_weight_masks"] = [last_loss_weight_mask] if last_loss_weight_mask is not None else []

        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits, weight_maps=None
    ):
        """
        Compute and update losses with optional edge weighting and pre-computed weight maps.
        Strategy:
        1. Extract edge mask from target masks
        2. Apply pre-computed weight map (if provided) for foreground/background weighting
        3. Set edge regions to edge_weight (typically 0) to prevent gradient backprop on edges
        """
        # Create weight masks for focal and dice losses based on edge_weight parameters
        focal_weight_mask = None
        dice_weight_mask = None

        if self.focal_edge_weight is not None or self.dice_edge_weight is not None or weight_maps is not None:
            # Extract edge mask from target masks
            edge_mask = extract_edge_mask(target_masks, edge_width=self.edge_width)

            # Create focal loss weight mask
            if self.focal_edge_weight is not None or weight_maps is not None:
                # Initialize with weight = 1.0
                focal_weight_mask = torch.ones_like(target_masks)

                # Apply edge weight to edge regions (overrides weight map)
                # This ensures edges always get edge_weight (typically 0) regardless of weight map
                if self.focal_edge_weight is not None:
                    focal_weight_mask = torch.where(
                        edge_mask > 0,
                        torch.full_like(edge_mask, self.focal_edge_weight),
                        focal_weight_mask
                    )
                
                # Apply pre-computed weight map if provided
                if weight_maps is not None:
                    # Resize weight_maps to match target_masks size if needed
                    if weight_maps.shape != target_masks.shape:
                        # weight_maps: [N, C, H, W], resize to target_masks size
                        weight_maps_resized = F.interpolate(
                            weight_maps.float(),
                            size=target_masks.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    else:
                        weight_maps_resized = weight_maps

                    # Use pre-computed weight map (already contains unet distance-based weights)
                    focal_weight_mask = torch.where(
                        weight_maps_resized > 0,
                        torch.full_like(weight_maps_resized, self.focal_edge_smallarea_weight),
                        focal_weight_mask
                    )

            # Create dice loss weight mask
            if self.dice_edge_weight is not None:
                # Non-edge regions: weight = 1.0
                # Edge regions: weight = dice_edge_weight (e.g., 0.0, 2.0, etc.)
                dice_weight_mask = torch.ones_like(target_masks)
                dice_weight_mask = torch.where(
                    edge_mask > 0,
                    torch.full_like(edge_mask, self.dice_edge_weight),
                    dice_weight_mask
                )

        target_masks = target_masks.expand_as(src_masks)

        # get focal, dice and iou loss on all output masks in a prediction step
        # Focal loss: use focal_edge_weight
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
            loss_weight_mask=focal_weight_mask,
        )
        # Dice loss: use dice_edge_weight
        loss_multidice = dice_loss(
            src_masks,
            target_masks,
            num_objects,
            loss_on_multimask=True,
            loss_weight_mask=dice_weight_mask,
        )
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2
        if loss_multimask.size(1) > 1:
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

        # Return loss_weight_mask for visualization
        # return loss_weight_mask
        return 1-edge_mask

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
