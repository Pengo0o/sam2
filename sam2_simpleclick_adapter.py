"""
SAM2 + SimpleClick迭代式推理适配器
将SimpleClick的自动点击策略应用到SAM2 Image Predictor上
"""

import numpy as np
import torch
from typing import Optional, Tuple, List
import cv2


class SAM2ClickerAdapter:
    """
    将SimpleClick的Clicker策略适配到SAM2的推理流程
    """

    def __init__(self, sam2_predictor, pred_threshold=0.49):
        """
        Args:
            sam2_predictor: SAM2ImagePredictor实例
            pred_threshold: 预测阈值，用于将概率转为二值mask
        """
        self.predictor = sam2_predictor
        self.pred_threshold = pred_threshold
        self.gt_mask = None
        self.not_ignore_mask = None
        self.not_clicked_map = None

        # 点击历史
        self.clicks_list = []
        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

    def set_ground_truth(self, gt_mask: np.ndarray, ignore_label=-1):
        """
        设置ground truth mask（用于自动生成点击）

        Args:
            gt_mask: Ground truth mask (H, W)
            ignore_label: 忽略的标签值
        """
        self.gt_mask = gt_mask == 1
        self.not_ignore_mask = gt_mask != ignore_label
        self.not_clicked_map = np.ones_like(self.gt_mask, dtype=bool)
        self.reset_clicks()

    def reset_clicks(self):
        """重置点击历史"""
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=bool)
        self.clicks_list = []
        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

    def _get_next_click(self, pred_mask: np.ndarray) -> Tuple[bool, Tuple[int, int]]:
        """
        根据预测结果自动生成下一个点击
        使用SimpleClick的距离变换策略

        Args:
            pred_mask: 当前预测的二值mask (H, W)

        Returns:
            (is_positive, (y, x)): 点击类型和坐标
        """
        # FN: GT是前景但预测为背景的区域
        fn_mask = np.logical_and(
            np.logical_and(self.gt_mask, np.logical_not(pred_mask)),
            self.not_ignore_mask
        )

        # FP: GT是背景但预测为前景的区域
        fp_mask = np.logical_and(
            np.logical_and(np.logical_not(self.gt_mask), pred_mask),
            self.not_ignore_mask
        )

        # Padding以避免边界效应
        fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
        fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        # 距离变换：找到离边界最远的错误点
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        # 去除padding
        fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
        fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        # 避免重复点击
        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        # 选择距离更大的错误类型
        is_positive = fn_max_dist > fp_max_dist

        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)

        coords = (int(coords_y[0]), int(coords_x[0]))

        return is_positive, coords

    def add_click(self, is_positive: bool, coords: Tuple[int, int]):
        """
        添加点击到历史记录

        Args:
            is_positive: 是否为正点击
            coords: 点击坐标 (y, x)
        """
        self.clicks_list.append({
            'is_positive': is_positive,
            'coords': coords
        })

        if is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def get_sam2_prompts(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        将点击历史转换为SAM2的输入格式

        Returns:
            point_coords: (N, 2) 数组，格式为 (x, y) - 注意SAM2使用 (x,y) 顺序
            point_labels: (N,) 数组，1表示正点击，0表示负点击
        """
        if not self.clicks_list:
            return None, None

        # 注意：SAM2使用 (x, y) 顺序，而Clicker使用 (y, x)
        point_coords = np.array([
            [click['coords'][1], click['coords'][0]]  # (x, y)
            for click in self.clicks_list
        ], dtype=np.float32)

        point_labels = np.array([
            1 if click['is_positive'] else 0
            for click in self.clicks_list
        ], dtype=np.int32)

        return point_coords, point_labels

    def iterative_predict(
        self,
        image: np.ndarray,
        gt_mask: np.ndarray,
        max_clicks: int = 20,
        target_iou: float = 0.90,
        multimask_output: bool = True
    ) -> Tuple[List[float], np.ndarray, List[dict]]:
        """
        执行迭代式预测（完整的SimpleClick评估流程）

        Args:
            image: 输入图像 (H, W, 3)
            gt_mask: Ground truth mask (H, W)
            max_clicks: 最大点击次数
            target_iou: 目标IoU阈值
            multimask_output: 是否输出多个mask候选

        Returns:
            ious_list: 每次迭代的IoU列表
            final_mask: 最终预测的mask
            clicks_history: 点击历史记录
        """
        # 1. 设置图像和GT
        self.predictor.set_image(image)
        self.set_ground_truth(gt_mask)

        # 初始化
        pred_mask = np.zeros_like(gt_mask, dtype=bool)
        ious_list = []
        prev_logits = None

        # 2. 迭代预测
        for click_idx in range(max_clicks):
            # 2.1 根据当前预测生成下一个点击
            is_positive, coords = self._get_next_click(pred_mask)
            self.add_click(is_positive, coords)

            # 2.2 获取SAM2格式的prompts
            point_coords, point_labels = self.get_sam2_prompts()

            # 2.3 调用SAM2预测
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=prev_logits,  # 传入上一次的logits
                multimask_output=multimask_output
            )

            # 2.4 选择最佳mask（如果有多个候选）
            if multimask_output or click_idx == 0:
                # 第一次点击：选择得分最高的mask
                best_idx = np.argmax(scores)
                pred_probs = masks[best_idx]
                prev_logits = logits[best_idx:best_idx+1, :, :]
            else:
                # 后续迭代：只有一个mask
                pred_probs = masks[0] if len(masks.shape) == 3 else masks
                prev_logits = logits[0:1, :, :] if len(logits.shape) == 4 else logits

            # 2.5 转为二值mask并计算IoU
            pred_mask = pred_probs > self.pred_threshold
            iou = self._compute_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            print(f"Click {click_idx + 1}: IoU = {iou:.4f}, "
                  f"{'Positive' if is_positive else 'Negative'} @ {coords}")

            # 2.6 检查是否达到目标
            if iou >= target_iou:
                print(f"Reached target IoU {target_iou:.2f} in {click_idx + 1} clicks!")
                break

        return ious_list, pred_mask, self.clicks_list

    @staticmethod
    def _compute_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """
        计算IoU指标

        Args:
            gt_mask: Ground truth mask
            pred_mask: Predicted mask

        Returns:
            iou: Intersection over Union
        """
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return intersection / union


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """
    使用示例：如何将SimpleClick策略应用到SAM2上
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from PIL import Image

    # 1. 初始化SAM2模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2_model = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
        ckpt_path="checkpoints/sam2.1_hiera_large.pt",
        device=device
    )
    predictor = SAM2ImagePredictor(sam2_model)

    # 2. 创建适配器
    adapter = SAM2ClickerAdapter(predictor, pred_threshold=0.49)

    # 3. 加载图像和GT
    image = np.array(Image.open("path/to/image.jpg").convert("RGB"))
    gt_mask = np.array(Image.open("path/to/mask.png"))  # 二值mask

    # 4. 执行迭代预测
    ious_list, final_mask, clicks_history = adapter.iterative_predict(
        image=image,
        gt_mask=gt_mask,
        max_clicks=30,
        target_iou=0.90,
        multimask_output=True
    )

    # 5. 输出结果
    print(f"\nFinal IoU: {ious_list[-1]:.4f}")
    print(f"Number of clicks: {len(clicks_history)}")
    print(f"IoU progression: {ious_list}")

    return final_mask, clicks_history


if __name__ == "__main__":
    example_usage()
