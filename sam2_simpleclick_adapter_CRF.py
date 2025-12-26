"""
SAM2 + SimpleClick迭代式推理适配器 + CRF后处理
将SimpleClick的自动点击策略应用到SAM2 Image Predictor上，并使用CRF优化预测结果
"""

import numpy as np
import torch
from typing import Optional, Tuple, List
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
CRF_AVAILABLE = True


class SAM2ClickerAdapterCRF:
    """
    将SimpleClick的Clicker策略适配到SAM2的推理流程，并使用CRF进行后处理优化
    """

    def __init__(self, sam2_predictor, pred_threshold=0.49, use_crf=True,
                 crf_iterations=5, crf_sxy_gaussian=5, crf_compat_gaussian=3,
                 crf_sxy_bilateral=25, crf_srgb_bilateral=5, crf_compat_bilateral=10):
        """
        Args:
            sam2_predictor: SAM2ImagePredictor实例
            pred_threshold: 预测阈值，用于将概率转为二值mask
            use_crf: 是否使用CRF后处理
            crf_iterations: CRF迭代次数
            crf_sxy_gaussian: Gaussian pairwise potential的空间标准差
            crf_compat_gaussian: Gaussian pairwise potential的兼容性
            crf_sxy_bilateral: Bilateral pairwise potential的空间标准差
            crf_srgb_bilateral: Bilateral pairwise potential的颜色标准差
            crf_compat_bilateral: Bilateral pairwise potential的兼容性
        """
        self.predictor = sam2_predictor
        self.pred_threshold = pred_threshold
        self.use_crf = use_crf and CRF_AVAILABLE

        # CRF参数
        self.crf_iterations = crf_iterations
        self.crf_sxy_gaussian = crf_sxy_gaussian
        self.crf_compat_gaussian = crf_compat_gaussian
        self.crf_sxy_bilateral = crf_sxy_bilateral
        self.crf_srgb_bilateral = crf_srgb_bilateral
        self.crf_compat_bilateral = crf_compat_bilateral

        self.gt_mask = None
        self.not_ignore_mask = None
        self.not_clicked_map = None
        self.current_image = None  # 保存当前图像用于CRF

        # 点击历史
        self.clicks_list = []
        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

    def crf_inference(self, image: np.ndarray, probs: np.ndarray,
                     scale_factor: float = 1.0) -> np.ndarray:
        """
        使用CRF对概率图进行优化

        Args:
            image: 原始图像 (H, W, 3), RGB格式, 值范围[0, 255]
            probs: 概率图 (2, H, W), [background_prob, foreground_prob]
            scale_factor: 缩放因子，用于调整CRF参数

        Returns:
            refined_probs: 优化后的概率图 (2, H, W)
        """
        if not self.use_crf:
            return probs

        h, w = image.shape[:2]
        n_labels = 2  # 二分类：背景和前景

        # 创建DenseCRF对象
        d = dcrf.DenseCRF2D(w, h, n_labels)

        # 设置unary energy (从softmax概率转换)
        unary = unary_from_softmax(probs)
        unary = np.ascontiguousarray(unary)
        d.setUnaryEnergy(unary)

        # 确保图像是uint8格式
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        img_c = np.ascontiguousarray(image)

        # 添加pairwise potentials
        # 1. Gaussian pairwise potential (基于位置)
        d.addPairwiseGaussian(
            sxy=self.crf_sxy_gaussian / scale_factor,
            compat=self.crf_compat_gaussian
        )

        # 2. Bilateral pairwise potential (基于位置和颜色)
        d.addPairwiseBilateral(
            sxy=self.crf_sxy_bilateral / scale_factor,
            srgb=self.crf_srgb_bilateral,
            rgbim=np.copy(img_c),
            compat=self.crf_compat_bilateral
        )

        # 执行推理
        Q = d.inference(self.crf_iterations)

        # 转换回(2, H, W)格式
        refined_probs = np.array(Q).reshape((n_labels, h, w))

        return refined_probs

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

    def _prepare_probs_for_crf(self, pred_probs: np.ndarray) -> np.ndarray:
        """
        将SAM2的预测概率转换为CRF所需的格式

        Args:
            pred_probs: SAM2的预测概率 (H, W), 值范围[0, 1]

        Returns:
            probs: (2, H, W) 格式的概率图，[background_prob, foreground_prob]
        """
        # 确保概率在[0, 1]范围内
        pred_probs = np.clip(pred_probs, 1e-7, 1 - 1e-7)

        # 转换为2通道格式
        foreground_prob = pred_probs
        background_prob = 1 - pred_probs

        probs = np.stack([background_prob, foreground_prob], axis=0)

        return probs

    def iterative_predict(
        self,
        image: np.ndarray,
        gt_mask: np.ndarray,
        max_clicks: int = 20,
        target_iou: float = 0.90,
        multimask_output: bool = True,
        verbose: bool = True
    ) -> Tuple[List[float], np.ndarray, List[dict], np.ndarray, int, List[float]]:
        """
        执行迭代式预测（完整的SimpleClick评估流程），使用CRF优化

        Args:
            image: 输入图像 (H, W, 3), RGB格式
            gt_mask: Ground truth mask (H, W)
            max_clicks: 最大点击次数
            target_iou: 目标IoU阈值
            multimask_output: 是否输出多个mask候选
            verbose: 是否打印详细信息

        Returns:
            ious_list: 每次迭代的IoU列表 (使用CRF后的结果)
            final_mask: 最终预测的mask (CRF优化后)
            clicks_history: 点击历史记录
            best_mask: IoU最高的预测mask
            best_click_idx: 达到最高IoU时的点击次数
            ious_no_crf: 不使用CRF时的IoU列表（用于对比）
        """
        # 1. 设置图像和GT
        self.predictor.set_image(image)
        self.set_ground_truth(gt_mask)
        self.current_image = image  # 保存原始图像用于CRF

        # 初始化
        pred_mask = np.zeros_like(gt_mask, dtype=bool)
        ious_list = []
        ious_no_crf = []  # 记录不使用CRF时的IoU
        prev_logits = None

        # 跟踪最佳结果
        best_iou = 0.0
        best_mask = None
        best_click_idx = 0

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

            # 2.5 应用CRF后处理
            if self.use_crf:
                # 准备CRF输入
                probs_for_crf = self._prepare_probs_for_crf(pred_probs)

                # CRF推理
                refined_probs = self.crf_inference(
                    image=self.current_image,
                    probs=probs_for_crf,
                    scale_factor=1.0
                )

                # 提取前景概率
                refined_pred_probs = refined_probs[1]  # 取前景通道

                # 计算不使用CRF时的IoU (用于对比)
                pred_mask_no_crf = pred_probs > self.pred_threshold
                iou_no_crf = self._compute_iou(gt_mask, pred_mask_no_crf)
                ious_no_crf.append(iou_no_crf)

                # 使用CRF优化后的概率
                pred_probs = refined_pred_probs
            else:
                ious_no_crf.append(None)

            # 2.6 转为二值mask并计算IoU
            pred_mask = pred_probs > self.pred_threshold

            print(np.unique(pred_mask==pred_mask_no_crf))
            iou = self._compute_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            # 2.7 更新最佳结果
            if iou > best_iou:
                best_iou = iou
                best_mask = pred_mask.copy()
                best_click_idx = click_idx + 1

            # 2.8 打印信息
            if verbose:
                crf_info = ""
                if self.use_crf and ious_no_crf[-1] is not None:
                    crf_gain = iou - ious_no_crf[-1]
                    crf_info = f", CRF gain: {crf_gain:+.4f}"

                print(f"Click {click_idx + 1}: IoU = {iou:.4f}{crf_info}, "
                      f"{'Positive' if is_positive else 'Negative'} @ {coords}")

            # 2.9 检查是否达到目标
            if iou >= target_iou:
                if verbose:
                    print(f"Reached target IoU {target_iou:.2f} in {click_idx + 1} clicks!")
                break

        return ious_list, pred_mask, self.clicks_list, best_mask, best_click_idx, ious_no_crf

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
    使用示例：如何将SimpleClick策略应用到SAM2上，并使用CRF优化
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

    # 2. 创建适配器 (启用CRF)
    adapter = SAM2ClickerAdapterCRF(
        predictor,
        pred_threshold=0.49,
        use_crf=True,
        crf_iterations=10,
        crf_sxy_gaussian=3,
        crf_compat_gaussian=3,
        crf_sxy_bilateral=83,
        crf_srgb_bilateral=5,
        crf_compat_bilateral=4
    )

    # 3. 加载图像和GT
    image = np.array(Image.open("path/to/image.jpg").convert("RGB"))
    gt_mask = np.array(Image.open("path/to/mask.png"))  # 二值mask

    # 4. 执行迭代预测
    ious_list, final_mask, clicks_history, best_mask, best_click_idx, ious_no_crf = adapter.iterative_predict(
        image=image,
        gt_mask=gt_mask,
        max_clicks=30,
        target_iou=0.90,
        multimask_output=True,
        verbose=True
    )

    # 5. 输出结果
    print(f"\nFinal IoU (with CRF): {ious_list[-1]:.4f}")
    if ious_no_crf[0] is not None:
        print(f"Final IoU (without CRF): {ious_no_crf[-1]:.4f}")
        print(f"CRF improvement: {ious_list[-1] - ious_no_crf[-1]:+.4f}")
    print(f"Number of clicks: {len(clicks_history)}")
    print(f"Best IoU: {max(ious_list):.4f} at click {best_click_idx}")

    return final_mask, clicks_history


if __name__ == "__main__":
    example_usage()