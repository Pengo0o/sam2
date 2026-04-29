# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the sav_dataset directory of this source tree.

# adapted from https://github.com/hkchengrex/vos-benchmark
# and  https://github.com/davisvideochallenge/davis2017-evaluation
# with their licenses found in the LICENSE_VOS_BENCHMARK and LICENSE_DAVIS files
# in the sav_dataset directory.
#
# Extended to support hard/easy segmentation:
# Each video's frames are divided into 3 equal temporal segments.
# The segment with the lowest mean IoU is labeled "hard";
# the other two segments are labeled "easy".
# Results are reported separately for hard, easy, and combined.

import math
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from os import path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tqdm
from PIL import Image
from skimage.morphology import disk


class VideoEvaluator:
    def __init__(self, gt_root, pred_root, skip_first_and_last=True) -> None:
        """
        gt_root: path to the folder storing the gt masks
        pred_root: path to the folder storing the predicted masks
        skip_first_and_last: whether we should skip the evaluation of the first and the last frame.
                             True for SA-V val and test, same as in DAVIS semi-supervised evaluation.
        """
        self.gt_root = gt_root
        self.pred_root = pred_root
        self.skip_first_and_last = skip_first_and_last

    def __call__(
        self, vid_name: str
    ) -> Tuple[str, Dict[str, List[float]], Dict[str, List[float]]]:
        """
        vid_name: name of the video to evaluate

        Returns:
            vid_name: the video name
            hard_metrics: dict with keys 'iou', 'f', 'biou', each a flat list of float values
                          from the temporal segment with the lowest mean IoU
            easy_metrics: same structure, from the two segments with higher mean IoU
        """
        to_evaluate, is_sav_format = self.scan_vid_folder(vid_name)

        # seg_data[seg_idx] = {'iou': [], 'f': [], 'biou': []}
        seg_data = {0: {"iou": [], "f": [], "biou": []},
                    1: {"iou": [], "f": [], "biou": []},
                    2: {"iou": [], "f": [], "biou": []}}

        for all_frames, obj_id, gt_path, pred_path in to_evaluate:
            if self.skip_first_and_last:
                all_frames = all_frames[1:-1]

            n = len(all_frames)
            if n == 0:
                continue

            # Divide frames into 3 temporal segments
            # seg0: [0, n//3), seg1: [n//3, 2*n//3), seg2: [2*n//3, n)
            boundaries = [0, n // 3, 2 * n // 3, n]
            segments = [
                all_frames[boundaries[i] : boundaries[i + 1]] for i in range(3)
            ]

            for seg_idx, seg_frames in enumerate(segments):
                if len(seg_frames) == 0:
                    continue

                evaluator = Evaluator(name=vid_name, obj_id=obj_id)
                for frame in seg_frames:
                    gt_array, pred_array = self.get_gt_and_pred(
                        gt_path, pred_path, frame, is_sav_format
                    )
                    evaluator.feed_frame(mask=pred_array, gt=gt_array)

                iou, boundary_f, boundary_iou = evaluator.conclude()
                seg_data[seg_idx]["iou"].extend(list(iou.values()))
                seg_data[seg_idx]["f"].extend(list(boundary_f.values()))
                seg_data[seg_idx]["biou"].extend(list(boundary_iou.values()))

        # Determine which segments are non-empty
        non_empty = {idx for idx, d in seg_data.items() if len(d["iou"]) > 0}

        if len(non_empty) == 0:
            # No data at all — return empty metrics
            empty = {"iou": [], "f": [], "biou": []}
            return vid_name, empty, empty

        if len(non_empty) == 1:
            # Only one segment has data — treat it as hard, easy is empty
            only_idx = next(iter(non_empty))
            hard_metrics = seg_data[only_idx]
            easy_metrics = {"iou": [], "f": [], "biou": []}
            return vid_name, hard_metrics, easy_metrics

        # Compute mean IoU per segment; use only non-empty segments
        mean_iou_per_seg = {
            idx: float(np.mean(seg_data[idx]["iou"])) for idx in non_empty
        }

        # The segment with the lowest mean IoU is hard
        hard_seg_idx = min(mean_iou_per_seg, key=mean_iou_per_seg.get)

        hard_metrics = seg_data[hard_seg_idx]
        easy_metrics = {"iou": [], "f": [], "biou": []}
        for idx in non_empty:
            if idx != hard_seg_idx:
                easy_metrics["iou"].extend(seg_data[idx]["iou"])
                easy_metrics["f"].extend(seg_data[idx]["f"])
                easy_metrics["biou"].extend(seg_data[idx]["biou"])

        return vid_name, hard_metrics, easy_metrics

    def get_gt_and_pred(
        self,
        gt_path: str,
        pred_path: str,
        f_name: str,
        is_sav_format: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the ground-truth and predicted masks for a single frame.
        """
        gt_mask_path = path.join(gt_path, f_name)
        pred_mask_path = path.join(pred_path, f_name)
        assert os.path.exists(pred_mask_path), f"{pred_mask_path} not found"

        gt_array = np.array(Image.open(gt_mask_path))
        pred_array = np.array(Image.open(pred_mask_path))
        assert (
            gt_array.shape[-2:] == pred_array.shape[-2:]
        ), f"shape mismatch: {gt_mask_path}, {pred_mask_path}"

        if is_sav_format:
            assert len(np.unique(gt_array)) <= 2, (
                f"found more than 1 object in {gt_mask_path} "
                "SA-V format assumes one object mask per png file."
            )
            assert len(np.unique(pred_array)) <= 2, (
                f"found more than 1 object in {pred_mask_path} "
                "SA-V format assumes one object mask per png file."
            )
            gt_array = gt_array > 0
            pred_array = pred_array > 0

        return gt_array, pred_array

    def scan_vid_folder(self, vid_name) -> Tuple[List, bool]:
        """
        Scan the folder structure of the video and return a list of folders for evaluate.
        """
        vid_gt_path = path.join(self.gt_root, vid_name)
        vid_pred_path = path.join(self.pred_root, vid_name)
        all_files_and_dirs = sorted(os.listdir(vid_gt_path))
        to_evaluate = []
        if all(name.endswith(".png") for name in all_files_and_dirs):
            # All files are png files, dataset structure similar to DAVIS
            is_sav_format = False
            frames = all_files_and_dirs
            obj_dir = None
            to_evaluate.append((frames, obj_dir, vid_gt_path, vid_pred_path))
        else:
            # SA-V dataset structure, going one layer down into each subdirectory
            is_sav_format = True
            for obj_dir in all_files_and_dirs:
                obj_gt_path = path.join(vid_gt_path, obj_dir)
                obj_pred_path = path.join(vid_pred_path, obj_dir)
                frames = sorted(os.listdir(obj_gt_path))
                to_evaluate.append((frames, obj_dir, obj_gt_path, obj_pred_path))
        return to_evaluate, is_sav_format


#################################################################################################################
# Functions below are from https://github.com/hkchengrex/vos-benchmark with minor modifications
# _seg2bmap from https://github.com/hkchengrex/vos-benchmark/blob/main/vos_benchmark/utils.py
# get_iou and Evaluator from https://github.com/hkchengrex/vos-benchmark/blob/main/vos_benchmark/evaluator.py
#################################################################################################################


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Cannot convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def get_iou(intersection, pixel_sum):
    # handle edge cases without resorting to epsilon
    if intersection == pixel_sum:
        # both mask and gt have zero pixels in them
        assert intersection == 0
        return 1

    return intersection / (pixel_sum - intersection)


class Evaluator:
    def __init__(self, boundary=0.008, name=None, obj_id=None):
        # boundary: used in computing boundary F-score and boundary IoU
        self.boundary = boundary
        self.name = name
        self.obj_id = obj_id
        self.objects_in_gt = set()
        self.objects_in_masks = set()

        self.object_iou = defaultdict(list)
        self.boundary_f = defaultdict(list)
        self.boundary_iou = defaultdict(list)

    def feed_frame(self, mask: np.ndarray, gt: np.ndarray):
        """
        Compute and accumulate metrics for a single frame (mask/gt pair)
        """

        # get all objects in the ground-truth
        gt_objects = np.unique(gt)
        gt_objects = gt_objects[gt_objects != 0].tolist()

        # get all objects in the predicted mask
        mask_objects = np.unique(mask)
        mask_objects = mask_objects[mask_objects != 0].tolist()

        self.objects_in_gt.update(set(gt_objects))
        self.objects_in_masks.update(set(mask_objects))

        all_objects = self.objects_in_gt.union(self.objects_in_masks)

        # boundary disk for boundary F-score. It is the same for all objects.
        bound_pix = np.ceil(self.boundary * np.linalg.norm(mask.shape))
        boundary_disk = disk(bound_pix)

        for obj_idx in all_objects:
            obj_mask = mask == obj_idx
            obj_gt = gt == obj_idx

            # object iou
            self.object_iou[obj_idx].append(
                get_iou((obj_mask * obj_gt).sum(), obj_mask.sum() + obj_gt.sum())
            )
            """
            # boundary f-score
            This part is copied from davis2017-evaluation
            """
            mask_boundary = _seg2bmap(obj_mask)
            gt_boundary = _seg2bmap(obj_gt)
            mask_dilated = cv2.dilate(mask_boundary.astype(np.uint8), boundary_disk)
            gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), boundary_disk)

            # Get the intersection
            gt_match = gt_boundary * mask_dilated
            fg_match = mask_boundary * gt_dilated

            # Area of the intersection
            n_fg = np.sum(mask_boundary)
            n_gt = np.sum(gt_boundary)

            # Compute precision and recall
            if n_fg == 0 and n_gt > 0:
                precision = 1
                recall = 0
            elif n_fg > 0 and n_gt == 0:
                precision = 0
                recall = 1
            elif n_fg == 0 and n_gt == 0:
                precision = 1
                recall = 1
            else:
                precision = np.sum(fg_match) / float(n_fg)
                recall = np.sum(gt_match) / float(n_gt)

            # Compute F measure
            if precision + recall == 0:
                F = 0
            else:
                F = 2 * precision * recall / (precision + recall)
            self.boundary_f[obj_idx].append(F)

            # Compute Boundary IoU: IoU of dilated boundary regions
            b_intersection = np.sum(mask_dilated * gt_dilated)
            b_sum = np.sum(mask_dilated) + np.sum(gt_dilated)
            if b_sum == 0:
                # both masks are empty
                b_iou = 1.0
            else:
                b_iou = float(b_intersection) / float(b_sum - b_intersection)
            self.boundary_iou[obj_idx].append(b_iou)

    def conclude(self):
        all_iou = {}
        all_boundary_f = {}
        all_boundary_iou = {}

        for object_id in self.objects_in_gt:
            all_iou[object_id] = np.mean(self.object_iou[object_id]) * 100
            all_boundary_f[object_id] = np.mean(self.boundary_f[object_id]) * 100
            all_boundary_iou[object_id] = np.mean(self.boundary_iou[object_id]) * 100

        return all_iou, all_boundary_f, all_boundary_iou


def benchmark(
    gt_roots,
    mask_roots,
    strict=True,
    num_processes=None,
    *,
    verbose=True,
    skip_first_and_last=True,
):
    """
    gt_roots: a list of paths to datasets, i.e., [path_to_DatasetA, path_to_DatasetB, ...]
    mask_roots: same as above, but the .png are masks predicted by the model
    strict: when True, all videos in the dataset must have corresponding predictions.
    skip_first_and_last: whether we should skip the first and the last frame in evaluation.

    Each video's frames are divided into 3 equal temporal segments.
    The segment with the lowest mean IoU is labeled "hard";
    the other two are labeled "easy".
    Results are reported for hard, easy, and combined.
    """

    assert len(gt_roots) == len(mask_roots)
    single_dataset = len(gt_roots) == 1

    if verbose:
        if skip_first_and_last:
            print(
                "We are *SKIPPING* the evaluation of the first and the last frame (standard for semi-supervised video object segmentation)."
            )
        else:
            print(
                "We are *NOT SKIPPING* the evaluation of the first and the last frame (*NOT STANDARD* for semi-supervised video object segmentation)."
            )
        print(
            "Hard/Easy split: each video is divided into 3 temporal segments; "
            "the segment with the lowest mean IoU is 'hard', the other two are 'easy'."
        )

    pool = Pool(num_processes)
    start = time.time()
    to_wait = []
    for gt_root, mask_root in zip(gt_roots, mask_roots):
        # Validate folders
        validated = True
        gt_videos = os.listdir(gt_root)
        mask_videos = os.listdir(mask_root)

        # if the user passed the root directory instead of Annotations
        if len(gt_videos) != len(mask_videos):
            if "Annotations" in gt_videos:
                if ".png" not in os.listdir(path.join(gt_root, "Annotations"))[0]:
                    gt_root = path.join(gt_root, "Annotations")
                    gt_videos = os.listdir(gt_root)

        # remove non-folder items
        gt_videos = list(filter(lambda x: path.isdir(path.join(gt_root, x)), gt_videos))
        mask_videos = list(
            filter(lambda x: path.isdir(path.join(mask_root, x)), mask_videos)
        )

        if not strict:
            videos = sorted(list(set(gt_videos) & set(mask_videos)))
        else:
            gt_extras = set(gt_videos) - set(mask_videos)
            mask_extras = set(mask_videos) - set(gt_videos)

            if len(gt_extras) > 0:
                print(
                    f"Videos that are in {gt_root} but not in {mask_root}: {gt_extras}"
                )
                validated = False
            if len(mask_extras) > 0:
                print(
                    f"Videos that are in {mask_root} but not in {gt_root}: {mask_extras}"
                )
                validated = False
            if not validated:
                print("Validation failed. Exiting.")
                exit(1)

            videos = sorted(gt_videos)

        if verbose:
            print(
                f"In dataset {gt_root}, we are evaluating on {len(videos)} videos: {videos}"
            )

        if single_dataset:
            if verbose:
                results = tqdm.tqdm(
                    pool.imap(
                        VideoEvaluator(
                            gt_root, mask_root, skip_first_and_last=skip_first_and_last
                        ),
                        videos,
                    ),
                    total=len(videos),
                )
            else:
                results = pool.map(
                    VideoEvaluator(
                        gt_root, mask_root, skip_first_and_last=skip_first_and_last
                    ),
                    videos,
                )
        else:
            to_wait.append(
                pool.map_async(
                    VideoEvaluator(
                        gt_root, mask_root, skip_first_and_last=skip_first_and_last
                    ),
                    videos,
                )
            )

    pool.close()

    all_global_results = []
    all_object_metrics = []

    for i, mask_root in enumerate(mask_roots):
        if not single_dataset:
            results = to_wait[i].get()

        hard_iou, hard_f, hard_biou = [], [], []
        easy_iou, easy_f, easy_biou = [], [], []
        video_metrics = {}  # vid_name -> (hard_metrics, easy_metrics)

        for vid_name, hard_metrics, easy_metrics in results:
            hard_iou.extend(hard_metrics["iou"])
            hard_f.extend(hard_metrics["f"])
            hard_biou.extend(hard_metrics["biou"])
            easy_iou.extend(easy_metrics["iou"])
            easy_f.extend(easy_metrics["f"])
            easy_biou.extend(easy_metrics["biou"])
            video_metrics[vid_name] = (hard_metrics, easy_metrics)

        def _mean(lst):
            return float(np.mean(lst)) if len(lst) > 0 else float("nan")

        # Hard scores
        h_j = _mean(hard_iou)
        h_f = _mean(hard_f)
        h_jf = (h_j + h_f) / 2
        h_biou = _mean(hard_biou)

        # Easy scores
        e_j = _mean(easy_iou)
        e_f = _mean(easy_f)
        e_jf = (e_j + e_f) / 2
        e_biou = _mean(easy_biou)

        # Combined scores
        all_iou = hard_iou + easy_iou
        all_f = hard_f + easy_f
        all_biou = hard_biou + easy_biou
        c_j = _mean(all_iou)
        c_f = _mean(all_f)
        c_jf = (c_j + c_f) / 2
        c_biou = _mean(all_biou)

        time_taken = time.time() - start

        # Build output string
        out_string = f'{"category":<8}, {"J&F":>4}, {"J":>4}, {"F":>4}, {"B_IoU":>5}\n'
        out_string += f'{"hard":<8}, {h_jf:>4.1f}, {h_j:>4.1f}, {h_f:>4.1f}, {h_biou:>5.1f}\n'
        out_string += f'{"easy":<8}, {e_jf:>4.1f}, {e_j:>4.1f}, {e_f:>4.1f}, {e_biou:>5.1f}\n'
        out_string += f'{"combined":<8}, {c_jf:>4.1f}, {c_j:>4.1f}, {c_f:>4.1f}, {c_biou:>5.1f}\n'

        if verbose:
            print("\n--- Hard/Easy/Combined Global Scores ---")
            print(out_string.replace(",", " "), end="")

            # Per-video detail
            ml = max(*[len(n) for n in video_metrics.keys()], len("sequence"))
            detail = f'\n{"sequence":<{ml}}, {"category":<8}, {"J&F":>4}, {"J":>4}, {"F":>4}, {"B_IoU":>5}\n'
            for vid_name, (hm, em) in sorted(video_metrics.items()):
                for cat, m in [("hard", hm), ("easy", em)]:
                    if len(m["iou"]) == 0:
                        continue
                    vj = _mean(m["iou"])
                    vf = _mean(m["f"])
                    vbiou = _mean(m["biou"])
                    vjf = (vj + vf) / 2
                    detail += (
                        f"{vid_name:<{ml}}, {cat:<8}, {vjf:>4.1f}, {vj:>4.1f}, {vf:>4.1f}, {vbiou:>5.1f}\n"
                    )
            print(detail, end="")

            print("\nSummary:")
            print(f"  hard    : J&F: {h_jf:.1f}  J: {h_j:.1f}  F: {h_f:.1f}  B_IoU: {h_biou:.1f}")
            print(f"  easy    : J&F: {e_jf:.1f}  J: {e_j:.1f}  F: {e_f:.1f}  B_IoU: {e_biou:.1f}")
            print(f"  combined: J&F: {c_jf:.1f}  J: {c_j:.1f}  F: {c_f:.1f}  B_IoU: {c_biou:.1f}")
            print(f"Time taken: {time_taken:.2f}s")

        # Save to CSV
        result_path = path.join(mask_root, "results_hard_easy.csv")
        print(f"Saving the results to {result_path}")
        with open(result_path, "w") as fp:
            fp.write(out_string)

        global_result = {
            "hard":     {"jf": h_jf, "j": h_j, "f": h_f, "biou": h_biou},
            "easy":     {"jf": e_jf, "j": e_j, "f": e_f, "biou": e_biou},
            "combined": {"jf": c_jf, "j": c_j, "f": c_f, "biou": c_biou},
        }
        all_global_results.append(global_result)
        all_object_metrics.append(video_metrics)

    return all_global_results, all_object_metrics
