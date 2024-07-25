
from dataclasses import dataclass
from typing import List
import torch
from itertools import accumulate

from src.metrics.intersection_over_union import intersection_over_union


@dataclass
class BoundingBoxInfo:
    img_idx: int
    class_pred: int  # integer from 0 to num_classes-1
    confidence: float
    x_mid: float
    y_mid: float
    width: float
    height: float


def mean_average_precision(
        bb_predictions: List[BoundingBoxInfo],
        bb_gt: List[BoundingBoxInfo],
        num_classes: int = 20,
        th_iou: float = 0.5
):
    """Computes the mean average precision score for bounding box detection (see: https://www.v7labs.com/blog/mean-average-precision)
    """
    # TP: True positive, prediction predicts box with class c, ground truth box of class c exists and has sufficient overlap (iou > iou_th)
    # FP: False positive, prediction predicts box with class c, there is no ground truth box of class c.
    # FN: False negative, prediction doesn't predict box with class c, there is a groudn truth box with class c.
    # TP + FN = all ground truth boxes

    # compute Average Precision (AP) per class
    aps = []
    for c in range(num_classes):
        # get all ground truth bounding boxes of this class
        bb_gt_c = [bb for bb in bb_gt if bb.class_pred == c]
        # if there are no ground truths, we are not interested in this class
        if len(bb_gt_c) == 0:
            continue
        # get all predictions bb for this class
        bb_pr_c = [bb for bb in bb_predictions if bb.class_pred == c]

        # sort predictions bb by confidence. The confidence will be used for the precision-recall curve:
        # first point is highest confidence: precision over recall, second point is: all samples with >= second highest confidence: precision over recall etc.
        bb_pr_c.sort(key=lambda obj: obj.confidence, reverse=True)

        # for each prediction compute the best iou to the ground truth and get precision, recall
        # true positive, false negative for each prediction (either value 0 or 1)
        tps = [0] * len(bb_pr_c)
        fps = [0] * len(bb_pr_c)
        # a ground truth bounding box can only be matched once, higher confidence has priority
        matched_gt_bb = []
        for idx_pr, bb_pr in enumerate(bb_pr_c):
            # only select the ground truth bbs which are in the same image, keep index of original list
            bb_gt_c_img = [
                bb_gt for bb_gt in bb_gt_c if bb_gt.img_idx == bb_pr.img_idx]
            bb_gt_c_img_idxs = [idx for idx in range(
                len(bb_gt_c)) if bb_gt_c[idx].img_idx == bb_pr.img_idx]

            # compute the iou with the ground truth bbs of the same image
            num_gt_bb = len(bb_gt_c_img)
            if num_gt_bb == 0:
                fps[idx_pr] = 1
                continue

            bb_pr_row = [bb_pr.x_mid, bb_pr.y_mid, bb_pr.width, bb_pr.height]
            bb_pr_tens = torch.tensor([bb_pr_row for _ in range(num_gt_bb)])
            bb_gt_c_img_tens = torch.tensor(
                [[bb.x_mid, bb.y_mid, bb.width, bb.height] for bb in bb_gt_c_img])

            ious = intersection_over_union(bb_pr_tens, bb_gt_c_img_tens)

            # if highest iou is > th_iou -> True Positive (TP), otherwise False Positive (FP)
            iou_max, iou_argmax = torch.max(ious, dim=0)
            bb_gt_idx = bb_gt_c_img_idxs[iou_argmax]
            if iou_max > th_iou and bb_gt_idx not in matched_gt_bb:
                tps[idx_pr] = 1
                matched_gt_bb.append(bb_gt_idx)
            else:
                fps[idx_pr] = 1

        # compute precision and recall
        # accumulated true positives, correspond to the number of true positives by descending confidence
        tps_acc = torch.tensor(list(accumulate(tps)))
        fns_acc = torch.tensor(list(accumulate(fps)))
        # total number of ground truth bounding boxes of this class
        num_bb_gt = len(bb_gt_c)
        precisions = tps_acc / (fns_acc + tps_acc)
        recalls = tps_acc / (num_bb_gt+1e-8)
        # add datapoint at recall is 0 and precision is 1
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # get average precision as area under precision-recall curve by using trapezoidal integration
        aps.append(torch.trapezoid(precisions, recalls))

    # average by classes to geat mean average precision
    mAP = sum(aps) / len(aps)

    return mAP


def mAP_multiple_thresholds(bb_predictions: List[BoundingBoxInfo],
                            bb_gt: List[BoundingBoxInfo],
                            num_classes: int = 20,
                            th_low: float = 0.5, th_high: float = 0.95, th_step: float = 0.05):
    # TODO
    pass
