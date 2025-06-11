# The AJI and PQ metrics – adapted from HoVer-Net:
# https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045?via%3Dihub

import numpy as np
from skimage.measure import label
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_score, recall_score, f1_score


def to_numpy(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return x


def TP(im1, label) -> float:
    return float(np.logical_and(im1, label).sum())


def FP(im1, label) -> float:
    return float(np.logical_and(im1, np.logical_not(label)).sum())


def FN(im1, label) -> float:
    return float(np.logical_and(np.logical_not(im1), label).sum())


def TN(im1, label) -> float:
    area: int = im1.size
    return float(area - np.logical_or(im1, label).sum())


def judge_image(img1, label) -> None:
    if img1.shape != label.shape:
        raise ValueError(
            "Shape mismatch: im1 and label must have the same shape.")
    if label.max() > 1:
        raise ValueError("Label mask must be binary.")


def cutoff(img1, label, c: float):
    im1 = np.squeeze(img1 > c)
    label = np.squeeze(label > c)
    return im1, label


def _prepare_binary_inputs(img1, label, c: float):
    judge_image(img1, label)
    return cutoff(img1, label, c)


def compute_iou(img1, label, c: float) -> float:
    im1, label = _prepare_binary_inputs(img1, label, c)
    return TP(im1, label) / float(np.logical_or(im1, label).sum())


def compute_precision(img1, label, c: float) -> float:
    im1, label = _prepare_binary_inputs(img1, label, c)
    return float(precision_score(label.flatten(), im1.flatten(), average='micro'))


def compute_recall(img1, label, c: float) -> float:
    im1, label = _prepare_binary_inputs(img1, label, c)
    return float(recall_score(label.flatten(), im1.flatten(), average='micro'))


def compute_F1(img1, label, c: float) -> float:
    im1, label = _prepare_binary_inputs(img1, label, c)

    im1_np = to_numpy(im1).flatten()
    label_np = to_numpy(label).flatten()

    return float(f1_score(label_np, im1_np, average='micro'))


def compute_TP_ratio(img1, label, c: float) -> float:
    im1, label = _prepare_binary_inputs(img1, label, c)
    return TP(im1, label) / float(label.sum())


def get_dice_1(true, pred) -> float:
    true = (true > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)
    inter = true * pred
    denom = true + pred
    return 2.0 * np.sum(inter) / np.sum(denom)


def get_dice_2(true, pred) -> float:
    true = label(true, background=0)
    pred = label(pred, background=0)
    true_id = list(np.unique(np.asarray(true)))
    pred_id = list(np.unique(np.asarray(pred)))

    if 0 in true_id:
        true_id.remove(0)
    if 0 in pred_id:
        pred_id.remove(0)

    total_markup = 0
    total_intersect = 0
    for t in true_id:
        t_mask = (true == t).astype(np.uint8)
        for p in pred_id:
            p_mask = (pred == p).astype(np.uint8)
            intersect = p_mask * t_mask
            if intersect.sum() > 0:
                total_intersect += intersect.sum()
                total_markup += (t_mask.sum() + p_mask.sum())

    return 2 * total_intersect / total_markup


def get_fast_aji(true, pred) -> float:
    true = label(true, background=0).astype(np.uint8)
    pred = label(pred, background=0).astype(np.uint8)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    if len(true_id_list) <= 1 and len(pred_id_list) <= 1:
        return 1.0 if (true == pred).all() else 0.0
    if len(true_id_list) <= 1 or len(pred_id_list) <= 1:
        return 0.0

    true_masks = [None] + [(true == t).astype(np.uint8)
                           for t in true_id_list[1:]]
    pred_masks = [None] + [(pred == p).astype(np.uint8)
                           for p in pred_id_list[1:]]

    pairwise_inter = np.zeros(
        (len(true_id_list) - 1, len(pred_id_list) - 1), dtype=np.float64)
    pairwise_union = np.zeros_like(pairwise_inter)

    for t_idx, true_id in enumerate(true_id_list[1:]):
        t_mask = true_masks[true_id]
        overlap_pred_ids = np.unique(pred[t_mask > 0])
        for pred_id in overlap_pred_ids:
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id]
            inter = (t_mask * p_mask).sum()
            union = (t_mask + p_mask).sum() - inter
            pairwise_inter[t_idx, pred_id - 1] = inter
            pairwise_union[t_idx, pred_id - 1] = union

    if pairwise_inter.size == 0 or pairwise_union.size == 0:
        return 0.0

    pairwise_iou = pairwise_inter / (pairwise_union + 1e-6)

    if pairwise_iou.size == 0:
        return 0.0

    paired_pred = np.argmax(pairwise_iou, axis=1)
    max_iou = np.max(pairwise_iou, axis=1)
    paired_true = np.nonzero(max_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]

    overall_inter = pairwise_inter[paired_true, paired_pred].sum()
    overall_union = pairwise_union[paired_true, paired_pred].sum()

    paired_true_ids = [i + 1 for i in paired_true]
    paired_pred_ids = [i + 1 for i in paired_pred]
    unpaired_true_ids = [idx for idx in true_id_list[1:]
                         if idx not in paired_true_ids]
    unpaired_pred_ids = [idx for idx in pred_id_list[1:]
                         if idx not in paired_pred_ids]

    for t_id in unpaired_true_ids:
        overall_union += true_masks[t_id].sum()
    for p_id in unpaired_pred_ids:
        overall_union += pred_masks[p_id].sum()

    return overall_inter / overall_union if overall_union > 0 else 0.0


def get_fast_pq(true, pred, match_iou: float = 0.5):
    assert match_iou >= 0.0
    true = label(true, background=0)
    pred = label(pred, background=0)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None] + [(true == t).astype(np.uint8)
                           for t in true_id_list[1:]]
    pred_masks = [None] + [(pred == p).astype(np.uint8)
                           for p in pred_id_list[1:]]

    pairwise_iou = np.zeros(
        (len(true_id_list) - 1, len(pred_id_list) - 1), dtype=np.float64)

    for t_idx, true_id in enumerate(true_id_list[1:]):
        t_mask = true_masks[true_id]
        overlap_pred_ids = np.unique(pred[t_mask > 0])
        for pred_id in overlap_pred_ids:
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id]
            inter = (t_mask * p_mask).sum()
            union = (t_mask + p_mask).sum() - inter
            pairwise_iou[t_idx, pred_id - 1] = inter / union

    if match_iou >= 0.5:
        paired_true, paired_pred = np.nonzero(pairwise_iou > match_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1
        paired_pred += 1
    else:
        t_ind, p_ind = linear_sum_assignment(-pairwise_iou)
        ious = pairwise_iou[t_ind, p_ind]
        keep = ious > match_iou
        paired_true = (t_ind[keep] + 1).tolist()
        paired_pred = (p_ind[keep] + 1).tolist()
        paired_iou = ious[keep]

    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)

    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    sq = paired_iou.sum() / (tp + 1e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]
