import os
import glob
import numpy as np
import cv2 as cv
import tqdm


def parse_ccpd_bbox(filename):
    """CCPD names look like: 025-95_113-154&383_386&473-...jpg
    Field 3 is the plate box: x1&y1_x2&y2
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("-")
    if len(parts) < 3:
        return None
    try:
        left, right = parts[2].split("_")
        x1, y1 = left.split("&")
        x2, y2 = right.split("&")
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
        return np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)
    except Exception:
        return None


def corners_to_xyxy(det):
    """LPD_YuNet returns 4 corners + score: x1 y1 x2 y2 x3 y3 x4 y4 score."""
    if det is None or len(det) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    det = np.asarray(det)
    pts = det[:, :8].reshape(-1, 4, 2)
    xyxy = np.concatenate(
        [pts.min(axis=1), pts.max(axis=1), det[:, -1:]],
        axis=1,
    )
    return xyxy.astype(np.float32)


def box_iou(a, b):
    """a: (N,4) xyxy, b: (4,) xyxy -> (N,)"""
    if len(a) == 0:
        return np.zeros((0,), dtype=np.float32)
    tl = np.maximum(a[:, :2], b[:2])
    br = np.minimum(a[:, 2:4], b[2:4])
    wh = np.clip(br - tl, a_min=0, a_max=None)
    inter = wh[:, 0] * wh[:, 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / np.clip(union, 1e-6, None)


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


class CCPD:
    def __init__(self, root, iou_thresh=0.5):
        self.root = root
        self.iou_thresh = iou_thresh
        self.ap = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.img_list = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            self.img_list.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
        self.img_list = sorted(
            p for p in self.img_list if parse_ccpd_bbox(p) is not None
        )
        if len(self.img_list) == 0:
            raise FileNotFoundError(
                "No CCPD images found under {}. "
                "Put CCPD .jpg files here (filename must contain the box field).".format(root)
            )

    @property
    def name(self):
        return self.__class__.__name__

    def eval(self, model):
        scores = []
        matches = []
        n_gt = 0

        pbar = tqdm.tqdm(self.img_list)
        pbar.set_description_str("Evaluating {} with {} ".format(model.name, self.name))
        for path in pbar:
            img = cv.imread(path)
            if img is None:
                continue
            gt = parse_ccpd_bbox(path)
            n_gt += 1

            h, w = img.shape[:2]
            model.setInputSize([w, h])
            det = model.infer(img)
            pred = corners_to_xyxy(det)

            if len(pred) == 0:
                continue

            order = np.argsort(-pred[:, 4])
            pred = pred[order]
            ious = box_iou(pred[:, :4], gt)
            used = False
            for score, iou in zip(pred[:, 4], ious):
                scores.append(float(score))
                hit = (not used) and (iou >= self.iou_thresh)
                matches.append(1.0 if hit else 0.0)
                if hit:
                    used = True

        if n_gt == 0 or len(scores) == 0:
            self.ap = 0.0
            self.precision = 0.0
            self.recall = 0.0
            return

        scores = np.array(scores)
        matches = np.array(matches)
        order = np.argsort(-scores)
        matches = matches[order]
        tp = np.cumsum(matches)
        fp = np.cumsum(1.0 - matches)
        rec = tp / float(n_gt)
        prec = tp / np.clip(tp + fp, 1e-6, None)
        self.ap = float(voc_ap(rec, prec))
        self.precision = float(prec[-1])
        self.recall = float(rec[-1])

    def print_result(self):
        print("==================== Results ====================")
        print("Dataset: CCPD  |  IoU threshold: {}".format(self.iou_thresh))
        print("AP:        {:.4f}".format(self.ap))
        print("Precision: {:.4f}".format(self.precision))
        print("Recall:    {:.4f}".format(self.recall))
        print("=================================================")