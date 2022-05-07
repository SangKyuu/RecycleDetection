from detectron2.layers import batched_nms
import torch

def ml_nms(boxlist, nms_thresh, max_proposals=-1,
           score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.
    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    if boxlist.has('pred_boxes'):
        boxes = boxlist.pred_boxes.tensor
        labels = boxlist.pred_classes
    else:
        boxes = boxlist.proposal_boxes.tensor
        labels = boxlist.proposal_boxes.tensor.new_zeros(
            len(boxlist.proposal_boxes.tensor))
    scores = boxlist.scores
    keep = batched_nms(boxes, scores, labels, nms_thresh)
    # keep = batched_nms(boxes, scores, torch.zeros(len(labels)), nms_thresh)
    # keep = batched_softnms(boxes, scores, torch.zeros(len(labels)), nms_thresh)

    # keep = batched_nms(boxes[keep], scores[keep], torch.zeros(len(keep)), 0.85)

    if max_proposals > 0:
        keep = keep[: max_proposals]

    boxlist = boxlist[keep]
    return boxlist


def batched_softnms(boxes, scores, idxs, iou_threshold):
    assert boxes.shape[-1] == 4

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    sigma = 0.5
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs_ = torch.argsort(scores)

    while len(idxs_) > 0:
        s = scores.clone()
        last = len(idxs_) - 1
        i = idxs_[last]
        pick.append(i)

        xx1 = torch.max(x1[i], x1[idxs_[:last]])
        yy1 = torch.max(y1[i], y1[idxs_[:last]])
        xx2 = torch.min(x2[i], x2[idxs_[:last]])
        yy2 = torch.min(y2[i], y2[idxs_[:last]])

        w = torch.max(torch.zeros(1, device=boxes.device), xx2 - xx1 + 1)
        h = torch.max(torch.zeros(1, device=boxes.device), yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / (area[idxs_[:last]] + area[i] - w * h)
        s[idxs_[:last]] *= torch.exp(-overlap ** 2 / sigma)  # decay confidences

        # delete all indexes from the index list that have
        idxs_ = idxs_[:last]
        idxs_ = idxs_[s[idxs_] > iou_threshold]

    return torch.tensor(pick, device=boxes.device)