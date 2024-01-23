import torch
import torch.nn as nn
import torch.nn.functional as F


def semantic_affinity_loss(pred, ssc_target, mask):
    ###
    # This function is used to calculate the semantic segmentation loss
    # for the scene completion task.
    # The loss is about three part: precision, recall and specificity.
    # precision: the ratio of the number of correctly predicted positive
    #            examples to the total number of predicted positive examples.
    # recall: the ratio of the number of correctly predicted positive
    #         examples to the total number of positive examples.
    # specificity: the ratio of the number of correctly predicted negative
    #              examples to the total number of negative examples.
    ###

    # Get softmax probabilities
    pred = F.softmax(pred, dim=-1)
    loss = 0
    count = 0
    n_classes = pred.shape[-1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[..., i]

        # mask other voxels
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)  # TP
            loss_class = 0

            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision

            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))

                loss_class += loss_recall

            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class

    return loss / count


def geometry_affinity_loss(pred, ssc_target, empty_class_index, mask):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=-1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[..., empty_class_index]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    nonempty_target = ssc_target != empty_class_index
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()

    geo_loss = F.binary_cross_entropy(precision, torch.ones_like(precision)) + F.binary_cross_entropy(recall, torch.ones_like(recall))
    if spec > 0:
        geo_loss += F.binary_cross_entropy(spec, torch.ones_like(spec))
    return geo_loss


class AffinityLoss(nn.Module):
    def __init__(self,
                 empty_class_index=17,
                 semantic_loss_weight=1.0,
                 geometry_loss_weight=1.0,
                 loss_weight=1.0,
                 loss_name='loss_affinity'):
        super(AffinityLoss, self).__init__()

        self.empty_class_index = empty_class_index
        self.semantic_loss_weight = semantic_loss_weight
        self.geometry_loss_weight = geometry_loss_weight
        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def forward(self,
                pred_occ,
                label_occ,
                mask=None,
                **kwargs):
        """Forward function."""
        if mask == None:
            mask = torch.ones_like(label_occ, dtype=torch.bool)

        semantic_loss = self.semantic_loss_weight * semantic_affinity_loss(pred_occ, label_occ, mask)
        geometry_loss = self.geometry_loss_weight * geometry_affinity_loss(pred_occ, label_occ, self.empty_class_index,
                                                                       mask)
        loss = semantic_loss + geometry_loss
        return self.loss_weight * loss


if __name__ == '__main__':
    loss_fn = AffinityLoss()


    pred = torch.rand(2, 100, 100, 16, 18)
    ssc_target = torch.randint(low=0, high=17, size=(2, 100, 100, 16))
    # mask = torch.rand(1, 32, 32, 32)
    # mask = mask > 0.5
    mask = None
    loss = loss_fn(pred, ssc_target, mask)
    print(loss)
