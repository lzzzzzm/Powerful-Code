import torch
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
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)    # TP
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

if __name__ == '__main__':
    pred = torch.rand(1, 3, 32, 32, 32)
    ssc_target = torch.randint(low=0, high=2, size=(1, 32, 32, 32))
    mask = torch.rand(1, 32, 32, 32)
    mask = mask > 0.5

    loss = semantic_affinity_loss(pred, ssc_target, mask)
    print(loss)
