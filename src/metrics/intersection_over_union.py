import torch


def intersection_over_union(A: torch.tensor, B: torch.tensor) -> torch.tensor:
    """Special iou (Jaccard Index) method for the data-structures used in this loss function. 
    The iou of A[i, :] with B[i, :] is computed for every i.
        Args:
            A: A[i, 0]: x_midpoint, A[i, 1]: y_midpoint, A[i, 2]: width, A[i, 3]: height of sample i
            B: B[i, 0]: x_midpoint, B[i, 1]: y_midpoint, B[i, 2]: width, B[i, 3]: height of sample i

        Returns:
            1D tensor: iou for each sample
        """
    # step 1.: compute coordinates of intersection rectangle
    int_x0 = torch.max(A[:, 0] - A[:, 2]/2., B[:, 0] - B[:, 2]/2.)
    int_x1 = torch.min(A[:, 0] + A[:, 2]/2., B[:, 0] + B[:, 2]/2.)
    int_y0 = torch.max(A[:, 1] - A[:, 3]/2., B[:, 1] - B[:, 3]/2.)
    int_y1 = torch.min(A[:, 1] + A[:, 3]/2., B[:, 1] + B[:, 3]/2.)
    # step 2.: compute intersection area
    # if not overlapping, the difference would be negative -> clamp to 0
    int_width = torch.clamp(int_x1 - int_x0, min=0)
    int_height = torch.clamp(int_y1 - int_y0, min=0)
    int_area = int_width * int_height
    # step 3.: compute union area
    A_area = A[:, 2] * A[:, 3]
    B_area = B[:, 2] * B[:, 3]
    # see formula: https://en.wikipedia.org/wiki/Jaccard_index
    union_area = A_area + B_area - int_area

    # step 4.: return intersection over union, avoid division by 0
    return int_area / (union_area+0.000001)
