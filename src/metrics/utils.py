import torch
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.metrics.mean_average_precision import BoundingBoxInfo


def visualise_bbox_img(img: torch.tensor, bbox_infos: List[BoundingBoxInfo]):
    fig, ax = plt.subplots()
    ax.imshow(img[0, 0, :, :])

    # read label tensor, get class labels and bounding boxes and plot
    for bbox_info in bbox_infos:
        x0 = bbox_info.x_mid - bbox_info.width / 2.
        y0 = bbox_info.y_mid - bbox_info.height / 2.
        width = max(0, bbox_info.width)
        height = max(0, bbox_info.height)
        rect = patches.Rectangle(
            (x0, y0), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    return fig, ax
