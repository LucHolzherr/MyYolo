import unittest
import os
import sys
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import v2

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestMAP(unittest.TestCase):

    from src.metrics.mean_average_precision import BoundingBoxInfo

    def setUp(self):
        pass

    def test_map_my(self):
        from src.metrics.mean_average_precision import mean_average_precision, BoundingBoxInfo
        bb_predictions, bb_gts = self.create_example_data01()
        mAP = mean_average_precision(
            bb_predictions, bb_gts, num_classes=20, th_iou=0.5)

        print(f"my mAP: {mAP}")

    @staticmethod
    def create_example_data01() -> Tuple[List[BoundingBoxInfo], List[BoundingBoxInfo]]:
        """Creates some dummy bounding boxes in two images with ground truths.

        Returns:
            list of predictions and list of ground truths.
        """
        from src.metrics.mean_average_precision import BoundingBoxInfo
        # image 1:
        # bb1: high iou, correct class 0, high confidence
        bb_i1_pr1 = BoundingBoxInfo(0, 0, 0.9, 0.5, 0.5, 1.0, 1.0)
        bb_i1_gt1 = BoundingBoxInfo(0, 0, 1.0, 0.5, 0.5, 0.8, 0.8)

        # bb2: high iou but wrong class
        bb_i1_pr2 = BoundingBoxInfo(0, 0, 0.4, 2.5, 2.5, 1.0, 1.0)
        bb_i1_gt2 = BoundingBoxInfo(0, 1, 1.0, 2.5, 2.5, 1.0, 1.0)

        # bb3: no grount truth at this position
        bb_i1_pr3 = BoundingBoxInfo(0, 0, 0.5, 4.0, 4.0, 1.0, 1.0)

        # bb4:
        bb_i1_pr4 = BoundingBoxInfo(0, 0, 0.45, 3.5, 3.5, 1.0, 1.0)
        bb_i1_gt4 = BoundingBoxInfo(0, 0, 1.00, 3.6, 3.6, 1.0, 1.0)

        # image 2:
        # bb1: correct bounding box of class 1
        bb_i2_pr1 = BoundingBoxInfo(1, 1, 0.8, 0.5, 0.5, 1.0, 1.0)
        bb_i2_gt1 = BoundingBoxInfo(1, 1, 1, 0.5, 0.5, 0.8, 0.8)

        # bb2: low iou, correct class 0
        bb_i2_pr2 = BoundingBoxInfo(1, 0, 0.6, 2.5, 2.5, 1.0, 1.0)
        bb_i2_gt2 = BoundingBoxInfo(1, 0, 1, 3.4, 3.4, 1.2, 1.2)

        # bb3: class 1
        bb_i2_pr3 = BoundingBoxInfo(1, 1, 0.7, 1.5, 1.5, 1.0, 1.0)
        bb_i2_gt3 = BoundingBoxInfo(1, 1, 1.0, 1.2, 1.2, 1.0, 1.0)

        bb_predictions = [bb_i1_pr1, bb_i1_pr2,
                          bb_i1_pr3, bb_i1_pr4,
                          bb_i2_pr1, bb_i2_pr2, bb_i2_pr3]
        bb_gts = [bb_i1_gt1, bb_i1_gt2, bb_i1_gt4,
                  bb_i2_gt1, bb_i2_gt2, bb_i2_gt3]

        return (bb_predictions, bb_gts)

class TestUtils(unittest.TestCase):

    def setUp(self):
        from src.datasets.vocdata import VOCDataSet
        from src.datasets.kaggledata import KaggleDataSet
        from src.datasets.yolo_dataset import DataSetYolo
        vocdata = VOCDataSet()
        kaggledata = KaggleDataSet()
        transform = v2.Compose([v2.Resize(256),
                                v2.CenterCrop(224),
                                v2.ToTensor()]
                               )
        self.dataset = DataSetYolo(
            kaggledata, None, (224, 224), transform=transform)
        self.data_loader = DataLoader(
            self.dataset, batch_size=1, shuffle=False)
        # data iterator to loop over dataloader
        self.data_iter = iter(self.data_loader)
        # grond truth data
        self.img, self.label_tensor = next(self.data_iter)
        self.img = self.img[0:1, ...]
        self.label_tensor = self.label_tensor[0:1, ...]
        # set output tensor to ground truth for the setUp method
        C = self.dataset.num_classes
        B = self.dataset.B
        S = self.dataset.S
        self.output_tensor = torch.zeros(1, 5 * B + C, S, S)
        self.output_tensor[:, 0:C + 5, :, :] = self.label_tensor
        self.output_tensor[:, C + 5: C + 10, :,
                           :] = self.label_tensor[:, C: C + 5, :, :]

    def test_get_bb_info_from_tensor_01(self):
        from src.metrics.utils import visualise_bbox_img
        # convert label tensor to BoundingBoxInfo:
        bb_infos = self.dataset.get_bb_info_from_tensor(self.label_tensor)

        fig, ax = visualise_bbox_img(self.img, bb_infos)
        plt.show()

    def test_get_bb_info_from_tensor_02(self):
        from src.metrics.utils import visualise_bbox_img
        C = self.dataset.num_classes
        B = self.dataset.B
        idxs_box = torch.nonzero(
            self.label_tensor[:, C, :, :] > 0.9, as_tuple=False)

        num_boxes = idxs_box.shape[0]
        for box_idx in range(B):
            self.output_tensor[idxs_box[:, 0], C+5*box_idx, idxs_box[:, 1],
                               idxs_box[:, 2]] -= 0.2 * torch.rand(num_boxes)

        bb_infos = self.dataset.get_bb_info_from_tensor(self.output_tensor)
        fig, ax = visualise_bbox_img(self.img, bb_infos)
        plt.show()

    def test_get_bb_info_from_tensor_03(self):
        from src.metrics.utils import visualise_bbox_img
        # get second sample
        img2, label_tensor2 = next(self.data_iter)
        img2 = img2[0:1, ...]
        label_tensor2 = label_tensor2[0:1, ...]
        C = self.dataset.num_classes
        B = self.dataset.B
        S = self.dataset.S
        output_tensor2 = torch.zeros(1, 5 * B + C, S, S)
        output_tensor2[:, 0:C + 5, :, :] = label_tensor2
        output_tensor2[:, C + 5: C + 10, :,
                       :] = label_tensor2[:, C: C + 5, :, :]

        output_conc = torch.concat((self.output_tensor, output_tensor2), dim=0)
        imgs_conc = torch.concat((self.img, img2), dim=0)
        for box_idx in range(B):
            output_conc[:, C+5*box_idx, :, :] -= 0.2 * \
                torch.rand((output_conc.shape[0], S, S))

        bb_infos = self.dataset.get_bb_info_from_tensor(output_conc)
        bb_infos_per_img = [
            [bb_info for bb_info in bb_infos if bb_info.img_idx == idx] for idx in range(2)]
        for img_idx, bbs in enumerate(bb_infos_per_img):
            fig, ax = visualise_bbox_img(
                imgs_conc[img_idx:img_idx+1, ...], bbs)
        plt.show()


if __name__ == '__main__':
    unittest.main()
