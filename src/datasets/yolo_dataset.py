import torch.utils.data.dataset
import torch.nn.functional as F
from torchvision import tv_tensors
from torchvision.transforms import v2

# for creating images
from torchvision.transforms import ToPILImage, ToTensor
from PIL import ImageDraw


from typing import Sequence
# for plotting data
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.metrics.mean_average_precision import BoundingBoxInfo


class DataSetYolo(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            indices: Sequence[int],
            input_shape,
            transform=v2.Identity(),
            inv_norm_transf=v2.Identity(),  # TODO EXPLAIN
            num_cells: int = 7,
            num_boxes_per_cell: int = 2,
    ):
        """Initializes the dataset
        """
        super(DataSetYolo, self).__init__()
        self.dataset = dataset
        if indices is None:
            self.indices = list(torch.arange(0, len(dataset)+0.01, 1).int())
        else:
            self.indices = indices
        self.input_shape = input_shape
        self._S = num_cells
        self._B = num_boxes_per_cell
        # map labels to class indices
        self._num_classes = self.dataset.num_classes
        self.transform = transform
        self.inv_norm_transf = inv_norm_transf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # load image and label dictionary
        img_og, label = self.dataset[self.indices[idx]]
        # transform image and boundingboxes into correct size
        img, bboxes = self._get_img_bbox(
            img_og, label, self.transform)
        # get label tensor
        label_tensor = self._create_label_tensor(label, bboxes)

        return (img, label_tensor)

    def _get_img_bbox(self, img, label, transformation: v2.Transform) -> torch.tensor:
        # create bounding box object from label info
        keys = ['xmin', 'ymin', 'xmax', 'ymax']
        label_obj = label['object']
        boxes = tv_tensors.BoundingBoxes(
            [[float(label_obj[idx]['bndbox'][key]) for key in keys]
                for idx in range(len(label_obj))
             ],
            format="XYXY", canvas_size=img.shape[-2:]
        )

        out_img, out_boxes = transformation(img, boxes)
        return out_img, out_boxes

    def _create_label_tensor(self, label_dict, bboxes):
        label_tensor = torch.zeros(5 + self._num_classes, self._S, self._S)
        obj_dict = label_dict['object']
        class_names = [obj_dict[idx]['class']
                       for idx in range(len(obj_dict))]
        cell_length = self.input_shape[0] / self._S
        # loop over objects-bboxes, fill those cells in the
        centers_idx = [((box[0] + box[2]) / 2 / cell_length,
                        (box[1] + box[3]) / 2 / cell_length) for box in bboxes]

        # to handle case where both image centers are in the same cell, keep track of image center cell
        center_cells = []
        for obj_idx, c_idx in enumerate(centers_idx):
            if class_names[obj_idx] not in self.dataset.inv_labels_map:
                continue
            # get cell idxs
            idx_x, idx_y = int(c_idx[0]), int(c_idx[1])
            idx_x = min(idx_x, self._S-1)
            idx_y = min(idx_y, self._S-1)

            bbox = bboxes[obj_idx]
            # if bounding box has area 0, skip
            area_this = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area_this == 0:
                continue
            # special case where two image centers are in the same cell: keep object with larger bounding box
            if (idx_x, idx_y) in center_cells:
                # print("special case!")
                idx_other = center_cells.index((idx_x, idx_y))
                bbox_other = bboxes[idx_other]
                area_other = (bbox_other[2] -
                              bbox_other[0]) * (bbox_other[3] - bbox_other[1])
                if area_this > area_other:
                    label_tensor[:, idx_y, idx_x] = 0
                else:
                    continue
            # else:
            #     print("normal case")
            center_cells.append((idx_x, idx_y))
            # fill class info
            class_idx = self.dataset.inv_labels_map[class_names[obj_idx]]
            label_tensor[class_idx, idx_y, idx_x] = 1
            # fill confidence
            label_tensor[self._num_classes, idx_y, idx_x] = 1
            # fill x, y, widht, height
            # x and y are ratios relative to the cell (0 is left/top edge, 1 is right/bottom edge)
            label_tensor[self._num_classes +
                         1, idx_y, idx_x] = c_idx[0] - idx_x
            label_tensor[self._num_classes +
                         2, idx_y, idx_x] = c_idx[1] - idx_y

            # width is the ratio relative to the entire image, between 0 and 1
            width = (bbox[2] - bbox[0]) / self.input_shape[0]
            height = (bbox[3] - bbox[1]) / self.input_shape[0]
            label_tensor[self._num_classes + 3,
                         idx_y, idx_x] = width
            label_tensor[self._num_classes + 4,
                         idx_y, idx_x] = height

        return label_tensor

    def create_bbox_image_from_label_tensor(
            self, img: torch.tensor, label_tensor: torch.tensor, conf_threshold: float = 0.5) -> torch.tensor:
        """Returns an image with the bounding box and class label drawn.

        Args:
            img: input image
            label_tensor: either label_tensor or output tensor
            conf_threshold: confidence threshold used to display the bounding box
        """
        # de-normalize image
        img_copy = self.inv_norm_transf(img)
        # create pil image
        to_pil = ToPILImage()
        img_pil = to_pil(img_copy)
        draw = ImageDraw.Draw(img_pil)

        # load a custom font with a larger size
        #  font = ImageFont.truetype("arial.ttf", size=24)

        cell_length = self.input_shape[0] / self._S
        # TODO: vectorize the for loop
        for idx_y in range(self._S):
            for idx_x in range(self._S):
                cell_label = label_tensor[:, idx_y, idx_x]
                conf_scores = cell_label[self.num_classes::5]
                # check if has higher confidence than threshold
                max_idx = torch.argmax(conf_scores)
                if conf_scores[max_idx] < conf_threshold:
                    continue

                # get class name using softmax
                softmax = F.softmax(cell_label[0:self.num_classes], dim=0)
                class_idx = torch.argmax(softmax)
                class_name = self.dataset.labels_map[int(class_idx)]

                # get corresponding bounding box
                bbox_og = cell_label[self.num_classes + max_idx * 5:
                                     self.num_classes + (1+max_idx) * 5]
                bbox_center_x = (idx_x + bbox_og[1]) * cell_length
                bbox_center_y = (idx_y + bbox_og[2]) * cell_length
                bbox_width = bbox_og[3] * self.input_shape[1]
                bbox_height = bbox_og[4] * self.input_shape[0]
                bbox_x = bbox_center_x - bbox_width/2.
                bbox_y = bbox_center_y - bbox_height/2.

                # sanity check the bbox values
                if bbox_x < 0 or bbox_x > self.input_shape[0]:
                    # print(f'WARNING: x coordinate out of bounds: {bbox_x}')
                    bbox_x = torch.clamp(
                        bbox_x, min=0.0, max=self.input_shape[0])
                if bbox_y < 0 or bbox_y > self.input_shape[1]:
                    # print(f'WARNING: y coordinate out of bounds: {bbox_y}')
                    bbox_y = torch.clamp(
                        bbox_y, min=0.0, max=self.input_shape[1])
                if bbox_width < 0 or bbox_width > self.input_shape[0]:
                    # print(f'WARNING: width out of bounds: {bbox_width}')
                    bbox_width = torch.clamp(
                        bbox_width, min=0.0, max=self.input_shape[0])
                if bbox_height < 0 or bbox_height > self.input_shape[1]:
                    # print(f'WARNING: height out of bounds: {bbox_height}')
                    bbox_height = torch.clamp(
                        bbox_height, min=0.0, max=self.input_shape[1])

                draw.rectangle([bbox_x, bbox_y, bbox_x+bbox_width, bbox_y+bbox_height],
                               outline='red', width=2)
                # , font=font)
                draw.text((bbox_x, bbox_y),
                          f"{class_name}, {conf_scores[max_idx]:.2f}")

        to_tensor = ToTensor()
        img_bboxes = to_tensor(img_pil)
        img_bboxes = torch.permute(img_bboxes, (1, 2, 0))
        return img_bboxes

    def display_data_sample(self, data_idx):
        img, label_tensor = self.__getitem__(data_idx)

        fig, ax = plt.subplots()
        ax.imshow(img[0, :, :])

        # read label tensor, get class labels and bounding boxes and plot
        cell_length = self.input_shape[0] / self._S
        for idx_y in range(self._S):
            for idx_x in range(self._S):
                cell_label = label_tensor[:, idx_y, idx_x]
                non_zero_mask = cell_label[0:self._num_classes] != 0

                if 1 not in non_zero_mask:
                    continue

                non_zero_indices = torch.nonzero(non_zero_mask).squeeze()
                class_name = self.dataset.labels_map[int(non_zero_indices)]
                bbox_center_x = (cell_label[self._num_classes +
                                            1] + idx_x) * cell_length
                bbox_center_y = (cell_label[self._num_classes +
                                            2] + idx_y) * cell_length
                bbox_width = cell_label[self._num_classes +
                                        3] * self.input_shape[1]
                bbox_height = cell_label[self._num_classes +
                                         4] * self.input_shape[0]

                bbox_x = bbox_center_x - bbox_width/2
                bbox_y = bbox_center_y - bbox_height/2

                rect = patches.Rectangle(
                    (bbox_x, bbox_y), bbox_width, bbox_height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(bbox_x, bbox_y, s=class_name,
                        color='black', fontsize=12)

        plt.show()

    def get_bb_info_from_tensor(self, in_tens: torch.tensor, conf_th: float = 0.4, img_width: int = 224):
        """For every sample and cell in in_tens, the bounding box info is returned
        """
        cell_width = int(self.input_shape[1] / self.S)
        # reshape input to NUM_SAMPLES * S * S, channels
        in_resh = torch.permute(in_tens, (0, 2, 3, 1))
        in_resh = torch.flatten(in_resh, end_dim=-2)
        # get cells with box in it (high enough confidence)

        # case label tensor
        if in_resh.shape[1] == self.num_classes + 5:
            idxs_box = torch.nonzero(
                in_resh[:, self.num_classes] > conf_th, as_tuple=False).flatten().to(in_tens.device)

            _, class_pred = torch.max(
                in_resh[idxs_box, 0:self.num_classes], dim=1)
            conf_scores = in_resh[idxs_box, self.num_classes]

            img_idxs = (idxs_box / (self.S * self.S)).int()
            cell_idxs_flat = idxs_box % (self.S * self.S)
            cell_idxs_x = cell_idxs_flat % self.S
            cell_idxs_y = (cell_idxs_flat / self.S).int()
            x_mids = (in_resh[idxs_box, self.num_classes+1] +
                      cell_idxs_x) * cell_width
            y_mids = (in_resh[idxs_box, self.num_classes+2] +
                      cell_idxs_y) * cell_width
            widths = in_resh[idxs_box, self.num_classes+3] * img_width
            heights = in_resh[idxs_box, self.num_classes+4] * img_width

            bb_infos = [BoundingBoxInfo(int(img_idxs[idx]), int(class_pred[idx]), float(conf_scores[idx]), float(x_mids[idx]),
                                        float(y_mids[idx]), float(widths[idx]), float(heights[idx])) for idx in range(len(idxs_box))]

            return bb_infos

        # case yolo output tensor:
        else:
            # multiple boxes per cell -> take the one with highest confidence
            idxs_box = torch.tensor([]).int().to(in_tens.device)
            for box_idx in range(self.B):
                idxs_box_i = torch.nonzero(
                    in_resh[:, self.num_classes + 5 * box_idx] > conf_th, as_tuple=False).flatten()
                # append idxs_box_i to idxs_box
                idxs_box = torch.cat((idxs_box, idxs_box_i))

            # unique sorted idxs
            idxs_box = torch.unique(idxs_box)
            idxs_box, _ = torch.sort(idxs_box)

            # get second index which specifys which box has the higher confidence
            _, box_idxs_per_cell = torch.max(
                in_resh[idxs_box, self.num_classes::5], dim=1)

            _, class_pred = torch.max(
                in_resh[idxs_box, 0:self.num_classes], dim=1)
            conf_scores = in_resh[idxs_box,
                                  self.num_classes+5*box_idxs_per_cell]

            img_idxs = (idxs_box / (self.S * self.S)).int()
            cell_idxs_flat = idxs_box % (self.S * self.S)
            cell_idxs_x = cell_idxs_flat % self.S
            cell_idxs_y = (cell_idxs_flat / self.S).int()

            x_vals = in_resh[idxs_box,
                             self.num_classes+5*box_idxs_per_cell + 1]
            y_vals = in_resh[idxs_box,
                             self.num_classes+5*box_idxs_per_cell + 2]
            width_vals = in_resh[idxs_box, self.num_classes +
                                 5*box_idxs_per_cell+3]
            height_vals = in_resh[idxs_box, self.num_classes +
                                  5*box_idxs_per_cell+4]

            x_mids = (x_vals + cell_idxs_x) * cell_width
            y_mids = (y_vals + cell_idxs_y) * cell_width
            widths = width_vals * img_width
            heights = height_vals * img_width

            bb_infos = [BoundingBoxInfo(int(img_idxs[idx]), int(class_pred[idx]), float(conf_scores[idx]), float(x_mids[idx]),
                                        float(y_mids[idx]), float(widths[idx]), float(heights[idx])) for idx in range(len(idxs_box))]

            return bb_infos

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def S(self):
        return self._S

    @property
    def B(self):
        return self._B
