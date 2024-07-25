import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from collections import Counter
from src.metrics.intersection_over_union import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, num_classes: int = 20, num_cells: int = 7, num_boxes_per_cell: int = 2, class_counts: Counter = None):
        super(YoloLoss, self).__init__()
        self._num_classes = num_classes
        self._num_cells = num_cells
        self._num_boxes_per_cell = num_boxes_per_cell
        # weights of summed loss
        self._w_coord = 5   # from paper
        self._w_size = 5  # from paper
        self._w_noobj = 0.5   # from paper
        # sum of squared errors loss method
        self.SSE = nn.MSELoss(reduction='sum')
        if class_counts is not None:
            self.class_weights = []
            total_weight = 0.0
            for key, item in class_counts.items():
                weight = 1./item
                self.class_weights.append(weight)
                total_weight += weight
            # normalize to 1
            for idx in range(len(self.class_weights)):
                self.class_weights[idx] /= total_weight
        else:
            self.class_weights = [1] * num_classes
        self.class_weights = torch.tensor(self.class_weights)

    def forward(self, output, target, writer: SummaryWriter = None, epoch: int = 0) -> float:
        """evaluation of loss function

        Args:
            output: yolo output, tensor of shape (N, M + B * 5)
            target: label tensor, shape (N, M + 5)
        Returns:
            float: Loss value
        """
        device = output.device
        C = self._num_classes

        # for each loss, it's a squared sum of the loss of each cell.
        # -> reshape the tensors to shape (N*S*S, M+B*5).
        output_resh = torch.permute(output, (0, 2, 3, 1))
        # flatten the first 3 dimensions to N*S*S, keep last dimension M+B*5
        output_resh = torch.flatten(output_resh, end_dim=-2)
        num_samples = output_resh.shape[0]

        # same for target
        target_resh = torch.permute(target, (0, 2, 3, 1))
        target_resh = torch.flatten(target_resh, end_dim=-2)

        # vector indicating whether object is present, this corresponds to the indicator function 1^obj_i
        obj_present = torch.zeros((num_samples, 1), device=device)
        obj_present[:, 0] = target_resh[:, C]

        # indices of present vector
        idxs_present = torch.nonzero(obj_present == 1, as_tuple=False)
        idxs_present = idxs_present[:, 0]
        idxs_not_present = self.get_other_indices(
            idxs_present, num_samples, device)

        # get the corresponding output boxes for the target box, this corresponds to the indicator function 1^obj_ijs
        # TODO: vectorization of for loop
        # avoid using lists as we want to have all data on the gpu
        ious = torch.zeros(self._num_boxes_per_cell,
                           num_samples, device=device)
        for idx_j in range(self._num_boxes_per_cell):
            output_bbox = output_resh[idxs_present,
                                      C+1+5 * idx_j:C + 5*(idx_j+1)]
            target_bbox = target_resh[idxs_present, C+1: C + 6]
            # x and y are relative to cell_width, but the width and height are relative to the image width -> transform into same system
            output_bbox[:, 2:4] *= self._num_cells
            target_bbox[:, 2:4] *= self._num_cells
            ious[idx_j, idxs_present] = intersection_over_union(
                output_bbox,
                target_bbox
            )

        # get indices of maximum Jaccard Index for each cell
        max_idxs = torch.argmax(ious, dim=0)

        # create bounding box coordinates to compare with target for the cells where a object is present
        # 0: confidence, 1: x, 2: y, 3: sqrt(width), 5: sqrt(height)
        bboxes_in = torch.zeros((idxs_present.shape[0], 5), device=device)
        bboxes_targ = torch.zeros((idxs_present.shape[0], 5), device=device)
        # TODO: vectorize the for loop
        for idx, idx_p in enumerate(idxs_present):
            bbox_og = output_resh[
                idx_p, C + 5*max_idxs[idx_p]:
                    C + 5*max_idxs[idx_p]+5
            ]
            bboxes_in[idx, 0:3] = bbox_og[0:3]
            # the loss needs the square root of width and height
            # prediction could be negative -> take abs in sqrt, add epsilon for avoiding sqrt(0).
            # punish negative widths/heights with the sign function
            bboxes_in[idx, 3:5] = torch.sign(
                bbox_og[3:5]) * torch.sqrt(torch.abs(bbox_og[3:5])+0.000001)

            bbox_targ_og = target_resh[idx_p, C: C+5]
            bboxes_targ[idx, 0:3] = bbox_targ_og[0:3]
            bboxes_targ[idx, 3:5] = torch.sqrt(
                bbox_targ_og[3:5] + 0.000001)

        # loss consist of multiple parts which are summed
        # first term is the coordinate Loss
        x_in = bboxes_in[:, 1]
        x_target = bboxes_targ[:, 1]
        y_in = bboxes_in[:, 2]
        y_target = bboxes_targ[:, 2]
        L_coord = self._w_coord * \
            (self.SSE(x_in, x_target) + self.SSE(y_in, y_target))

        # second term is the width and height loss
        width_in = bboxes_in[:, 3]
        height_in = bboxes_in[:, 4]
        width_target = bboxes_targ[:, 3]
        height_target = bboxes_targ[:, 4]
        L_size = self._w_size * \
            (self.SSE(width_in, width_target) + self.SSE(height_in, height_target))

        # third term is the confidence score
        L_conf = self.SSE(bboxes_in[:, 0], bboxes_targ[:, 0])

        # fourth term is the no confidence score:
        # TODO: make generic for the concept of B boxes
        conf_noobj_in_01 = output_resh[idxs_not_present, C]
        conf_noobj_in_02 = output_resh[idxs_not_present, C+5]
        conf_noobj_target = target_resh[idxs_not_present, C]
        L_noconf = self._w_noobj * \
            (self.SSE(conf_noobj_in_01, conf_noobj_target) +
             self.SSE(conf_noobj_in_02, conf_noobj_target))

        # fifth term is the class probability score, ground truth class propabilites are 0 if no object preesnt, 1 otherwise
        class_in = output_resh[idxs_present, 0:C]
        class_target = target_resh[idxs_present, 0:C]
        # torch.nonzero(class_target,
        cEL = nn.CrossEntropyLoss(weight=self.class_weights, reduction='sum')
        L_class = cEL(class_in, class_target)
        # L_class = self.SSE(class_in, class_target)

        if writer is not None:
            tag_scalar_dict = {
                'L_coord': L_coord,
                'L_size': L_size,
                'L_conf': L_conf,
                'L_noconf': L_noconf,
                'L_class': L_class
            }
            writer.add_scalars(
                main_tag='loss', tag_scalar_dict=tag_scalar_dict, global_step=epoch)

        # print(
        #     f"L_coord: {L_coord}, L_size: {L_size}, L_conf: {L_conf}, L_noconf: {L_noconf}, L_class: {L_class}\n")

        return (L_coord + L_size + L_conf + L_noconf + L_class) / output.shape[0]

    @property
    def w_coord(self):
        return self._w_coord

    @property
    def w_noobj(self):
        return self._w_noobj

    @property
    def w_size(self):
        return self._w_size

    @staticmethod
    def get_other_indices(idxs, N: int, device):
        all_indices = torch.arange(N, device=device)
        mask = torch.ones(N, dtype=bool, device=device)
        mask[idxs] = False
        return all_indices[mask]
