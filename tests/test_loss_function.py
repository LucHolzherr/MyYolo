import torch
from torch.utils.data import DataLoader
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestMyYoloLoss(unittest.TestCase):

    def setUp(self):
        from src.datasets.vocdata import VOCDataSet
        from src.datasets.yolo_dataset import DataSetYolo
        from src.losses.yolo_loss import YoloLoss
        self.loss = YoloLoss()
        dataset_voc = VOCDataSet()
        self.dataset = DataSetYolo(
            dataset_voc, indices=None, input_shape=(224, 224))
        self.data_loader = DataLoader(
            self.dataset, batch_size=1, shuffle=False)
        # grond truth data
        _, self.label_tensor = next(iter(self.data_loader))
        # _, self.label_tensor = self.dataset[1]
        # create the input tensor (=output tensor of the yolo network) for which the loss is computed
        self.output_tensor = torch.zeros(
            1, 5 * self.dataset._B + self.dataset.num_classes, self.dataset._S, self.dataset._S
        )
        C = self.dataset._num_classes
        self.output_tensor[:, 0:C + 5, :, :] = self.label_tensor
        self.output_tensor[:, C + 5: C + 10, :,
                           :] = self.label_tensor[:, C: C + 5, :, :]

    def test_dummy_example(self):
        # create a label tensor with 1 bounding box
        label_tens = torch.zeros((1, 25, 7, 7))
        label_tens[0, 0, 3, 3] = 1  # class
        label_tens[0, 20, 3, 3] = 1  # confidence
        # x-coordinate relative to image cell and cell-width
        label_tens[0, 21, 3, 3] = 0.1
        label_tens[0, 22, 3, 3] = 0.1  # y-coordiante
        label_tens[0, 23, 3, 3] = 0.2  # width relative to image width
        label_tens[0, 24, 3, 3] = 0.2  # height relative to image height
        # create a object tensor which is shifted
        output_tens = torch.zeros((1, 30, 7, 7))
        output_tens[0, 0, 3, 3] = 1  # class
        output_tens[0, 20, 3, 3] = 1  # confidence
        # x-coordinate relative to image cell and cell-width
        output_tens[0, 21, 3, 3] = 0.9
        output_tens[0, 22, 3, 3] = 0.9  # y-coordiante
        output_tens[0, 23, 3, 3] = 0.2  # width relative to image width
        # height relative to image height
        output_tens[0, 24, 3, 3] = 0.2

        # compute loss -> iou > 0 -> loss is only in coordinate
        loss = self.loss(output_tens, label_tens)
        # expected loss:
        L_exp = self.loss.w_coord * (0.8**2 + 0.8**2)
        self.assertAlmostEqual(float(L_exp), float(loss), places=4)

    def test_coordinate_loss(self):
        """Test which tests whether the coordinate loss was implemented correctly
        """
        C = self.dataset._num_classes
        # shift x and y by fraction of the width, ensuring that it still overlaps with the ground truth
        fract_big = 0.5
        fract_small = 0.33
        self.shift_coordinates(self.output_tensor, fract_big, fract_small)

        # compute loss
        self.output_tensor.requires_grad_(True)
        loss = self.loss(self.output_tensor, self.label_tensor)
        loss.backward()

        # compute expected error:
        L_expected = self.compute_coordinate_loss(
            self.label_tensor, fract_small)

        self.print_end_msg(L_expected, loss)
        self.assertAlmostEqual(float(L_expected), float(loss), places=4)

        # compute expected gradient
        grad_out = self.output_tensor.grad
        grad_exp = torch.zeros(grad_out.shape)

        idxs_obj = torch.nonzero(self.label_tensor[..., C, :, :])
        widths = self.label_tensor[idxs_obj[:, 0],
                                   C+3, idxs_obj[:, 1], idxs_obj[:, 2]]
        heights = self.label_tensor[idxs_obj[:, 0],
                                    C+4, idxs_obj[:, 1], idxs_obj[:, 2]]
        grad_values_x = 2 * fract_small * self.loss.w_coord * -widths
        grad_values_y = 2 * fract_small * self.loss.w_coord * -heights
        grad_exp[idxs_obj[:, 0], C+6, idxs_obj[:, 1],
                 idxs_obj[:, 2]] = grad_values_x
        grad_exp[idxs_obj[:, 0], C+7, idxs_obj[:, 1],
                 idxs_obj[:, 2]] = grad_values_y
        torch.testing.assert_close(
            grad_out, grad_exp, rtol=1e-03, atol=1e-04)

    def shift_coordinates(self, out_tens, fract_big, fract_small):
        """Helper function to shift coordinates for the test_coordinate_loss test.
        """
        C = self.dataset._num_classes
        out_tens[..., C+1, :, :] -= out_tens[..., C+3, :, :] * fract_big
        # smaller shift in the second box, this one should be chosen in the loss function evaluation
        out_tens[..., C+6, :, :] -= out_tens[..., C+8, :, :] * fract_small
        out_tens[..., C+2, :, :] -= out_tens[..., C+4, :, :] * fract_big
        out_tens[..., C+7, :, :] -= out_tens[..., C+9, :, :] * fract_small

    def compute_coordinate_loss(self, label_tens, fract_small):
        """Helper function to shift coordinates for the test_coordinate_loss test.
        """
        C = self.dataset._num_classes
        idxs_obj = torch.nonzero(label_tens[..., C, :, :])
        if len(label_tens.shape) == 4:
            widths = label_tens[idxs_obj[:, 0],
                                C+3, idxs_obj[:, 1], idxs_obj[:, 2]]
            heights = label_tens[idxs_obj[:, 0],
                                 C+4, idxs_obj[:, 1], idxs_obj[:, 2]]
        elif len(label_tens.shape) == 3:
            widths = label_tens[C+3, idxs_obj[:, 0], idxs_obj[:, 1]]
            heights = label_tens[C+4, idxs_obj[:, 0], idxs_obj[:, 1]]
        else:
            raise ValueError(
                f"label_tens has unexpected shape: {label_tens.shape}")

        x_diffs_squared = torch.sum((widths * fract_small)**2)
        y_diffs_squared = torch.sum((heights * fract_small)**2)
        L_expected = self.loss.w_coord * (x_diffs_squared + y_diffs_squared)

        return L_expected

    def test_size_loss(self):
        """Test for size Loss
        """
        C = self.dataset._num_classes
        # scale width and height by a fraction
        fract_big = 1.5
        fract_small = 1.2
        self.output_tensor[:, C+3, :, :] *= fract_small
        # smaller shift in the second box, this one should be chosen in the loss function evaluation
        self.output_tensor[:, C+8, :, :] *= fract_big
        # same for height
        self.output_tensor[:, C+4, :, :] *= fract_small
        self.output_tensor[:, C+9, :, :] *= fract_big

        # compute loss
        self.output_tensor.requires_grad_(True)
        loss = self.loss(self.output_tensor, self.label_tensor)
        loss.backward()

        # compute expected error:
        # get cell containing an object
        idxs_obj = torch.nonzero(self.label_tensor[:, C, :, :])
        widths_label = self.label_tensor[idxs_obj[:, 0],
                                         C+3, idxs_obj[:, 1], idxs_obj[:, 2]]
        heights_label = self.label_tensor[idxs_obj[:, 0],
                                          C+4, idxs_obj[:, 1], idxs_obj[:, 2]]
        widths_out = self.output_tensor[idxs_obj[:, 0],
                                        C+3, idxs_obj[:, 1], idxs_obj[:, 2]]
        heights_out = self.output_tensor[idxs_obj[:, 0],
                                         C+4, idxs_obj[:, 1], idxs_obj[:, 2]]

        w_diffs = torch.sum(
            (torch.sqrt(widths_label) - torch.sqrt(widths_out))**2)
        h_diffs = torch.sum(
            (torch.sqrt(heights_label) - torch.sqrt(heights_out))**2)
        L_expected = self.loss.w_size * (w_diffs + h_diffs)
        self.assertAlmostEqual(float(L_expected), float(loss), places=4)

        # compute expected gradient
        grad_out = self.output_tensor.grad
        grad_exp = torch.zeros(grad_out.shape)

        idxs_obj = torch.nonzero(self.label_tensor[..., C, :, :])
        widths_sqrt = torch.sqrt(widths_out)
        heights_sqrt = torch.sqrt(heights_out)

        grad_values_x = self.loss.w_size * \
            (widths_sqrt - torch.sqrt(widths_label)) / widths_sqrt
        grad_values_y = self.loss.w_size * \
            (heights_sqrt - torch.sqrt(heights_label)) / heights_sqrt

        grad_exp[idxs_obj[:, 0], C+3, idxs_obj[:, 1],
                 idxs_obj[:, 2]] = grad_values_x
        grad_exp[idxs_obj[:, 0], C+4, idxs_obj[:, 1],
                 idxs_obj[:, 2]] = grad_values_y

        torch.testing.assert_close(
            grad_out, grad_exp, rtol=1e-03, atol=1e-04)

    def test_size_loss_02(self):
        """Test special case where the input widths are negative
        """
        C = self.dataset._num_classes
        # change sign of width
        self.output_tensor[:, C+3, :, :] *= -1.0
        self.output_tensor[:, C+8, :, :] *= -1.5

        # compute loss
        loss = self.loss(self.output_tensor, self.label_tensor)

        # get cell containing an object
        idxs_obj = torch.nonzero(self.label_tensor[:, C, :, :])

        # expected loss
        widths = torch.sqrt(
            self.label_tensor[idxs_obj[:, 0], C+3, idxs_obj[:, 1], idxs_obj[:, 2]])
        L_expected = self.loss.w_size * torch.sum((2.0 * widths)**2)
        self.print_end_msg(L_expected, loss)
        self.assertAlmostEqual(float(L_expected), float(loss), places=3)

    def test_confidence_loss(self):
        """Test for confidence loss
        """
        C = self.dataset._num_classes
        # reduce confidence in second box
        diff = 0.3
        idxs_obj = torch.nonzero(self.label_tensor[:, C, :, :])
        self.output_tensor[idxs_obj[:, 0],
                           C+5,
                           idxs_obj[:, 1],
                           idxs_obj[:, 2]] -= diff
        # shift first box, so second box should be chosen in loss evaluation
        self.output_tensor[:, C+1, :, :] -= 0.1

        # compute loss
        loss = self.loss(self.output_tensor, self.label_tensor)

        # compute expected error:
        L_expected = diff**2 * idxs_obj.shape[0]
        self.print_end_msg(L_expected, loss)
        self.assertAlmostEqual(float(L_expected), float(loss), places=4)

    def test_noobj_loss(self):
        """test the no-confidence loss: if a ground truth cell does not contain a object, the input should not predict an object
        """
        C = self.dataset._num_classes
        # get cell containing an object
        idxs_obj = torch.nonzero(self.label_tensor[:, C, :, :])
        # set all confidences to 0.1 excpet the ones where an object is remain at 1.0
        conf_wrong = 0.1
        self.output_tensor[:, C, :, :] = conf_wrong
        self.output_tensor[idxs_obj[:, 0], C,
                           idxs_obj[:, 1], idxs_obj[:, 2]] = 1.0

        # compute loss
        self.output_tensor.requires_grad_(True)
        loss = self.loss(self.output_tensor, self.label_tensor)

        loss.backward()
        # compute expected error:
        # get cell containing an object
        num_cells = self.label_tensor.shape[0] * \
            self.label_tensor.shape[2] * self.label_tensor.shape[3]
        L_expected = self.loss.w_noobj * \
            conf_wrong**2 * (num_cells - idxs_obj.shape[0])

        self.print_end_msg(L_expected, loss)
        self.assertAlmostEqual(float(L_expected), float(loss), places=4)

        # compute the expected gradient
        grad_out = self.output_tensor.grad
        grad_exp = torch.zeros(grad_out.shape)
        grad_val = 2 * self.loss.w_noobj * conf_wrong
        grad_exp[:, C, :, :] = grad_val
        grad_exp[idxs_obj[:, 0], C,
                 idxs_obj[:, 1], idxs_obj[:, 2]] = 0.0
        torch.testing.assert_close(
            grad_out, grad_exp, rtol=1e-03, atol=1e-04)

    def test_class_loss(self):
        C = self.dataset._num_classes
        # change class probabilities
        prob_wrong = 1.9
        self.output_tensor[:, 0:C, :, :] = prob_wrong

        # compute loss
        self.output_tensor.requires_grad_(True)
        loss = self.loss(self.output_tensor, self.label_tensor)
        loss.backward()

        # compute expected loss
        # label is 0 everywhere and 1 for the actual object
        idxs_obj = torch.nonzero(self.label_tensor[:, C, :, :])
        L_expected = idxs_obj.shape[0] * \
            ((C - 1) * prob_wrong**2 + (1.0 - prob_wrong)**2)

        self.print_end_msg(L_expected, loss)
        self.assertAlmostEqual(float(L_expected), float(loss), places=4)

        # compute expected gradients
        grad_out = self.output_tensor.grad
        # expect 2 * prob_wrong gradient except for where the label is 1, there we expect -1.8 as gradient
        grad_exp = torch.zeros(grad_out.shape)
        grad_exp[idxs_obj[:, 0], 0:C, idxs_obj[:, 1],
                 idxs_obj[:, 2]] = 2 * prob_wrong
        idxs_class = torch.nonzero(self.label_tensor[:, 0:C, :, :] == 1.0)
        grad_exp[idxs_class[:, 0], idxs_class[:, 1],
                 idxs_class[:, 2], idxs_class[:, 3]] = 2*(prob_wrong - 1.0)

        torch.testing.assert_close(
            grad_out, grad_exp, rtol=1e-03, atol=1e-04)

    def test_batch_size(self):
        """Tests correct behaviour if batch size is larger than 0
        """
        C = self.dataset._num_classes
        # _, label_tens02 = self.dataset[2]
        _, label_tens02 = next(iter(self.data_loader))
        # concatenate label tensors along first dimension
        label_tens = torch.cat((self.label_tensor, label_tens02), dim=0)
        out_tens = torch.zeros(
            2, 5 * self.dataset._B + self.dataset.num_classes, self.dataset._S, self.dataset._S
        )
        out_tens[:, 0:C, :, :] = label_tens[:, 0:C, :, :]
        out_tens[:, C + 5: C + 10, :, :] = label_tens[:, C: C + 5, :, :]

        # shift coordinate of first and second sample
        fract_small1 = 0.1
        fract_small2 = 0.2
        self.shift_coordinates(
            out_tens[0, :, :, :], fract_big=0.3, fract_small=fract_small1)
        self.shift_coordinates(
            out_tens[1, :, :, :], fract_big=0.6, fract_small=fract_small2)

        # compute loss
        loss = self.loss(out_tens, label_tens)

        # compute expected loss
        L_01 = self.compute_coordinate_loss(
            label_tens[0, :, :, :], fract_small1)
        L_02 = self.compute_coordinate_loss(
            label_tens[1, :, :, :], fract_small2)
        L_expected = (L_01 + L_02) / 2
        self.print_end_msg(L_expected, loss)
        self.assertAlmostEqual(float(L_expected), float(loss), places=4)

    def test_gradients(self):
        """Tests whether the gradients are 0 if the output tensor has random numbers in cell where no objects are
        """
        C = self.dataset._num_classes
        idxs_noobj = torch.nonzero(self.label_tensor[..., C, :, :] == 0)
        self.output_tensor[idxs_noobj[:, 0], 0:C, idxs_noobj[:, 1],
                           idxs_noobj[:, 2]] = torch.rand((idxs_noobj.shape[0], 20))
        self.output_tensor[idxs_noobj[:, 0], C+1:C+5, idxs_noobj[:, 1],
                           idxs_noobj[:, 2]] = torch.rand((idxs_noobj.shape[0], self.output_tensor.shape[1]-C-1-5))
        self.output_tensor[idxs_noobj[:, 0], C+6:, idxs_noobj[:, 1],
                           idxs_noobj[:, 2]] = torch.rand((idxs_noobj.shape[0], self.output_tensor.shape[1]-C-1-5))
        # we require gradients
        self.output_tensor.requires_grad_(True)
        # compute loss and its gradients
        loss = self.loss(self.output_tensor, self.label_tensor)
        loss.backward()
        # evaluate gradients
        grad_out = self.output_tensor.grad
        grad_exp = torch.zeros(grad_out.shape)
        torch.testing.assert_close(
            grad_out, grad_exp, rtol=1e-03, atol=1e-04)

    def test_asdf(self):
        C = self.dataset._num_classes
        idxs_noobj = torch.nonzero(self.label_tensor[..., C, :, :] == 0)
        self.output_tensor[idxs_noobj[:, 0], :, idxs_noobj[:, 1],
                           idxs_noobj[:, 2]] = torch.rand((idxs_noobj.shape[0], self.output_tensor.shape[1]))

        # shift x and y by fraction of the width, ensuring that it still overlaps with the ground truth
        fract_big = 0.5
        fract_small = 0.33
        self.shift_coordinates(self.output_tensor, fract_big, fract_small)
        loss = self.loss(self.output_tensor, self.label_tensor)

    @staticmethod
    def print_end_msg(L_expected, loss):
        print(f"loss: {loss}, expected loss: {L_expected}")


if __name__ == '__main__':
    unittest.main()
