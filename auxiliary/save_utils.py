import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class SaveImages:
    def __init__(self, cmap_type=None):
        self.merged_img = None
        self.cmap_type = cmap_type

    def _colored_depthmap(self, np_depth):
        # Scaling from d_min & d_max
        d_min = np.min(np_depth)
        d_max = np.max(np_depth)
        depth_relative = (np_depth - d_min) / (d_max - d_min)

        # Output colored depthmap
        if self.cmap_type is None:
            return 255 * cv2.cvtColor(depth_relative, cv2.COLOR_GRAY2RGB)
        elif self.cmap_type == "jet":
            return 255 * plt.cm.jet(depth_relative)[:, :, :3]
        elif self.cmap_type == "viridis":
            return 255 * plt.cm.viridis(depth_relative)[:, :, :3]
        else:
            raise NotImplementedError

    def update_merged_image(self, rgb, gt_depth, pred_depth, pred_edge):
        # Check tensor shape
        assert rgb.dim() == 3, "input shape should be [3,H,W]"
        assert gt_depth.dim() == 3, "input shape should be [1,H,W]"
        assert pred_depth.dim() == 3, "input shape should be [1,H,W]"
        assert pred_edge.dim() == 3, "input shape should be [1,H,W]"

        # To numpy
        np_rgb = 255 * np.transpose(np.squeeze(rgb.cpu().numpy()), (1, 2, 0))
        np_gt_depth = np.squeeze(gt_depth.cpu().numpy())
        np_pred_depth = np.squeeze(pred_depth.detach().cpu().numpy())
        np_pred_edge = 255 * cv2.cvtColor(np.squeeze(pred_edge.detach().cpu().numpy()), cv2.COLOR_GRAY2RGB)

        # Convert depthmap -> colormap
        color_gt_depth = self._colored_depthmap(np_gt_depth)
        color_pred_depth = self._colored_depthmap(np_pred_depth)

        # Update merged image
        result_pair = np.hstack([np_rgb, color_gt_depth, color_pred_depth, np_pred_edge])
        if self.merged_img is None:
            self.merged_img = result_pair
        else:
            self.merged_img = np.vstack([self.merged_img, result_pair])

    def output_tensor_image(self):
        return torch.from_numpy(self.merged_img)

    def save_merged_image(self, filename):
        # Save into file
        pil_merged_img = Image.fromarray(self.merged_img.astype('uint8'))
        pil_merged_img.save(filename)

    def reset(self):
        self.merged_img = None
