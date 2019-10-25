import cv2
import torch
import numpy as np


def inpaint_depth(depth, silhouette, psize=1):
    """Inpaint background area
    [Reference] 
    https://docs.opencv.org/3.3.1/df/d3d/tutorial_py_inpainting.html

    Args:
        depth      : torch.Tensor(H,W)
        silhouette : torch.Tensor(H,W)
    Returns:
        inpainted_depth : torch.Tensor(1,H,W)
    """
    # Convert into numpy array (depth/silhouette)
    np_depth = depth.data.cpu().numpy()
    np_silhouette = (silhouette < 1).data.cpu().numpy()  # mask target is 1, otherwise 0

    # Expand
    np_depth_padded = np.pad(np_depth, (psize, psize), 'edge')  # [H+1,W+1]
    np_silhouette_padded = np.pad(np_silhouette, (psize, psize), "edge")  # [H+1,W+1]

    # Normalize depth
    d_min = np_depth_padded.min()
    d_max = np_depth_padded.max()
    # Quantize depth into 8-bit(255) [0,255]
    np_depth_padded_norm = (np_depth_padded - d_min) * 255.0 / (d_max - d_min)

    # Inpaint
    np_filled_depth = cv2.inpaint(np_depth_padded_norm, np_silhouette_padded, 3,
                                  cv2.INPAINT_TELEA)[psize:-psize, psize:-psize]
    np_filled_depth = np_filled_depth * (d_max - d_min) / 255.0 + d_min  # original scale

    # Insert inpainted value in the pixel of original depth
    np_filled_depth = np_filled_depth * np_silhouette + np_depth * (1.0 - np_silhouette)

    return torch.from_numpy(np_filled_depth[None, :, :]).float()
