import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import depth2normal


def calculate_depth_loss(est_depths, gt_depths, loss_type="l1"):
    """Calculate loss between estimated depthmap and GT depthmap

    Args:
        est_depths: [B,1,H,W]
        gt_depths: [B,1,H,W]
        loss_type: Choose loss type from ['l1','l2']
    """
    assert est_depths.dim() == gt_depths.dim(), "inconsistent dimensions"
    assert loss_type in ["l1", "l2"], "loss_type should be l1/l2"

    valid_mask = (gt_depths > 0).detach()
    diff = est_depths - gt_depths

    if loss_type == "l1":
        return diff[valid_mask].abs().mean()
    elif loss_type == "l2":
        return (diff[valid_mask] ** 2).mean()


def calculate_normal_loss(est_depths, gt_depths, inv_intrinsics):
    """Calculate loss between estimated normalmap and GT normalmap 
    [Reference]
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf

    Args:
        est_depths: [B,1,H,W]
        gt_depths: [B,1,H,W]
        inv_intrinsics: This is used for normalmap calculation
    """
    assert est_depths.dim() == gt_depths.dim(), "inconsistent dimensions"

    valid_mask = (gt_depths > 0).detach().permute(0, 2, 3, 1)

    # Calculate normal map [B,H,W,3]
    gt_normals = depth2normal(gt_depths, inv_intrinsics).permute(0, 2, 3, 1)
    est_normals = depth2normal(est_depths, inv_intrinsics).permute(0, 2, 3, 1)
    # dot product
    flat_n_target = gt_normals.contiguous().view(-1, 1, 3)  # [B*H*W,1,3]
    flat_n_pred = est_normals.contiguous().view(-1, 3, 1)  # [B*H*W,3,1]
    flat_v_mask = valid_mask.contiguous().view(-1, 1)  # [B*H*W,1]

    dot_product = torch.bmm(flat_n_target, flat_n_pred)[flat_v_mask]
    loss = -dot_product.mean()  # minus!!!

    return loss
