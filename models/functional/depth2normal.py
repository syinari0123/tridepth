import torch
import torch.nn.functional as F


def compute_3dpts_batch(depths, inv_intrinsics):
    b, _, h, w = depths.size()
    # shape: [1, H, W]
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depths)
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depths)
    ones = torch.ones(1, h, w).type_as(depths)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (inv_intrinsics @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depths


def normalize_l2(vector):
    return vector * (1 / (torch.norm(vector, p=2, dim=1, keepdim=True) + 1e-7))


def depth2normal(depths, inv_intrinsics, nei=3):
    """Calculate differentiable normalmap from depthmap.
    [Reference]
    https://github.com/zhenheny/LEGO/tree/master/depth2normal

    Args:
        depths: [B,1,H,W]
        inv_intrinsics : [B,3,3]
        nei: The number of pixels refering as neighbors (Default: 3 pixel)
    Returns:
        normals: [B,3,H,W]
    """
    batch, _, height, width = depths.shape
    pts_3d_map = compute_3dpts_batch(depths, inv_intrinsics)

    # Shift the 3d pts map by nei (Default:3pixel) along 8 directions
    pts_3d_map_ctr = pts_3d_map[:, :, nei:-nei, nei:-nei]
    pts_3d_map_x0 = pts_3d_map[:, :, nei:-nei, 0:-(2 * nei)]
    pts_3d_map_y0 = pts_3d_map[:, :, 0:-(2 * nei), nei:-nei]
    pts_3d_map_x1 = pts_3d_map[:, :, nei:-nei, 2 * nei:]
    pts_3d_map_y1 = pts_3d_map[:, :, 2 * nei:, nei:-nei]
    pts_3d_map_x0y0 = pts_3d_map[:, :, 0:-(2 * nei), 0:-(2 * nei)]
    pts_3d_map_x0y1 = pts_3d_map[:, :, 2 * nei:, 0:-(2 * nei)]
    pts_3d_map_x1y0 = pts_3d_map[:, :, 0:-(2 * nei), 2 * nei:]
    pts_3d_map_x1y1 = pts_3d_map[:, :, 2 * nei:, 2 * nei:]

    # Generate difference between the central pixel and one of 8 neighboring pixels
    diff_x0 = pts_3d_map_ctr - pts_3d_map_x0
    diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
    diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
    diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
    diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
    diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
    diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
    diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

    # Flatten the diff to a #pixle by 3 matrix
    pix_num = batch * (width - 2 * nei) * (height - 2 * nei)
    diff_x0 = diff_x0.permute(0, 2, 3, 1).reshape(pix_num, 3)
    diff_y0 = diff_y0.permute(0, 2, 3, 1).reshape(pix_num, 3)
    diff_x1 = diff_x1.permute(0, 2, 3, 1).reshape(pix_num, 3)
    diff_y1 = diff_y1.permute(0, 2, 3, 1).reshape(pix_num, 3)

    diff_x0y0 = diff_x0y0.permute(0, 2, 3, 1).reshape(pix_num, 3)
    diff_x0y1 = diff_x0y1.permute(0, 2, 3, 1).reshape(pix_num, 3)
    diff_x1y0 = diff_x1y0.permute(0, 2, 3, 1).reshape(pix_num, 3)
    diff_x1y1 = diff_x1y1.permute(0, 2, 3, 1).reshape(pix_num, 3)

    # Calculate normal by cross product of two vectors
    normals0 = normalize_l2(torch.cross(diff_x1, diff_y1)).unsqueeze(0)
    normals1 = normalize_l2(torch.cross(diff_x0, diff_y0)).unsqueeze(0)
    normals2 = normalize_l2(torch.cross(diff_x0y1, diff_x0y0)).unsqueeze(0)
    normals3 = normalize_l2(torch.cross(diff_x1y0, diff_x1y1)).unsqueeze(0)

    normal_vector = torch.sum(
        torch.cat((normals0, normals1, normals2, normals3), 0), dim=0)
    normal_vector = normalize_l2(normal_vector)

    normal_map = normal_vector.reshape(batch, (height - 2 * nei), (width - 2 * nei), 3)

    normal_map = normal_map.permute(0, 3, 1, 2)
    normal_map = F.pad(normal_map, (nei, nei, nei, nei), "constant", 0)

    return normal_map
