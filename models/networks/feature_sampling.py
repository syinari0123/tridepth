import torch.nn.functional as F


def feature_sampling(scene_feats, verts_2d):
    """
    Args:
        scene_feats: [B,F,H,W]
        verts_2d: [B,N,2] range=(0,1)
    Returns:
        vert_feats: [B,F,N]
    """
    # Normaize coords from [0,1] to [-1,1]
    norm_verts_2d = (verts_2d * 2.0 - 1.0).unsqueeze(2)  # [B,N,1,2]

    return F.grid_sample(scene_feats, norm_verts_2d,
                         mode='bilinear',
                         padding_mode='border')[:, :, :, 0]
