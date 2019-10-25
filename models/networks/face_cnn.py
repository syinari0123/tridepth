import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max


def face_conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv1d(in_planes, out_planes, kernel_size=1),
        nn.BatchNorm1d(out_planes),
        nn.ReLU(inplace=True)
    )


class FaceDepthPredictor(nn.Module):
    def __init__(self, feat_size, w_centroid=True, depth_scale_factors=(10.0, 0.01)):
        super(FaceDepthPredictor, self).__init__()
        self.w_centroid = w_centroid
        self.alpha, self.beta = depth_scale_factors

        # Decode feature from face centroid feature (local feature)
        self.conv1 = face_conv(feat_size, 128)
        self.conv2 = face_conv(128, 64)
        self.conv3 = face_conv(64, 64)

        # Integrate pooling feature (global feature)
        if self.w_centroid:
            self.conv4 = face_conv(64 + feat_size, 128)
        else:
            self.conv4 = face_conv(64, 128, 1)
        self.conv5 = face_conv(128, 32)
        self.conv6 = face_conv(32, 8)

        # Predict parameters to determine face depth
        self.predict_depths = nn.Conv1d(8, 1, 1)
        self.predict_angles = nn.Conv1d(8, 2, 1)

    def estimate_3d_position(self, x, eps=1e-7):
        """Estimate 3D position
        """
        # 1. depth prediction
        cent_depth = torch.sigmoid(self.predict_depths(x))                               # [B,1,M]
        cent_depth = self.alpha * cent_depth + self.beta  # Rescale for each dataset

        # 2. angle prediction
        cent_angle = self.predict_angles(x)                                              # [B,2,M]

        # - angle_theta: range=[-pi/2,pi/2], polar angle in spherical coords
        cent_angle_theta = torch.tanh(cent_angle[:, :1, :]).clamp(min=-1 + eps, max=1 - eps)
        cent_angle_theta = cent_angle_theta * math.pi / 2.0                              # [B,1,M]

        # - angle_phi: [0,2pi], azimuthal angle in spherical coords
        cent_angle_phi = cent_angle[:, 1:, :]                                            # [B,1,M]

        # 3. Calculate plane equation params from centroid's depth & angle
        face_alpha = -torch.mul(torch.tan(cent_angle_theta), torch.cos(cent_angle_phi))  # [B,1,M]
        face_beta = -torch.mul(torch.tan(cent_angle_theta), torch.sin(cent_angle_phi))   # [B,1,M]
        face_depth_params = torch.cat((face_alpha, face_beta, cent_depth), 1)            # [B,3,M]
        face_depth_params = face_depth_params.transpose(1, 2)                            # [B,M,3]

        return face_depth_params

    def forward(self, pool_feats, cent_feats):
        # Convolutions (consider face's global information)
        x = self.conv1(pool_feats)
        x = self.conv2(x)
        x = self.conv3(x)

        # Concat centroid feature (consider face's local information)
        if self.w_centroid:
            x = torch.cat((x, cent_feats), 1)  # [B,576,M]
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # Estimate 3D position
        face_depth_params = self.estimate_3d_position(x)

        return face_depth_params
