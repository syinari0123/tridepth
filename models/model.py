import torch
import torch.nn as nn
import torch.nn.functional as F

from tridepth.extractor import calculate_canny_edges
from tridepth import TriDepth
from models.networks import DRN_d_54, FaceDepthPredictor
from models.functional import calculate_depth_loss, calculate_normal_loss


class Model(nn.Module):
    def __init__(self, cam_mat, model_type="upconv", loss_type="l1", normal_weight=0.5, feat_size=512):
        super(Model, self).__init__()
        self.cam_mat = cam_mat
        self.model_type = model_type
        self.loss_type = loss_type
        self.normal_weight = normal_weight
        assert self.model_type in ["simple", "upconv"]

        # Prepare networks
        self.base_model = DRN_d_54(model_type=self.model_type, feat_size=feat_size)
        self.facedepth_predictor = FaceDepthPredictor(feat_size=feat_size)

    def forward(self, scenes, mesh_list):
        """
        """
        # Construct base 2D mesh for Triangular-patch-cloud (TriDepth)
        tridepth = TriDepth(base_imgs=scenes,
                            cam_mat=self.cam_mat,
                            mesh_list=mesh_list,
                            device=scenes.device)

        # Image feature exractor
        scene_feats = self.base_model(scenes)
        assert scene_feats.size()[-2:] == scenes.size()[-2:]

        # Extract face features from scene_feats
        face_pool_feats = tridepth.extarct_face_features(scene_feats, feat_type="face_pooling")
        face_cent_feats = tridepth.extarct_face_features(scene_feats, feat_type="face_centroid")

        # Predict face depth parameters
        face_depth_params = self.facedepth_predictor(face_pool_feats, face_cent_feats)

        # Assign predicted parameters into tridepth
        tridepth.assign_facedepths(face_depth_params)

        return tridepth

    def loss(self, tridepth, gt_depths):
        """
        """
        batch_size, _, height, width = gt_depths.shape
        device = gt_depths.device

        # Depth rendering
        est_depths = tridepth.render_depths(render_size=(height, width))

        # Calculate losses (depth & normal)
        depth_loss = calculate_depth_loss(est_depths, gt_depths, loss_type=self.loss_type)
        if self.normal_weight == 0:
            normal_loss = torch.zeros(1).to(device)
        else:
            inv_intrinsics = self.cam_mat(img_size=(height, width),
                                          inv_mat=True,
                                          t_tensor=True,
                                          batch_size=batch_size).to(device)
            normal_loss = calculate_normal_loss(est_depths, gt_depths, inv_intrinsics)

        # Total Loss
        total_loss = depth_loss + self.normal_weight * normal_loss

        # Store each loss into dictionary
        loss_dic = {}
        loss_dic["total"] = total_loss
        loss_dic["depth"] = depth_loss
        loss_dic["normal"] = normal_loss

        return total_loss, loss_dic
