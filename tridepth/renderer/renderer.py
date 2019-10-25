import torch
import torch.nn as nn

import tridepth_renderer.cuda.rasterize as rasterize_cuda
from . import vertices_to_faces, rasterize_image


def flip(x, dim):
    """
    Flip tensor in specified dimension.
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long).cuda()
    return x[tuple(indices)]


class Renderer(nn.Module):
    def __init__(self, render_size=(228, 304)):
        super(Renderer, self).__init__()
        # Rendering size (output size)
        self.render_size = render_size

        # Other parameters
        self.anti_aliasing = True
        self.background_color = [0, 0, 0]
        self.fill_back = False
        self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]])

        # light
        self.light_intensity_ambient = 0.5
        self.light_intensity_directional = 0.5
        self.light_color_ambient = [1, 1, 1]
        self.light_color_directional = [1, 1, 1]
        self.light_direction = [0, 1, 0]

        # rasterization
        self.rasterizer_eps = 1e-3

    def project_cam_to_depthmap(self, verts_3d, intrinsics, orig_size, eps=1e-7):
        '''Convert verts from 3D-pointcloud-format to 2.5D-depthmap-format (projection) range=[0,1]
        Args:
            verts: [B,N,3] (3D pointcloud format)
            intrinsics: [B,3,3]
            orig_size: (height, width)
        '''
        x, y, z = verts_3d[:, :, 0], verts_3d[:, :, 1], verts_3d[:, :, 2]
        x_ = x / (z + eps)
        y_ = y / (z + eps)

        verts_depth = torch.stack([x_, y_, torch.ones_like(z)], dim=-1)
        verts_depth = torch.matmul(verts_depth, intrinsics.transpose(1, 2))
        u, v = verts_depth[:, :, 0], verts_depth[:, :, 1]

        # map u,v from [0, img_size] to [0, 1] to use by the renderer
        u = (u / orig_size[1]).clamp(min=0, max=1)
        v = (v / orig_size[0]).clamp(min=0, max=1)
        vertices = torch.stack([u, v, z], dim=-1)
        return vertices

    def convert_depthmap_to_cam(self, verts_3d, intrinsics, orig_size):
        """Converts from depthmap-format to 3D-pointcloud-format (camera coords)
        Args:
            verts_3d: [B,N,3] (3D pointcloud format)
            intrinsics: [B,3,3]
            orig_size: (height, width)
        """
        # Change scale [0,1] -> [0, img_size]
        verts_depth = verts_3d[:, :, 2:].transpose(2, 1)  # [B,1,N]
        verts_3d_cam = torch.cat((verts_3d[:, :, :1] * orig_size[1],
                                  verts_3d[:, :, 1:2] * orig_size[0],
                                  torch.ones_like(verts_3d[:, :, 2:])), 2)
        verts_3d_cam = verts_3d_cam.transpose(2, 1)  # [B,3,N]

        # Conver to camera coords
        verts_3d_cam = (intrinsics.inverse() @ verts_3d_cam) * verts_depth  # [B,3,N]
        return verts_3d_cam.transpose(2, 1)

    def _transform_depthmap(self, verts_3d, intrinsics, R_mat, t_mat, orig_size):
        """Reproject depthmap format verts basd on the Camera transform matrix (R/t)
        Args:
            verts: [B,N,3] (depthmap format range=[0,1])
            intrinsics: [B,3,3]
            R_mat: [B,3,3]
            t_mat: [B,3,1]
            orig_size: (height, width)

        Note: 
            Currently image -> texture converting is not implemented yet.
            So, when you render rgb img (for 2-view-sfm), it would be better to use torch.grid_sample().
            (Recommended for depthmap/silhouette rendering)
        """
        # Convert from depthmap to camera coords
        verts_3d_cam = self.convert_depthmap_to_cam(verts_3d, intrinsics, orig_size)
        verts_3d_cam = verts_3d_cam.transpose(2, 1)  # [B,3,N]

        # Camera transform (R_mat & t_mat)
        pose_mat = torch.cat([R_mat, t_mat], dim=2)  # [B,3,4]
        verts_3d_ones = torch.ones_like(verts_3d_cam[:, :1, :])  # [B,1,N]
        verts_3d_cam_hom = torch.cat((verts_3d_cam, verts_3d_ones), 1)  # [B,4,N]
        verts_3d_cam2 = (pose_mat @ verts_3d_cam_hom).transpose(2, 1)  # [B,N,3]

        # Project into pixel-coords as depthmap format
        verts_3d_depth2 = self.project_cam_to_depthmap(verts_3d_cam2, intrinsics, orig_size)

        return verts_3d_depth2

    def forward(self, verts, faces, textures=None, intrinsics=None, R_mat=None, t_mat=None,
                mode=["rgb", "depth", "silhouette"], render_size=None):
        """Implementation of forward rendering methods.
        You should specify the rendering mode from [scene, silhouette, depth, face_index].
        """

        # Check batchsize
        assert verts.shape[0] == faces.shape[0], \
            "batchsize is not same between verts and faces"

        if "face_index" in mode:
            assert (intrinsics is None) and (R_mat is None) and (t_mat is None), \
                "K/R/t is not necessary in face_index-rendering-mode"
            return self._render_face_index_map(verts, faces, render_size)

        elif ("rgb" in mode) or ("depth" in mode) or ("silhouette" in mode):
            return self._render(verts, faces, textures, render_size, mode,
                                intrinsics=intrinsics, R_mat=R_mat, t_mat=t_mat)

        else:
            raise ValueError(
                "Choose mode from [None, 'silhouettes', 'depth', 'face_index']")

    def _render(self, verts, faces, textures=None, render_size=None, mode=["rgb", "depth", "silhouette"],
                intrinsics=None, R_mat=None, t_mat=None):
        """Rendering depth images from 3d mesh (which is depthmap format)
        Args:
            verts: [B,N,3(uvd)] (depthmap format) You need to concat verts_depths with verts_2d. range should be [0,1]
            faces: [B,M,3] (index pairs of face)
            textures:
            render_size: Specify the output size in the form of tuple (height, width)
            mode (str): Choose from ['rgb','depth','silhouette'] 
            intrinsics: [B,3,3]
            R_mat: [B,3,3]
            t_mat: [B,3,1]
        Returns:
            rendered_depths : [B,1,H,W]
        """

        assert verts.shape[2] == 3, "This function can deal with only 3d mesh.(depthmap format)"

        # Fill back
        if self.fill_back:
            faces = torch.cat(
                (faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        if render_size is None:
            render_size = self.render_size

        # Prepare elements
        img_max_size = max(render_size)
        height, width = render_size

        # If intrinsics/R_mat/t_mat are specified, reproject vertices on the other viewpoint.
        if (intrinsics is not None) and (R_mat is not None) and (t_mat is not None):
            verts = self._transform_depthmap(verts, intrinsics, R_mat, t_mat, render_size)

        # Resize verts [0,1] -> [-1,1] (You need to pay attention to scale!)
        verts = torch.cat(((verts[:, :, :1] * width / img_max_size) * 2.0 - 1.0,
                           (verts[:, :, 1:2] * height / img_max_size) * 2.0 - 1.0,
                           verts[:, :, 2:]), 2)
        faces = vertices_to_faces(verts, faces)

        # Rasterization
        render_dic = rasterize_image(faces, textures, img_max_size, self.anti_aliasing, mode=mode)
        # Final adjustment
        if "rgb" in mode:
            render_rgbs = flip(render_dic["rgb"], 2)[:, :, :height, :width]
            render_dic["rgb"] = render_rgbs.contiguous()
        if "depth" in mode:
            render_depths = flip(render_dic["depth"], 1)[:, :height, :width].unsqueeze(1)
            render_dic["depth"] = render_depths.contiguous()
        if "silhouette" in mode:
            render_silhouettes = flip(render_dic["silhouette"], 1)[:, :height, :width].unsqueeze(1)
            render_dic["silhouette"] = render_silhouettes.contiguous()

        return render_dic

    def _render_face_index_map(self, verts, faces, render_size=None):
        """Rendering face_index_map from 2d mesh for creating face silhouettes. 
        Args:
            verts: [B,N,2(uv)] 2D mesh extracted from scene image.
            faces: [B,M,3] (index pairs of face)
            render_size: Specify the output size in the form of tuple (height, width)
        Returns:
            face_index_map: [B,H,W] (pixel value means face idx in [1,M_max])
        """

        # Fill back
        if self.fill_back:
            faces = torch.cat(
                (faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        if render_size is None:
            render_size = self.render_size

        # Add z-axis to verts ([B,N_max,2]->[B,N_max,3])
        if verts.shape[2] == 2:
            z_verts = torch.ones_like(verts[:, :, :1])
            verts = torch.cat((verts, z_verts), 2)  # [B,N_max,3]

        # Prepare elements for rasterization
        batch_size = faces.shape[0]
        img_max_size = max(render_size)
        height, width = render_size

        # Resize verts [0,1] -> [-1,1] (You need to pay attention to scale!)
        verts = torch.cat(((verts[:, :, :1] * width / img_max_size) * 2.0 - 1.0,
                           (verts[:, :, 1:2] * height / img_max_size) * 2.0 - 1.0,
                           verts[:, :, 2:]), 2)
        faces = vertices_to_faces(verts, faces)

        # Prepare other elements
        face_index_map = torch.cuda.IntTensor(batch_size, img_max_size, img_max_size).fill_(-1)
        weight_map = torch.cuda.FloatTensor(batch_size, img_max_size, img_max_size, 3).fill_(0.0)
        depth_map = torch.cuda.FloatTensor(batch_size, img_max_size, img_max_size).fill_(100.0)  # self.far
        face_inv_map = torch.cuda.FloatTensor(1).fill_(0)
        faces_inv = torch.zeros_like(faces)

        # Face index rasterization
        face_index_map, _, _, _ = rasterize_cuda.forward_face_index_map(faces, face_index_map, weight_map,
                                                                        depth_map, face_inv_map, faces_inv,
                                                                        img_max_size, False, False, False)

        # Change pixel value in background area (-1 -> 0)
        face_index_map = face_index_map + 1

        return face_index_map[:, :height, :width].contiguous()
