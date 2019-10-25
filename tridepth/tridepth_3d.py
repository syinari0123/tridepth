import torch
from copy import deepcopy

from tridepth import BatchBaseMesh
from tridepth.renderer import Renderer, vertices_to_faces, write_obj_with_texture


class TriDepth:
    def __init__(self, base_imgs, cam_mat, mesh_list,
                 render_size=None, device=torch.device('cuda')):
        # Parameters
        self.batchsize, _, self.height, self.width = base_imgs.shape
        self.device = device

        # Basic elements
        self.base_imgs = base_imgs
        self.cam_mat = cam_mat

        # Prepare batched triangle-patch-cloud (batched verts & faces)
        self.base_mesh = BatchBaseMesh(mesh_list, device=device)
        self.base_edges = self.base_mesh.edgemaps

        # - Main properties
        self.verts_2d = self.base_mesh.verts_2d               # [B,Nmax(=3Mmax),2](BatchTensor)
        self.faces = self.base_mesh.faces                     # [B,Mmax,3](BatchTensor)
        self.adj_edges = self.base_mesh.adj_edges             # [B,Lmax,2(=f1,f2),2(=v1,v2)](BatchTensor)
        self.num_patch_list = self.base_mesh.faces.size_list  # list

        # - 3D structure predicted by CNNs (call self.assign_facedepths)
        self.verts_3d = deepcopy(self.verts_2d)               # [B,Nmax,3](BatchTensor)

        # Renderer
        self.render_size = (self.height, self.width) if render_size is None else render_size
        self.render = Renderer(render_size=self.render_size)

    def extarct_face_features(self, scene_feats, feat_type):
        """
        Args:
            scene_feats: [B,F,H,W]
            feat_type: 'face_centroid' vs 'face_pooling'
        Returns:
            mesh_feats: [B,F,M]
        """
        assert feat_type in ["face_centroid", "face_pooling"]

        if feat_type == "face_centroid":
            face_verts_2d = vertices_to_faces(self.verts_2d.tensor,
                                              self.faces.tensor)  # [B,M,3(id),2(uv)]
            face_centroids = face_verts_2d.mean(dim=2)  # [B,M,2]

            from models.networks import feature_sampling
            face_centroid_feats = feature_sampling(scene_feats, face_centroids)

            return face_centroid_feats

        elif feat_type == "face_pooling":
            # Rendering face index map ([B,H,W] whose value is face_idx in [1,M], face_idx=0 means background)
            face_index_maps = self.render(self.verts_2d.tensor,
                                          self.faces.tensor,
                                          mode="face_index")  # [B,H,W]

            from models.networks import FacePooling
            face_pooling = FacePooling(pool_type="max")  # Default: max
            face_pooling_feats = face_pooling(scene_feats, face_index_maps, max_index=self.faces.max_size)  # [B,F,M]

            return face_pooling_feats

    def assign_facedepths(self, face_depth_params):
        """
        Args:
            face_depth_params: [B,Mmax,3]

        ===
        Calculate depths of each vertex (self.verts_2d) from predicted face_depth_params.
        This process is summarized as following.

        1. Calculate face's planar equation (z=alpha*x'+beta*y'+gamma(=d)) from following elements.
            normal-vector = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
                          = (a,b,c)
            face-centroid = (u,v,d)

        2. All the depths of pixels on a face are calculated from planar equation with these elements.
            D(x,y) = -(a/c)*(x-u) - (b/c)*(y-v) + d
                   = -tan(theta)*cos(phi)*(x-u) - tan(theta)*sin(phi)*(y-v) + d
                   = [[-tan(theta)*cos(phi)],       [[x - u],
                      [-tan(theta)*sin(phi)],    @   [y - v],
                      [          d         ]]^T      [  1  ]]
                    = [alpha,beta,d] @ [x',y',1]
                    = face_depth_params @ face_verts_coords

           As the intermediate elements, we calculate the following elements separately.
            face_depth_params: [alpha,beta,d] in planar equation (z=alpha*x'+beta*y'+d);  # shape=[B,Mmax,3]
            face_verts_coords: [x',y',1], pixel coords of 3 vertices on each face;        # shape=[B,Mmax,3(id),3(uv1)]

        3. Finally, we obtain the depths of each face_verts_2d ([B,Mmax,3,2])
            face_verts_depth = face_depth_params @ face_verts_coords                      # shape=[B,Mmax,3,1]
        """
        # Prepare face_verts_coords for process2
        face_verts_2d = vertices_to_faces(self.verts_2d.tensor, self.faces.tensor)  # [B,Mmax,3(id),2(uv)]
        face_centroids = face_verts_2d.mean(dim=2, keepdim=True)                    # [B,Mmax,1,2(uv)]
        face_verts_coords = face_verts_2d - face_centroids                          # [B,Mmax,3(id),2(uv)]
        face_verts_zaxis = torch.ones_like(face_verts_coords[:, :, :, :1])
        face_verts_coords = torch.cat((face_verts_coords, face_verts_zaxis), 3)     # [B,Mmax,3(id),3(uv1)]

        # Reshape for multiplication (process3)
        face_depth_params = face_depth_params.contiguous().view(-1, 1, 3)     # [B*Mmax,1,3(=alpha,beta,d)]
        face_verts_coords = face_verts_coords.view(-1, 3, 3).transpose(1, 2)  # [B*Mmax,3(=uv1),3(=id)]

        # Multiplication ([BMmax,1,3(=abd)] @ [BMmax,3(=xy1),3(=id)] -> [BMmax,1,3(=id)])
        face_verts_depth = torch.bmm(face_depth_params, face_verts_coords)
        face_verts_depth = face_verts_depth.view(self.batchsize, -1, 3, 1)    # [B,Mmax,3(=id),1]

        # This func is for training (to avoid vanishing gradient problem)
        face_verts_depth = face_verts_depth.clamp(min=0.01, max=10.01)

        # Integrate predicted depths into face_verts_2d (shape=[B,3Mmax,3(uvd)])
        verts_3d_tensor = torch.cat((face_verts_2d, face_verts_depth), 3).view(self.batchsize, -1, 3)
        self.verts_3d.update_tensor(verts_3d_tensor)

    def render_depths(self, render_size=None):
        """
        Args:
            train_flag (bool): whether training or not.
        Returns:
            render_depths: [B,1,H,W]
        """
        if render_size is None:
            render_size = self.render_size

        # Rendering
        render_dic = self.render(self.verts_3d.tensor, self.faces.tensor,
                                 render_size=render_size,
                                 mode=["depth", "silhouette"])
        # Extract elements from render_dic
        render_depths = render_dic["depth"]

        return render_depths

    def _calculate_patch_distances(self, bth_verts, bth_adj_edges):
        """Calculate patch distances
        Args:
            bth_verts: [N,3]
            bth_adj_edges: [L,2(=f1,f2),2(=v1,v2)]
        """
        edge_size = bth_adj_edges.shape[0]
        adj_edge_vert_depths = bth_verts[bth_adj_edges.view(-1).long(), 2].view(edge_size, 2, 2)  # [L,2,2(=v1,v2)]
        adj_edge_dist = torch.abs(adj_edge_vert_depths[:, 0] - adj_edge_vert_depths[:, 1]).mean(dim=1)  # [L,2(=v1,v2)]
        return adj_edge_dist

    def _connect_patches(self, bth_faces, bth_adj_edges, mask=None):
        """Connect patches specified in (masked) bth_adj_edges
        Args:
            bth_faces: [M,3]
            bth_adj_edges: [L,2(=f1,f2),2(=v1,v2)]
            mask: [L](boolean)
        """
        if mask is not None:
            assert mask.shape[0] == bth_adj_edges.shape[0]
            bth_adj_edges = bth_adj_edges[mask]

        connected_faces = bth_faces.clone()
        for l in range(bth_adj_edges.shape[0]):
            e1, e2 = bth_adj_edges[l]

            # Replace vert_id in connected_faces (e1->e2)
            connected_faces = torch.where(connected_faces == e1[0], e2[0], connected_faces)  # v1
            connected_faces = torch.where(connected_faces == e1[1], e2[1], connected_faces)  # v2

            # Replace vert_id in filter
            if l < len(bth_adj_edges) - 1:
                bth_adj_edges[l + 1:] = torch.where(bth_adj_edges[l + 1:] == e1[0], e2[0], bth_adj_edges[l + 1:])  # v1
                bth_adj_edges[l + 1:] = torch.where(bth_adj_edges[l + 1:] == e1[1], e2[1], bth_adj_edges[l + 1:])  # v2

        return connected_faces

    def save_into_obj(self, filename, b_idx=0, texture="img", connect_th=0.1):
        """Save 3D structure into obj-file, 
        whose filename and batch_index should be specified in arguments.
        """
        assert texture in ["img", "edge"]

        # Prepare textures
        if texture == "img":
            max_val = self.base_imgs[b_idx].max()
            min_val = self.base_imgs[b_idx].min()
            textures = (self.base_imgs[b_idx] - min_val) / (max_val - min_val)
            textures = textures.permute(1, 2, 0).detach()
        elif texture == "edge":
            max_val = self.base_edges[b_idx].max()
            min_val = self.base_edges[b_idx].min()
            textures = (self.base_edges[b_idx] - min_val) / (max_val - min_val)
            textures = textures[0].detach()

        # Prepare bth elements of mesh
        bth_adj_edges = self.adj_edges.raw_data(b_idx)
        bth_verts = self.verts_3d.raw_data(b_idx)
        bth_faces = self.faces.raw_data(b_idx)

        # Connect edges
        if connect_th is not None:
            connect_mask = self._calculate_patch_distances(bth_verts, bth_adj_edges) < connect_th
            connected_faces = self._connect_patches(bth_faces, bth_adj_edges, mask=connect_mask)
            faces = connected_faces.detach().cpu().numpy()
        else:
            faces = self.base_mesh.mesh_list[b_idx].faces

        # Convert depthmap to cam_coords
        device = self.verts_3d.tensor.device
        intrinsics = self.cam_mat(img_size=self.render_size,
                                  t_tensor=True,
                                  batch_size=self.batchsize).to(device)
        verts_3d_cam = self.render.convert_depthmap_to_cam(self.verts_3d.tensor, intrinsics, self.render_size)

        # To numpy
        np_verts_3d_cam = verts_3d_cam[b_idx].detach().cpu().numpy()
        np_verts_2d = self.base_mesh.mesh_list[b_idx].verts_2d

        # Save as obj-file with textures
        write_obj_with_texture(filename,
                               np_verts_3d_cam, faces,
                               textures.cpu().numpy(),
                               np_verts_2d)
