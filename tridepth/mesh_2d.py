import numpy as np
import trimesh
import torch


class BatchTensor:
    def __init__(self, array_list, device=torch.device("cuda")):
        self.device = device

        # Main elements
        self.tensor = None     # [B,Nmax,...](torch.Tensor)
        self.size_list = None  # [N_1,N_2,...,N_B]
        self.max_size = None   # Nmax
        self._add_offset(array_list)

        self.shape = self.tensor.shape

    def _add_offset(self, array_list):
        self.size_list = [len(a) for a in array_list]
        self.max_size = max(self.size_list)

        tensor_list = []
        for idx in range(len(self.size_list)):
            offset = self.max_size - self.size_list[idx]
            if offset > 0:
                offset_shape = (offset,) + array_list[idx].shape[1:]
                array_dtype = array_list[idx].dtype
                tmp_array = np.concatenate((array_list[idx], np.zeros(offset_shape, dtype=array_dtype)), 0)
            else:
                tmp_array = array_list[idx]
            tensor_list.append(torch.from_numpy(tmp_array))
        self.tensor = torch.stack(tensor_list, 0).to(self.device)

    def update_tensor(self, tgt_tensor):
        assert self.tensor.shape[:2] == tgt_tensor.shape[:2]
        self.tensor = tgt_tensor

    def raw_data(self, b_idx):
        return self.tensor[b_idx, :self.size_list[b_idx]]


class BatchBaseMesh:
    def __init__(self, mesh_list, device=torch.device("cuda")):
        self.device = device
        self.mesh_list = mesh_list  # Original mesh list

        # Batched disconnected mesh (BatchTensor)
        self.verts_2d = None   # [B,Nmax,2](float) (offset=0)
        self.faces = None      # [B,Mmax,3](int) (offset=0)
        self.edgemaps = None   # [B,1,H,W](float)
        self.adj_edges = None  # [B,Lmax,2(=f1,f2),2(=v1,v2)](int) (offset=0)

        self._initialize()

    def _initialize(self):
        """Prepare batch of disconnected mesh
        """
        # Prepare list of each elements
        verts_2d_list = [m.verts_2d for m in self.mesh_list]
        faces_list = [m.faces for m in self.mesh_list]
        edgemap_list = [torch.from_numpy(m.edgemap).unsqueeze(0) for m in self.mesh_list]
        adj_edges_list = [m.adj_edges for m in self.mesh_list]

        # Create BatchTensor
        self.verts_2d = BatchTensor(verts_2d_list, device=self.device)
        self.faces = BatchTensor(faces_list, device=self.device)
        self.adj_edges = BatchTensor(adj_edges_list, device=self.device)

        self.edgemaps = torch.stack(edgemap_list, 0)

    def save_obj(self, filename, b_idx=0):
        """Save mesh as obj format (Please specify batch_index)
        """
        self.mesh_list[b_idx].save_obj(filename)


class BaseMesh:
    def __init__(self, mesh_dic):
        # Original mesh
        self.raw_verts_2d = mesh_dic["vertices"]
        self.raw_faces = mesh_dic["triangles"]
        self.edgemap = mesh_dic["edgemap"] / 255.0

        # Disconnected mesh (ndarray)
        self.verts_2d = None   # [N(=3M),2](float)
        self.faces = None      # [M,3](int)
        self.adj_edges = None  # [L,2(=f1,f2),2(=v1,v2)](int)

        self._initialize()

    def _convert_edges(self, adj_face, adj_edge):
        """Convert edges from self.raw_verts_2d to self.verts_2d
        Args:
            faces: [f1, f2]
            edges: [e1, e2]
        Returns:
            new_edges: [[f1_v1, f1_v2], [f2_v1, f2_v2]]
        """
        def _convert_vert_id(face_id, vert_id):
            """Convert vert_id in lth-face to new_vert_id
            Args:
                face_id (int): 
                vert_id (int): this id should be within self.raw_faces[face_id]
            Returns:
                new_vert_id (int):
            """
            lth_face = self.raw_faces[face_id]
            assert(np.min(abs(lth_face - vert_id)) == 0)
            new_vert_id = np.argmin(abs(lth_face - vert_id)) + (face_id * 3)
            return new_vert_id

        # Convert id for each elements in adj_edges
        face1_v1 = _convert_vert_id(adj_face[0], adj_edge[0])
        face1_v2 = _convert_vert_id(adj_face[0], adj_edge[1])
        face2_v1 = _convert_vert_id(adj_face[1], adj_edge[0])
        face2_v2 = _convert_vert_id(adj_face[1], adj_edge[1])

        return [[face1_v1, face1_v2], [face2_v1, face2_v2]]

    def _initialize(self):
        """Prepare triangular-patch-cloud, separating all the faces of the given mesh
        """
        # For triangular-patch-cloud
        self.verts_2d = self.raw_verts_2d[self.raw_faces.reshape(-1), :]  # [3M,2]
        self.faces = np.arange(len(self.verts_2d)).reshape(-1, 3)         # [M,3]

        # For disconnected mesh
        mesh_for_adj = trimesh.Trimesh(vertices=self.raw_verts_2d,
                                       faces=self.raw_faces, process=False)
        raw_adj_faces = mesh_for_adj.face_adjacency        # [L,2(=raw_face_ids)]
        raw_adj_edges = mesh_for_adj.face_adjacency_edges  # [L,2(=raw_vert_ids)]
        edge_size = raw_adj_edges.shape[0]

        # Convert vert_ids in adj_edges from self.raw_verts_2d to self.verts_2d (shape=[L,2(=f1,f2),2(=v1,v2)])
        self.adj_edges = np.array([self._convert_edges(raw_adj_faces[l], raw_adj_edges[l]) for l in range(edge_size)])

        # Cast each ndarray to specified type
        self.verts_2d = self.verts_2d.astype(np.float32)  # float
        self.faces = self.faces.astype(np.int32)          # int
        self.edgemap = self.edgemap.astype(np.float32)    # int
        self.adj_edges = self.adj_edges.astype(np.int32)  # int

    def save_obj(self, filename):
        """Save mesh into obj-file (for debug)
        """
        verts_3d = np.concatenate((self.verts_2d, np.ones_like(self.verts_2d[:, :1])), 1)
        mesh = trimesh.Trimesh(vertices=verts_3d, faces=self.faces, process=False)
        trimesh.exchange.export.export_mesh(mesh, filename)
