import os
import numpy as np
import h5py
from scipy import io

import sys
sys.path.append('..')
from dataloaders import NYUDataset
from dataloaders import make_dataset, transforms

to_tensor = transforms.ToTensor()


class NYUDatasetEval(NYUDataset):
    """
    """

    def __init__(self, root_dir, method_name, data_type="train", img_size=(228, 304),
                 val_split_rate=0.01):
        super().__init__(root_dir, data_type, img_size, val_split_rate)

        # Load existing method's depthmap results (Mat-file)
        if method_name == "eigen":
            mat_path = "existing_results/eigen_depth.mat"
            mat_depth = np.array(io.loadmat(mat_path)["depths"])       # [240,320,654]
            self.mat_depth = mat_depth.transpose(2, 0, 1)              # [654,240,320]
        elif method_name == "laina":
            mat_path = "existing_results/laina_depth.mat"
            mat_depth = np.array(h5py.File(mat_path)["predictions"])   # [654,640,480]
            self.mat_depth = mat_depth.transpose(0, 2, 1)              # [654,480,640]
        elif method_name == "dorn":
            mat_path = "existing_results/dorn_depth.mat"
            self.mat_depth = np.array(io.loadmat(mat_path)["depths"])  # [654,480,640]
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (rgb, depth) the raw data.
        """
        raw_rgb, raw_depth, _ = self.__getraw__(index)
        if self.transform is not None:
            rgb_np, depth_np, rgb_np_for_edge = self.transform(raw_rgb, raw_depth)
        else:
            raise RuntimeError("transform not defined")

        input_tensor = to_tensor(rgb_np)
        depth_tensor = to_tensor(depth_np).unsqueeze(0)  # [1,H,W]

        # Extract mesh
        base_mesh = self.mesh_extractor(np.uint8(rgb_np_for_edge))

        # Preserve original resolution for evaluation/visualization
        orig_transform = transforms.Compose([
            transforms.CenterCrop((456, 608)),
        ])
        orig_input_tensor = orig_transform(raw_rgb)
        orig_depth_tensor = orig_transform(raw_depth)

        # To tensor
        orig_input_tensor = to_tensor(orig_input_tensor)
        orig_depth_tensor = to_tensor(orig_depth_tensor).unsqueeze(0)

        # Estimated depthmaps (added for evaluation)
        est_depth_np = np.asfarray(self.mat_depth[index], dtype='float')  # numpy
        est_depth_tensor = to_tensor(est_depth_np).unsqueeze(0)

        return input_tensor, depth_tensor, base_mesh, orig_input_tensor, orig_depth_tensor, est_depth_tensor
