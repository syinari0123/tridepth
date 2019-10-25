import os
import h5py
import numpy as np
import torch
import torch.utils.data as data

from dataloaders import make_dataset, transforms


RAW_HEIGHT, RAW_WIDTH = 480, 640  # raw image size
to_tensor = transforms.ToTensor()
color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)


def h5_loader(path):
    """Image/Depth extractor from h5 format file.
    Args:
        path (str): Path to h5 format file
    Returns:
        rgb (np.array): RGB image (shape=[H,W,3])
        depth (np.array): Depth image (shape=[H,W])
    """
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth


class NYUCamMat:
    """Calculate Resized Camera Intrinsic Matrix of NYUDepth dataset.

    Attributes:
        fx, fy (float): Original focal length expressed in pixel units
        cx, cy (float): Original principal point
        orig_height, orig_width (int): Original image size
    """

    def __init__(self):
        self.fx = 5.1885790117450188e+02
        self.fy = 5.1946961112127485e+02
        self.cx = 3.2558244941119034e+02
        self.cy = 2.5373616633400465e+02

    def __call__(self, batch_size=None, class_ids=None, img_size=(228, 304), inv_mat=False, t_tensor=True):
        """Calulate intrinsic matrix of input image size.

        Args:
            class_ids (list): None list (for calculation of batch_size)
            img_size (tuple): Target image size
            inv_mat (bool): Whether you need inverse matrix or not
            t_tensor (bool): Whether you need torch.Tensor or not
            batchsize (int): Batch size

        Returns:
            bool : Camera (Inverse) Matrix (np.array or torch.Tensor) (size:[1,3,3])
        """
        # Specify batchsize by batch_size or class_ids
        assert (batch_size is not None) or (class_ids is not None)
        if batch_size is None:
            batch_size = len(class_ids)

        # Prepare camera matrix
        scale_h = float(img_size[0] / RAW_HEIGHT)
        scale_w = float(img_size[1] / RAW_WIDTH)
        np_camera_mat = np.array([[self.fx * scale_w, 0., self.cx * scale_w],
                                  [0., self.fy * scale_h, self.cy * scale_h],
                                  [0., 0., 1.]], dtype=np.float32)

        # Whether you need inverse matrix or not.
        if inv_mat:
            np_camera_mat = np.linalg.inv(np_camera_mat)

        # Whether you need torch.Tensor or np.array or not.
        if t_tensor:
            t_mat = torch.from_numpy(np_camera_mat).unsqueeze(0)
            return t_mat.repeat(batch_size, 1, 1)
        else:
            np_mat = np_camera_mat[np.newaxis, :, :]
            return np.tile(np_mat, (batch_size, 1, 1))


class NYUDataset(data.Dataset):
    """NYUDepth Dataset Loader 

    Attributes:
        root_dir (str): Dataset directory path
        out_size (tuple): Input image size to CNN (Default: 228x304)
        val_split_rate (float): Ratio (0~1) of validation data in entire dataset (Default: 0.01)
    """

    def __init__(self, root_dir, data_type="train", img_size=(228, 304),
                 val_split_rate=0.01):
        self.root_dir = root_dir
        self.img_size = img_size
        self.val_split_rate = val_split_rate
        self.data_type = data_type

        # Prepare image list for each data type
        if (data_type == "train") or (data_type == "val"):
            # Create image list
            train_dir = os.path.join(self.root_dir, "train")
            self.imgs = make_dataset(train_dir, data_type, val_split_rate)

            # Transform (including data augmentation)
            if data_type == "train":
                self.transform = self.train_transform
            elif data_type == "val":
                self.transform = self.val_transform
            else:
                raise RuntimeError("Invalid dataset type: " + data_type + "\n"
                                   "Supported dataset types are: train, val, test")
        elif data_type == "test":
            test_dir = os.path.join(self.root_dir, "val")
            self.imgs = make_dataset(test_dir, data_type)
            self.transform = self.val_transform
        else:
            raise RuntimeError("Invalid dataset type: " + data_type + "\n"
                               "Supported dataset types are: train, val, test")

        # Prepare mesh extractor
        from tridepth.extractor import Mesh2DExtractor
        self.mesh_extractor = Mesh2DExtractor(canny_params={"denoise": False},
                                              at_params={"filter_itr": 4, "error_thresh": 0.01})

    def train_transform(self, rgb, depth):
        """
        [Reference]
        https://github.com/fangchangma/sparse-to-dense.pytorch/blob/master/dataloaders/nyu_dataloader.py

        Args:
            rgb (np.array): RGB image (shape=[H,W,3])
            depth (np.array): Depth image (shape=[H,W])

        Returns:
            torch.Tensor: Tranformed RGB image
            torch.Tensor: Transformed Depth image
            np.array: Transformed RGB image without color jitter (for 2D mesh creation)
        """
        # Parameters for each augmentation
        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # Perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / RAW_HEIGHT),
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.img_size),
            transforms.HorizontalFlip(do_flip)
        ])

        # Apply this transform to rgb/depth
        rgb_np_orig = transform(rgb)
        rgb_np_for_edge = np.asfarray(rgb_np_orig)  # Used for canny edge detection
        rgb_np = color_jitter(rgb_np_orig)  # random color jittering
        rgb_np = np.asfarray(rgb_np) / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np, rgb_np_for_edge

    def val_transform(self, rgb, depth):
        """
        [Reference]
        https://github.com/fangchangma/sparse-to-dense.pytorch/blob/master/dataloaders/nyu_dataloader.py

        Args:
            rgb (np.array): RGB image (shape=[H,W,3])
            depth (np.array): Depth image (shape=[H,W])

        Returns:
            torch.Tensor: Tranformed RGB image
            torch.Tensor: Transformed Depth image
            np.array: Transformed RGB image without color jitter (for 2D mesh creation)
        """
        transform = transforms.Compose([
            transforms.Resize(240.0 / RAW_HEIGHT),
            transforms.CenterCrop(self.img_size),
        ])

        # Apply this transform to rgb/depth
        rgb_np_orig = transform(rgb)
        rgb_np_for_edge = np.asfarray(rgb_np_orig)  # Used for mesh creation
        rgb_np = np.asfarray(rgb_np_orig) / 255
        depth_np = transform(depth)

        return rgb_np, depth_np, rgb_np_for_edge

    def __getraw__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, class_id = self.imgs[index]
        rgb, depth = h5_loader(path)
        return rgb, depth, class_id

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

        # To tensor
        input_tensor = to_tensor(rgb_np)
        depth_tensor = to_tensor(depth_np).unsqueeze(0)  # [1,H,W]

        # Extract mesh
        base_mesh = self.mesh_extractor(np.uint8(rgb_np_for_edge))

        if self.data_type == "train":
            return input_tensor, depth_tensor, base_mesh

        if (self.data_type == "val") or (self.data_type == "test"):
            # Preserve original resolution for evaluation/visualization
            orig_transform = transforms.Compose([
                transforms.CenterCrop((456, 608)),
            ])
            orig_input_tensor = orig_transform(raw_rgb)
            orig_depth_tensor = orig_transform(raw_depth)

            # To tensor
            orig_input_tensor = to_tensor(orig_input_tensor)
            orig_depth_tensor = to_tensor(orig_depth_tensor).unsqueeze(0)

            return input_tensor, depth_tensor, base_mesh, orig_input_tensor, orig_depth_tensor

    def __len__(self):
        return len(self.imgs)
