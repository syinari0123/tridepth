import torch
from dataloaders import NYUDataset


def prepare_dataloader(data_path,
                       datatype_list=["train", "val", "test"],
                       batchsize=1,
                       workers=0,
                       img_size=(228, 304),
                       val_split_rate=0.01):

    assert all([dtype in ["train", "val", "test"] for dtype in datatype_list]), \
        "You need to select datatype from train/val/test strings."

    dataloader_dic = {"train": None, "val": None, "test": None}

    if "train" in datatype_list:
        train_dataset = NYUDataset(
            root_dir=data_path, data_type="train", img_size=img_size,
            val_split_rate=val_split_rate
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batchsize, drop_last=True,
            shuffle=True, num_workers=int(workers), collate_fn=train_collate
        )
        dataloader_dic["train"] = train_dataloader

    if "val" in datatype_list:
        val_dataset = NYUDataset(
            root_dir=data_path, data_type="val", img_size=img_size,
            val_split_rate=val_split_rate
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, drop_last=True,
            shuffle=False, num_workers=int(workers), collate_fn=val_test_collate
        )
        dataloader_dic["val"] = val_dataloader

    if "test" in datatype_list:
        test_dataset = NYUDataset(
            root_dir=data_path, data_type="test", img_size=img_size,
            val_split_rate=val_split_rate
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, drop_last=True,
            shuffle=False, num_workers=int(workers), collate_fn=val_test_collate,
        )
        dataloader_dic["test"] = test_dataloader

    return dataloader_dic


def train_collate(batch):
    """
    Returns:
        input_tensor: [3,H,W]
        depth_tensor: [1,H,W]
        mesh_list
    """
    # Fixed size
    scenes = torch.stack([item[0] for item in batch], 0)  # [B,3,H,W]
    depths = torch.stack([item[1] for item in batch], 0)  # [B,1,H,W]
    mesh_list = [item[2] for item in batch]         # list of mesh

    return [scenes, depths, mesh_list]


def val_test_collate(batch):
    """
    Returns:
        input_tensor: [3,H,W]
        depth_tensor: [1,H,W]
        mesh_list
        orig_input_tensor: [3,H,W]
        orig_depth_tensor: [1,H,W]
    """
    # Fixed size
    scenes = torch.stack([item[0] for item in batch], 0)       # [B,3,H,W]
    depths = torch.stack([item[1] for item in batch], 0)       # [B,1,H,W]
    mesh_list = [item[2] for item in batch]                    # list of mesh
    orig_scenes = torch.stack([item[3] for item in batch], 0)  # [B,3,H',W']
    orig_depths = torch.stack([item[4] for item in batch], 0)  # [B,1,H',W']

    return [scenes, depths, mesh_list, orig_scenes, orig_depths]
