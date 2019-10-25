import torch
from dataloaders_eval import NYUDatasetEval


def prepare_eval_dataloader(data_path, method_name,
                            img_size=(228, 304)):

    test_dataset = NYUDatasetEval(
        root_dir=data_path, method_name=method_name, data_type="test",
        img_size=img_size, val_split_rate=0
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, drop_last=True,
        shuffle=False, num_workers=0, collate_fn=eval_collate,
    )

    return test_dataloader


def eval_collate(batch):
    """
    Returns:
        input_tensor: [3,H,W]
        depth_tensor: [1,H,W]
    """
    # Fixed size
    scenes = torch.stack([item[0] for item in batch], 0)  # [B,3,H,W]
    depths = torch.stack([item[1] for item in batch], 0)  # [B,1,H,W]

    mesh_list = [item[2] for item in batch]               # list of mesh

    # [[1,H,W],...] This format can deal with different size in batch
    orig_scenes = torch.stack([item[3] for item in batch], 0)
    orig_depths = torch.stack([item[4] for item in batch], 0)
    orig_est_depths = torch.stack([item[5] for item in batch], 0)

    return [scenes, depths, mesh_list, orig_scenes, orig_depths, orig_est_depths]
