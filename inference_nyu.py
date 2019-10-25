import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn

# local libraries
from auxiliary import str2bool, fix_random_seed
from dataloaders import NYUCamMat, prepare_dataloader
from models import Model
from models.functional import depth2normal
from auxiliary import SaveImages


def main():
    # CUDA settings
    fix_random_seed(seed=46)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    print("=> Creating model")
    cudnn.benchmark = True
    model = Model(cam_mat=NYUCamMat(), model_type=args.model_type,
                  loss_type="l1", normal_weight=0.5).to(device)
    if args.pretrained_path != "":
        print("=> using checkpoints")
        model.load_state_dict(torch.load(args.pretrained_path))

    # Dataset
    print("=> Preparing dataloader")
    test_dataloader = prepare_dataloader(args.data_path,
                                         datatype_list=["test"],
                                         batchsize=1,
                                         workers=0,
                                         img_size=(228, 304),
                                         val_split_rate=0.01)["test"]
    print('Test set\t', len(test_dataloader))

    # Prepare result dir
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    # Prepare saver
    img_saver = SaveImages(cmap_type="jet")

    model.eval()
    for idx, (scenes, gt_depths, mesh_list, _, _) in enumerate(test_dataloader):
        print("{}/{}".format(idx, len(test_dataloader)))

        batch_size, _, height, width = scenes.shape
        # To gpu
        scenes = scenes.to(device)
        gt_depths = gt_depths.to(device)

        # Load information
        batchsize, _, height, width = scenes.shape

        # Inference + loss calculation
        with torch.no_grad():
            tridepth = model(scenes, mesh_list)

        # Convert tridepth format from depthmap to camera_coords and save it as obj-format.
        if args.rep_type == "patch_cloud":
            connect_th = None
        elif args.rep_type == "mesh":
            connect_th = 0.1
        tpc_save_file = os.path.join(args.result_path, "{:06d}.obj".format(idx))
        tridepth.save_into_obj(tpc_save_file, b_idx=0, texture="img", connect_th=connect_th)

        # Calculate evaluation scores (and average it)
        pred_depths = tridepth.render_depths(render_size=(height, width))

        # Calculate normalmap
        #inv_intrinsics = model.cam_mat(img_size=(height, width),
        #                               inv_mat=True,
        #                               t_tensor=True,
        #                               batch_size=batch_size).to(device)
        #gt_normals = depth2normal(gt_depths, inv_intrinsics)
        #pred_normals = depth2normal(pred_depths, inv_intrinsics)
        # Normalize [-1,1] -> [0,1]
        #gt_normals = (gt_normals + 1.0) / 2.0
        #pred_normals = (pred_normals + 1.0) / 2.0

        # Save into file
        img_save_file = os.path.join(args.result_path, "{:06d}.png".format(idx))
        img_saver.reset()
        img_saver.update_merged_image(scenes[0], gt_depths[0], pred_depths[0], tridepth.base_edges[0])
        img_saver.save_merged_image(img_save_file)
    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dir path
    parser.add_argument('--data-path', default="~/datasets/nyudepthv2", type=str)
    parser.add_argument('--pretrained-path', default="pretrained/weight_upconv.pth", type=str)
    parser.add_argument('--model-type', type=str, default="upconv", choices=["simple", "upconv"])
    parser.add_argument('--seed', default=46, type=int)

    # output representation type
    parser.add_argument('--rep-type', type=str, default="patch_cloud", choices=["patch_cloud", "mesh"])

    # output path
    parser.add_argument('--result-path', type=str, default="result")

    args = parser.parse_args()
    print(args)
    main()
