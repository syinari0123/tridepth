import argparse
import os
import cv2
import torch
import torch.nn.functional as F
import datetime
import numpy as np

import sys
sys.path.append('..')

from auxiliary import str2bool, fix_random_seed, prepare_logdir, save_arguments
from auxiliary import TermLogger
from auxiliary import SaveImages, depth_evaluations, EvalResultWriter
from dataloaders_eval import NYUDatasetEval, prepare_eval_dataloader
from dataloaders import NYUCamMat
from models.functional import inpaint_depth
from mesh_simplification import create_grid_mesh, load_obj
from tridepth.renderer import Renderer, write_obj_with_texture


def main(args):
    # CUDA settings
    fix_random_seed(seed=46)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Logger
    log_dir = prepare_logdir(log_path=args.result_path, descript=args.method_name)
    save_arguments(args, log_dir)

    # Mesh save dir
    mesh_dir = os.path.join(log_dir, "mesh")
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    # Dataset
    print("=> preparing dataloader")
    cam_mat = NYUCamMat()
    test_dataloader = prepare_eval_dataloader(args.data_path,
                                              method_name=args.method_name,
                                              img_size=(228, 304))
    print('Test set\t', len(test_dataloader))

    # Prepare renderer
    render = Renderer()
    render.fill_back = True  # All the faces of mesh are inverse from single view

    # Evaluation (image saver + score calculator)
    img_mesh_saver = SaveImages(cmap_type="jet")
    img_raw_saver = SaveImages(cmap_type="jet")
    eval_avg_mesh_writer = EvalResultWriter(os.path.join(log_dir, "naive_mesh_avg.csv"))
    eval_avg_raw_writer = EvalResultWriter(os.path.join(log_dir, "naive_raw_avg.csv"))
    eval_all_mesh_writer = EvalResultWriter(os.path.join(log_dir, "naive_mesh_all.csv"))
    eval_all_raw_writer = EvalResultWriter(os.path.join(log_dir, "naive_raw_all.csv"))

    # Define logger
    print("=> start evaluation")
    logger = TermLogger(
        n_epochs=len(test_dataloader), train_size=0, valid_size=0,
        use_flag=args.print_progress
    )

    # Start evaluation
    logger.start_bar("epoch")
    for e_idx, (scenes, gt_depths, mesh_list, orig_scenes, orig_gt_depths, est_depths) in enumerate(test_dataloader):
        logger.update_bar(e_idx, "epoch")

        # To GPU
        orig_gt_depths = orig_gt_depths.to(device)

        # Parameters
        _, _, height, width = scenes.shape
        _, _, o_height, o_width = orig_scenes.shape
        num_patch = len(mesh_list[0].faces)
        num_verts = num_patch * 3

        # To numpy
        np_orig_scenes = orig_scenes[0].permute(1, 2, 0).cpu().numpy()
        np_orig_est_depths = cv2.resize(est_depths[0, 0].cpu().numpy(), (o_width, o_height))
        np_orig_intrinsic = cam_mat(batch_size=1, img_size=(o_height, o_width), t_tensor=False)[0]  # [2,3]

        # Create Dense-grid-mesh by estimeted dense depthmap
        full_obj_name = "{}/{:06d}_full.obj".format(mesh_dir, e_idx)
        orig_vert_size, orig_face_size = create_grid_mesh(np_orig_scenes, np_orig_est_depths, np_orig_intrinsic,
                                                          obj_name=full_obj_name)

        # Params for simplification
        if args.simplify_mode == "face":
            simple_ratio = num_patch / orig_face_size
        elif args.simplify_mode == "vertex":
            simple_ratio = num_verts / orig_vert_size

        exec_file_path = "thirdparty/Fast-Quadric-Mesh-Simplification/bin.Linux/simplify"  # TODO: You need to do "chmod +x simplify"
        simple_obj_name = full_obj_name.replace("full", "simple")

        # Execute simplification
        os.system("{} {} {} {}".format(exec_file_path, full_obj_name, simple_obj_name, simple_ratio))

        # Reload simplified mesh (for depthmap rendering)
        simple_verts_3d, simple_faces = load_obj(simple_obj_name, device=device)

        # Re-save simplified mesh with texture
        orig_intrinsic = cam_mat(batch_size=1, img_size=(o_height, o_width)).to(device)
        np_verts_2d = render.project_cam_to_depthmap(simple_verts_3d, orig_intrinsic, orig_size=(o_height, o_width))

        write_obj_with_texture(simple_obj_name,
                               simple_verts_3d[0].detach().cpu().numpy(),
                               simple_faces[0].detach().cpu().numpy(),
                               np_orig_scenes / 255.0,
                               np_verts_2d[0].detach().cpu().numpy())

        # Only delete full mesh
        #os.system('rm {}'.format(full_obj_name))

        # Convert from Camera-coords to UV-coords (depthmap format)
        orig_intrinsic = cam_mat(batch_size=1, img_size=(o_height, o_width), t_tensor=True).to(device)
        simple_verts = render.project_cam_to_depthmap(simple_verts_3d, orig_intrinsic, orig_size=(o_height, o_width))

        # Rendering depth & silhouette of simplified mesh
        render_dic = render(simple_verts, simple_faces, mode=["depth", "silhouette"],
                            render_size=(o_height, o_width))
        render_depth = render_dic["depth"]
        render_silhouette = render_dic["silhouette"]

        # Inpaint depthmap based on silhouette
        naive_mesh_depths = torch.stack([inpaint_depth(render_depth[0, 0], render_silhouette[0, 0])], 0).to(device)
        naive_raw_depths = torch.from_numpy(np_orig_est_depths).view(1, 1, o_height, o_width).to(device)

        # Calculate evaluation scores (and average it)
        eval_mesh_scores = depth_evaluations(naive_mesh_depths.detach(), orig_gt_depths.detach())
        eval_raw_scores = depth_evaluations(naive_raw_depths.detach(), orig_gt_depths.detach())

        # Append size (only for mesh)
        eval_mesh_scores["num_patch"] = len(simple_faces[0])
        eval_mesh_scores["num_vertex"] = len(simple_verts[0])

        # Write into writer
        eval_avg_mesh_writer.update(eval_mesh_scores, 1)
        eval_avg_raw_writer.update(eval_raw_scores, 1)
        eval_all_mesh_writer.update(eval_mesh_scores, 1)
        eval_all_raw_writer.update(eval_raw_scores, 1)

        # Write all-evaluation scores into csv
        eval_all_mesh_writer.write_avg_into_csv(e_idx)
        eval_all_raw_writer.write_avg_into_csv(e_idx)

        # Save images
        edgemaps = torch.from_numpy(mesh_list[0].edgemap).unsqueeze(0).unsqueeze(0)

        mesh_depths = F.interpolate(naive_mesh_depths, size=(height, width)).to(device)
        raw_depths = F.interpolate(naive_raw_depths, size=(height, width)).to(device)
        mesh_edges = F.interpolate(edgemaps, size=(height, width)).to(device)

        # Write images into file
        img_mesh_saver.reset()
        img_raw_saver.reset()
        img_mesh_saver.update_merged_image(scenes[0], gt_depths[0], mesh_depths[0], mesh_edges[0])
        img_raw_saver.update_merged_image(scenes[0], gt_depths[0], raw_depths[0], mesh_edges[0])
        img_mesh_saver.save_merged_image(os.path.join(mesh_dir, "{:06d}_mesh.png".format(e_idx)))
        img_raw_saver.save_merged_image(os.path.join(mesh_dir, "{:06d}_raw.png".format(e_idx)))

    # Write evaluation scores into csv
    eval_avg_mesh_writer.write_avg_into_csv(0)
    eval_avg_raw_writer.write_avg_into_csv(0)

    logger.finish_bar("epoch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic settings
    parser.add_argument('--data-path', type=str, default="/media/kmmasaya/Maxtor/nyudepthv2/nyudepthv2")
    parser.add_argument('--method-name', type=str, default="laina", choices=["eigen", "laina", "dorn"])
    parser.add_argument('--simplify-mode', type=str, default="face", choices=["vertex", "face"])

    # Dataset settings
    parser.add_argument('--result-path', type=str, default="result")
    parser.add_argument('--print-progress', type=str2bool, default="true")

    args = parser.parse_args()
    print(args)
    main(args)
