import os
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from auxiliary import TermLogger, AverageMeter, SaveImages
from auxiliary import worst_scores, depth_evaluations, EvalResultWriter


class TriDepthTrainer:
    def __init__(self, model, optimizer, dataloaders,
                 trainer_args={"log_root": None, "nepoch": None, "print_freq": None,
                               "img_print_freq": None, "print_progress": True},
                 device=torch.device("cuda")):
        # Load elements
        self.model = model
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.args = trainer_args
        self.device = device

        # Image saver
        self.img_saver = SaveImages(cmap_type="jet")

        # Evaluated score writers
        self.train_eval_writer = EvalResultWriter(os.path.join(self.args["log_root"], "train.csv"))
        self.val_eval_writer = EvalResultWriter(os.path.join(self.args["log_root"], "val.csv"))
        self.test_eval_writer = EvalResultWriter(os.path.join(self.args["log_root"], "test.csv"))

        # Best score preservers
        self.best_val_scores = worst_scores
        self.best_test_scores = None
        self.best_txt = os.path.join(self.args["log_root"], "best.txt")

        # Tensorboard writer
        self.tb_itr = 0
        self.tb_writer = SummaryWriter(self.args["log_root"])

    def _prepare_logger(self):
        """Prepare logger for visualization of train loop
        """
        # Logger
        self.logger = TermLogger(
            n_epochs=self.args["nepoch"],
            train_size=len(self.dataloaders["train"]),
            valid_size=max(len(self.dataloaders["val"]), len(self.dataloaders["test"])),
            use_flag=self.args["print_progress"]
        )

    def _train_epoch(self, epoch):
        # Prepare average_meter
        avg_train_loss = AverageMeter()

        # Start training
        self.model.train()
        self.logger.start_bar("train")
        end = time.time()
        for t_idx, (scenes, gt_depths, mesh_list) in enumerate(self.dataloaders["train"]):
            # Measure time of loading batched-data (data_time)
            data_time = time.time() - end
            end = time.time()

            # Load information
            batchsize = scenes.shape[0]

            # To GPU
            scenes = scenes.to(self.device)
            gt_depths = gt_depths.to(self.device)

            # Inference + loss calculation
            self.optimizer.zero_grad()
            tridepth = self.model(scenes, mesh_list)
            loss, losses = self.model.loss(tridepth, gt_depths)

            # Back-propagation
            loss.backward()
            self.optimizer.step()

            # Measure time of GPU operation (gpu_time)
            gpu_time = time.time() - end
            end = time.time()

            # Calculate evaluation scores (and average it)
            est_depths = tridepth.render_depths()
            eval_scores = depth_evaluations(est_depths.detach(), gt_depths.detach())

            # Append the parameter size of patch-cloud (num of patch)
            eval_scores["num_patch"] = np.array(tridepth.num_patch_list).mean()
            eval_scores["num_vertex"] = np.array(tridepth.num_patch_list).mean() * 3.0
            self.train_eval_writer.update(eval_scores, batchsize)

            # Tensorboard visualizer
            self._write_train_tensorboard(losses, tridepth, gt_depths, n_img=3)

            # Log message (average loss)
            avg_train_loss.update(loss.item(), batchsize)
            self.logger.update_bar(t_idx + 1, "train")
            self.logger.write_log('Train: Data {:.3f} GPU {:.3f} Loss {}'.format(
                data_time, gpu_time, avg_train_loss), "train")

            t_idx += 1

        # Write evaluation scores into csv
        self.train_eval_writer.write_avg_into_csv(epoch)

        # Final log message in this epoch
        self.logger.write_log(' * Avg TrainLoss : {:.3f}'.format(avg_train_loss.avg[0]), "train")

    def _eval_epoch(self, epoch, eval_type="val"):
        # If datasize=0, only return None instead of eval_scores
        if len(self.dataloaders[eval_type]) == 0:
            return None

        # Prepare average_meter
        avg_eval_loss = AverageMeter()
        if eval_type == "val":
            eval_writer = self.val_eval_writer
        elif eval_type == "test":
            eval_writer = self.test_eval_writer
        eval_writer.reset()
        self.img_saver.reset()

        # Start evalidation
        self.model.eval()
        self.logger.start_bar("eval")
        end = time.time()
        for e_idx, (scenes, gt_depths, mesh_list, orig_scenes, orig_gt_depths) in enumerate(self.dataloaders[eval_type]):
            # Measure time of loading batched-data (data_time)
            data_time = time.time() - end
            end = time.time()

            # To GPU
            scenes = scenes.to(self.device)
            gt_depths = gt_depths.to(self.device)
            orig_gt_depths = orig_gt_depths.to(self.device)

            # Load information
            batchsize, _, height, width = scenes.shape
            batchsize, _, orig_height, orig_width = orig_gt_depths.shape

            # Inference + loss calculation
            with torch.no_grad():
                tridepth = self.model(scenes, mesh_list)
            loss, _ = self.model.loss(tridepth, orig_gt_depths)

            # Measure time of GPU operation (gpu_time)
            gpu_time = time.time() - end
            end = time.time()

            # Calculate evaluation scores (and average it)
            orig_est_depths = tridepth.render_depths(render_size=(orig_height, orig_width))
            eval_scores = depth_evaluations(orig_est_depths.detach(), orig_gt_depths.detach())

            # Append the parameter size of patch-cloud (num of patch)
            eval_scores["num_patch"] = np.array(tridepth.num_patch_list).mean()
            eval_scores["num_vertex"] = np.array(tridepth.num_patch_list).mean() * 3.0
            eval_writer.update(eval_scores, batchsize)

            # Log message (average loss)
            avg_eval_loss.update(loss.item(), batchsize)
            self.logger.update_bar(e_idx + 1, "eval")
            self.logger.write_log('{}: Data {:.3f} GPU {:.3f} Loss {}'.format(
                eval_type.capitalize(), data_time, gpu_time, avg_eval_loss), "eval")

            # Write results into images (for test time)
            if eval_type == "test" and e_idx % (len(self.dataloaders[eval_type]) // 8) == 0:
                est_depths = F.interpolate(orig_est_depths, size=(height, width)).to(self.device)
                est_edges = F.interpolate(tridepth.base_edges, size=(height, width)).to(self.device)
                self.img_saver.update_merged_image(scenes[0], gt_depths[0], est_depths[0], est_edges[0])

        # Save result into images (for test time)
        if eval_type == "test":
            self.img_saver.save_merged_image(os.path.join(self.args["log_root"], "comparison_{}.png".format(epoch)))

        # Write evaluation scores into csv
        eval_scores = eval_writer.write_avg_into_csv(epoch)

        # Tensorboard visualizer
        self._write_eval_tensorboard(eval_type, eval_scores, epoch)

        # Final log message in this epoch
        self.logger.write_log(' * Avg {:} Loss : {:.3f}'.format(eval_type.capitalize(), avg_eval_loss.avg[0]), "eval")

        return eval_scores

    def _convert_colormap(self, depthmaps, normalize=True):
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt

        colormaps = []
        for b in range(depthmaps.shape[0]):
            np_depth = depthmaps[b, 0].detach().cpu().numpy()
            # Valid mask
            valid_mask = np_depth > 0

            # Scaling from d_min & d_max
            if normalize:
                d_min = np.min(np_depth[valid_mask])
                d_max = np.max(np_depth[valid_mask])
                depth_relative = ((np_depth - d_min) / (d_max - d_min)) * valid_mask
            else:
                depth_relative = np_depth

            # Output colored depthmap
            color_depth = plt.cm.jet(depth_relative)[:, :, :3]
            colormaps.append(torch.from_numpy(color_depth).permute(2, 0, 1))

        return torch.stack(colormaps, 0).float()

    def _write_train_tensorboard(self, loss_dic, tridepth, gt_depths, n_img=3):
        """Visualize training process using tensorboard
        """

        if self.tb_itr % self.args["print_freq"] == 0:
            for key, value in loss_dic.items():
                self.tb_writer.add_scalar('train/{}_loss'.format(key), value.item(), self.tb_itr)

        if self.tb_itr % self.args["img_print_freq"] == 0:
            gt_colordepth = self._convert_colormap(gt_depths)
            render_colordepth = self._convert_colormap(tridepth.render_depths())
            pred_edges = tridepth.base_edges.detach().cpu().float().repeat(1, 3, 1, 1)
            results = torch.cat((tridepth.base_imgs.detach().cpu().float(),
                                 gt_colordepth, render_colordepth, pred_edges), 2)
            self.tb_writer.add_image('train/results', make_grid(results[:n_img], normalize=True), self.tb_itr)

        # Update
        self.tb_itr += 1

    def _write_eval_tensorboard(self, eval_type, eval_scores, epoch):
        """Visualize evaluation scores in TensorboardX
        """
        for key, value in eval_scores.items():
            self.tb_writer.add_scalar("{}/{}".format(eval_type, key), value, epoch)

    def _preserve_best_score_from_val(self, epoch, val_scores, test_scores):
        """Remember best rmse and save its checkpoint
        """
        if val_scores is not None:
            is_best = val_scores["rmse"] < self.best_val_scores["rmse"]
        else:
            is_best = True

        if is_best:
            self.best_val_scores = val_scores
            self.best_test_scores = test_scores
            # Write into txt-file
            with open(self.best_txt, 'w') as txtfile:
                txtfile.write("epoch\t={}\n".format(epoch))
                for k, v in self.best_test_scores.items():
                    txtfile.write("{}\t={}\n".format(k, v))
            # Write into img-file
            self.img_saver.save_merged_image(os.path.join(self.args["log_root"], "comparison_best.png"))

        return is_best

    def _save_checkpoint(self, epoch, is_best):
        """Save checkpoint for each epoch in this training loop.
        """
        # Move to cpu (in order to deal with various gpu type)
        cpu_state_dict = self.model.state_dict()
        for key in cpu_state_dict.keys():
            cpu_state_dict[key] = cpu_state_dict[key].to(torch.device('cpu'))

        # Save checkpoint in this epoch
        checkpoint_filename = os.path.join(self.args["log_root"], 'checkpoint-' + str(epoch) + '.pth.tar')
        torch.save({"log_root": self.args["log_root"],  # For checking
                    "epoch": epoch,
                    "model": cpu_state_dict,
                    "optimizer": self.optimizer,
                    "best_val_scores": self.best_val_scores,
                    "best_test_scores": self.best_test_scores}, checkpoint_filename)

        # Save checkpoint with the best-val-scores
        if is_best:
            best_filename = os.path.join(self.args["log_root"], 'model_best.pth.tar')
            shutil.copyfile(checkpoint_filename, best_filename)

        # Save only newest + best checkpoint
        if epoch > 0:
            prev_checkpoint_filename = os.path.join(self.args["log_root"], 'checkpoint-' + str(epoch - 1) + '.pth.tar')
            if os.path.exists(prev_checkpoint_filename):
                os.remove(prev_checkpoint_filename)

    def load_checkpoint(self, filename):
        """Load previous checkpoint (sometimes it inculdes only the model's pretrained weight) 
        and resume its training based on it.
        Args:
            filename (str): filepath to the checkpoint ('*.pth')
        """
        if filename:
            print("=> using checkpoints from {}".format(filename))
            resume_states = torch.load(filename)
            self.model.load_state_dict(resume_states)

    def run(self):
        """Execute the training loop, composed of train/validation/test.
        """
        self._prepare_logger()
        self.logger.start_bar("epoch")
        for epoch in range(self.args["nepoch"]):
            self.logger.update_bar(epoch, "epoch")

            # Run epoch for train & validation & test
            self._train_epoch(epoch)
            val_scores = self._eval_epoch(epoch, eval_type="val")
            test_scores = self._eval_epoch(epoch, eval_type="test")

            # Save best validation results
            is_best = self._preserve_best_score_from_val(epoch, val_scores, test_scores)

            # Save checkpoint for each epoch
            self._save_checkpoint(epoch, is_best)

        self.logger.finish_bar("epoch")
