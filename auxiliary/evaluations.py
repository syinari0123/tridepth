import csv
import math
import numpy as np
import torch

from auxiliary import AverageMeter

worst_scores = {
    "mse": np.inf, "rmse": np.inf, "mae": np.inf,
    "lg10": np.inf, "absrel": np.inf,
    "irmse": np.inf, "imae": np.inf,
    "delta1": 0., "delta2": 0., "dealta3": 0.
}


class EvalResultWriter:
    def __init__(self, csv_filename):
        self.fieldnames = ['epoch', 'mse', 'rmse', 'mae', 'lg10', 'absrel',
                           'irmse', 'imae', 'delta1', 'delta2', 'delta3', 'num_patch', 'num_vertex']
        # Prepare csv file
        self.csv_filename = csv_filename
        self._prepare_csv_file()

        # Average calculator (using AvgMeter())
        self.avg_calc_dic = {}
        for fname in self.fieldnames:
            self.avg_calc_dic[fname] = AverageMeter()

    def _prepare_csv_file(self):
        """Prepare csv file writing csv header
        """
        # Write fieldnames into csv header
        with open(self.csv_filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

    def write_avg_into_csv(self, epoch):
        """Calculate avg scores and write them into csv-file
        Return:
            final_avg_results: Calculated average scores
        """
        # Calculate average score from AvgMeter()
        final_avg_results = {}
        for k, v in self.avg_calc_dic.items():
            final_avg_results[k] = v.avg[0]
        final_avg_results["epoch"] = epoch

        # Write into csv file
        with open(self.csv_filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(final_avg_results)

        # Reset all the averages for the next step
        self.reset()

        # Remove 'epoch' elements
        final_avg_results.pop("epoch")

        return final_avg_results

    def reset(self):
        """Reset all the scores in self.avg_calc_dic
        """
        for fname in self.fieldnames:
            self.avg_calc_dic[fname].reset()

    def update(self, result_dic, batchsize):
        """Add new results into self.avg_calc_dic
        """
        # Update each AvgMeter() in avg_calc_dic
        for k, v in result_dic.items():
            self.avg_calc_dic[k].update(v, batchsize)


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)


def depth_evaluations(est_depth, gt_depth):
    """Depthmap evaluation on general metrics
    """
    # Prepare dictionary
    result_dic = {}

    # Choose valid pixel in depthmap
    valid_mask = gt_depth > 0
    est_depth = est_depth[valid_mask]
    gt_depth = gt_depth[valid_mask]

    # Error based metrics
    abs_diff = (est_depth - gt_depth).abs()
    result_dic["mse"] = float((torch.pow(abs_diff, 2)).mean())
    result_dic["rmse"] = math.sqrt(result_dic["mse"])
    result_dic["mae"] = float(abs_diff.mean())
    result_dic["lg10"] = float((log10(est_depth) - log10(gt_depth)).abs().mean())
    result_dic["absrel"] = float((abs_diff / gt_depth).mean())

    # Ratio based metrics
    maxRatio = torch.max(est_depth / gt_depth, gt_depth / est_depth)
    result_dic["delta1"] = float((maxRatio < 1.25).float().mean())
    result_dic["delta2"] = float((maxRatio < 1.25 ** 2).float().mean())
    result_dic["delta3"] = float((maxRatio < 1.25 ** 3).float().mean())

    # Error on inverse depthmap
    inv_est_depth = 1 / est_depth
    inv_gt_depth = 1 / gt_depth
    abs_inv_diff = (inv_est_depth - inv_gt_depth).abs()
    result_dic["irmse"] = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
    result_dic["imae"] = float(abs_inv_diff.mean())

    return result_dic
