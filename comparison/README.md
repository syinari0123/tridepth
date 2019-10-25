# Comparison with existing representations

* Point-cloud representation (Deep depth prediction [[Eigen+,CVPR'15]](https://cs.nyu.edu/~deigen/dnl/) [[Laina+,3DV'16]](https://github.com/iro-cp/FCRN-DepthPrediction), [[Fu+,CVPR'18]](https://github.com/hufu6371/DORN))
* Mesh representation (Dense-mesh from deep depth prediction + [Quadric-mesh-simplification](https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification))

## Experimental results
| name | mse(↓) | rmse(↓) | mae(↓) | lg10(↓) | absrel(↓) | irmse(↓) | imae(↓) | delta1(↑) | delta2(↑) | delta3(↑) | num_patch | num_vertex |
|------|:---:|:----:|:---:|:----:|:------:|:-----:|:----:|:------:|:------:|:------:|:---------:|:---------:|
| [[Eigen+,ICCV'15]](https://cs.nyu.edu/~deigen/dnl/) (pointcloud) | 0.45118 | 0.57809 | 0.43468 | 0.070277 | 0.16350 | 0.10284 | 0.07548 | 0.75332 | 0.94444 | 0.98655 | N/A | 307K |
| [[Laina+,3DV'16]](https://github.com/iro-cp/FCRN-DepthPrediction) (pointcloud) | **0.35982** | 0.51124 | **0.36675** | 0.059723 | 0.13683 | 0.09333 | 0.065164 | 0.81487 | 0.95378 | 0.98812 | N/A | 307K |
| [[Fu+,CVPR'18]](https://github.com/hufu6371/DORN) (pointcloud)  | 0.39892 | 0.54591 | 0.35656 | **0.058634** | **0.12873** | 0.10073 | 0.064595 | **0.82581** | 0.94313 | 0.97758 | N/A | 307K
| [Eigen+,ICCV'15] (mesh, face) | 0.45856 | 0.58385 | 0.43819 | 0.070844 | 0.16523 | 0.10381 | 0.076031 | 0.75071 | 0.94290 | 0.98592 | 10K| 5K |
| [Laina+,3DV'16] (mesh, face) | 0.40480 | 0.53655 | 0.38105 | 0.061730 | 0.14397 | 0.09623 | 0.066949 | 0.80766 | 0.94901 | 0.98559 | 10K | 5K |
| [Fu+,CVPR'18] (mesh, face)  | 0.46615 | 0.59881 | 0.39206 | 0.064074 | 0.14518 | 0.10774 | 0.069776 | 0.80601 | 0.93508 | 0.97300 | 18K | 9K |
| [Eigen+,ICCV'15] (mesh, vertex) | 0.45209 | 0.57917 | 0.43567 | 0.070491 | 0.16398 | 0.10336 | 0.075749 | 0.75210 | 0.94404 | 0.98643 | 60K | 30K |
| [Laina+,3DV'16] (mesh, vertex) | 0.36093 | 0.51257 | 0.36779 | 0.059949 | 0.13734 | 0.09395 | 0.065460 | 0.81368 | 0.95336 | 0.98798 | 60K | 30K |
| [Fu+,CVPR'18] (mesh, vertex) | 0.39901 | 0.54632 | 0.35754 | 0.058840 | 0.12920 | 0.10117 | 0.064863 | 0.82494 | 0.94293 | 0.97766 | 60K | 30K |
| **Ours (triangular-patch-cloud)** | 0.37777 | **0.51016** | 0.37089 | 0.059513 | 0.14113 | **0.09047** | **0.064466** | 0.81781 | **0.95770** | **0.98933** | 10K | 30K | 90K |


## Run evaluation for each representation

You need to download all the results of NYU Depth v2 dataset (eigen, laina, fu).
```
cd existing_results
./download.sh
```

After downloading results, you can run evaluation code for each method.
```
python main.py \
    --data-path $DATA_DIR \
    --method-name "eigen" \     # Choose from ["eigen", "laina", "dorn"]
    --simplify-mode "vertex" \  # Choose from ["vertex", "face"]
    --result-path "result" \
    --print-progress "true"
```
DATA_DIR is the dataset path (Default: ../data/nyudepthv2).