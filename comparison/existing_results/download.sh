# Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture [Eigen & Fergus, ICCV2015]
# [link] https://cs.nyu.edu/~deigen/dnl/
wget -O eigen_depth.mat https://cs.nyu.edu/~deigen/dnl/predictions_depth_vgg.mat

# Deeper Depth Prediction with Fully Convolutional Residual Networks [Laina et al., 3DV2016]
# [link] https://github.com/iro-cp/FCRN-DepthPrediction
wget -O laina_depth.mat http://campar.in.tum.de/files/rupprecht/depthpred/predictions_NYUval.mat

# DORN: Deep Ordinal Regression Network for Monocular Depth Estimation [Fu et al., CVPR2018]
#   - We create this mat-file with author's code & pretrained model
# [link] https://github.com/hufu6371/DORN
wget https://www.dropbox.com/s/kh5rsr8p6vahcit/dorn_depth.mat