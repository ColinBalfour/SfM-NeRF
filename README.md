# Complete Structure-From-Motion (SfM) pipeline and Neural Radiance Field (NeRF) view syntehsis!

SfM pipeline takes input images, uses triangulation and epipolar math to compute initial estimates of pose, and refines using PnP. Final poses are then further optimized using bundle adjustment.

NeRF models can be trained given a set of input images and poses, such that an image can be rendered from any view. 
NeRF model was custom-built from scratch based on the [original NeRF paper](https://arxiv.org/pdf/2003.08934) 
