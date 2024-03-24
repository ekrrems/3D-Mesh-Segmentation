# 3D Mesh Segmentation
![3D Meshes](https://github.com/ekrrems/3D-Mesh-Segmentation/blob/main/dataset/MPI-FAUST/Capture.PNG)

## Overwiev
Segmentation of 3D meshes is a crucial task in computer vision and graphics, with applications in various domains such as medical imaging, robotics, and computer-aided design (CAD). This project explores Feature-Steered deep learning architectures and techniques for accurately segmenting 3D meshes into semantic parts.

## Features
- Data Preprocessing: Includes utilities for loading 3D mesh data and preprocessing it for training.
- Model Architecture: Provides different deep learning models tailored for 3D mesh segmentation.
- Training Pipeline: Implements training routines for training the segmentation models.
- Evaluation: Includes evaluation metrics and visualization tools for assessing model performance.
- Inference: Supports inference on new 3D meshes for segmentation tasks.
- Visualization: Visualize the segmented 3D Meshes

  ## Installation
  ```
  git clone https://github.com/ekrrems/3D-Mesh-Segmentation
  ```
  ## Usage
  ```
  cd 3D-Mesh-Segmentation/src
  python train.py
  python visualize.py 
  ```

  ## Licence
  This project is licensed under the [MIT License]().

  ## Reference
  - [FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis](https://openaccess.thecvf.com/content_cvpr_2018/papers/Verma_FeaStNet_Feature-Steered_Graph_CVPR_2018_paper.pdf)
