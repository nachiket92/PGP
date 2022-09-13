[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multimodal-trajectory-prediction-conditioned/trajectory-prediction-on-nuscenes)](https://paperswithcode.com/sota/trajectory-prediction-on-nuscenes?p=multimodal-trajectory-prediction-conditioned)

# Multimodal Trajectory Prediction Conditioned on Lane-Graph Traversals

![](https://github.com/nachiket92/PGP/blob/main/assets/intro.gif)




This repository contains code for ["Multimodal trajectory prediction conditioned on lane-graph traversals"](https://proceedings.mlr.press/v164/deo22a.html) by Nachiket Deo, Eric M. Wolff and Oscar Beijbom, presented at CoRL 2021.  

```bibtex
@inproceedings{deo2021multimodal,
  title={Multimodal Trajectory Prediction Conditioned on Lane-Graph Traversals},
  author={Deo, Nachiket and Wolff, Eric and Beijbom, Oscar},
  booktitle={5th Annual Conference on Robot Learning},
  year={2021}
}
```


**Note:** While I'm one of the authors of the paper, this is an independent re-implementation of the original code developed during an internship at Motional. The code follows the implementation details in the paper. Hope this helps!
 -Nachiket   


## Installation

1. Clone this repository 

2. Set up a new conda environment 
``` shell
conda create --name pgp python=3.7
```

3. Install dependencies
```shell
conda activate pgp

# nuScenes devkit
pip install nuscenes-devkit

# Pytorch: The code has been tested with Pytorch 1.7.1, CUDA 10.1, but should work with newer versions
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

# Additional utilities
pip install ray
pip install psutil
pip install positional-encodings==5.0.0
pip install imageio
pip install tensorboard
```


## Dataset

1. Download the [nuScenes dataset](https://www.nuscenes.org/download). For this project we just need the following.
    - Metadata for the Trainval split (v1.0)
    - Map expansion pack (v1.3)

2. Organize the nuScenes root directory as follows
```plain
└── nuScenes/
    ├── maps/
    |   ├── basemaps/
    |   ├── expansion/
    |   ├── prediction/
    |   ├── 36092f0b03a857c6a3403e25b4b7aab3.png
    |   ├── 37819e65e09e5547b8a3ceaefba56bb2.png
    |   ├── 53992ee3023e5494b90c316c183be829.png
    |   └── 93406b464a165eaba6d9de76ca09f5da.png
    └── v1.0-trainval
        ├── attribute.json
        ├── calibrated_sensor.json
        ...
        └── visibility.json         
```

3. Run the following script to extract pre-processed data. This speeds up training significantly.
```shell
python preprocess.py -c configs/preprocess_nuscenes.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data
```


## Inference

You can download the trained model weights using [this link](https://drive.google.com/file/d/1lHwC6I6VRLT-BLs9gRGu_xMaIupMlbtS/view?usp=sharing).

To evaluate on the nuScenes val set run the following script. This will generate a text file with evaluation metrics at the specified output directory. The results should match the [benchmark entry](https://eval.ai/web/challenges/challenge-page/591/leaderboard/1659) on Eval.ai. 
```shell
python evaluate.py -c configs/pgp_gatx2_lvm_traversal.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data -o path/to/output/directory -w path/to/trained/weights
```

To visualize predictions run the following script. This will generate gifs for a set of instance tokens (track ids) from nuScenes val at the specified output directory.  
```shell
python visualize.py -c configs/pgp_gatx2_lvm_traversal.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data -o path/to/output/directory -w path/to/trained/weights
```


## Training

To train the model from scratch, run
```shell
python train.py -c configs/pgp_gatx2_lvm_traversal.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data -o path/to/output/directory -n 100
```

The training script will save training checkpoints and tensorboard logs in the output directory.

To launch tensorboard, run
```shell
tensorboard --logdir=path/to/output/directory/tensorboard_logs
```
