﻿# SAIND-Scene-Animation-using-RGB-Images-aNd-Depth-maps

## :wrench: Installation
1. Clone this repo:
    ```bash
    git clone https://github.com/ViBa-cloud/SAIND-Scene-Animation-using-RGB-Images-aNd-Depth-maps.git
    ```
2. Install dependent packages:
  After installing [Anaconda](https://www.anaconda.com/), create a new Conda environment using `conda create --name saind_env --file requirements.txt`.

## :zap: Project Architecture

```
├── LDC
    ├── checkpoints
        ├──MDBD/5/5_model.pth
    ├── data                        # Paste your Images here for running edge detection
    |   ├── (Some images)           # Images to test LDC
    ├── result/MDBD2CLASSIC         # Edge maps will be generated here
    ├── utils                       # A series of tools used in this repo
    |   └── img_processing.py       # Miscellaneous tool functions
    ├── datasets.py                 # Tools for dataset managing 
    ├── losses2.py                  # Loss function used to train DexiNed (BDCNloss2)
    ├── main.py                     # The main python file with main functions and parameter settings
                                    # here you can test and train
    ├── modelB4.py                  # LDC (4 blocks) class in pytorch
    └── modelB5.py                  # LDC (5 blocks) class in pytorch

├── TokenFusion
    ├── image2image_translation
        ├── data/scenimefy_data     # This is where rgb & edge maps will be placed for depth generation
            ├── rgb                 # Place your rgb images here
            ├── edge_occlusion      # Place your corresponding edge maps here
            ├── resize.py           # To resize the rgb and edge_occlusion maps since TokenFusion requires 512x512 images
            ├── train_domain.txt    # This is where path to the rgb/edge maps is specified (if you want to train TokenFusion, we don't)
            ├── val_domain.txt      # This is where we specify path to the rgb/edge maps is specified to run inference
        ├── models
            ├── model_cfg.py        # Functions for the generating the depth maps
            ├── model_pruning.py    # Functions for Transformer 
            ├── modules.py          # Implementation of Token exchange mechanism
        ├── results                 # This is where the depth maps will be generated
        ├── utils                   # Has a series of tools used in the TokenFusion
        ├── cfg.py                  # configuration file
        ├── eval.py                 # To do inference and generate the depth map
        ├── main.py                 # To do training for Token Fusion (we don't)
├──  Scenimefy
    ├── Anime_dataset               # Place the shinkai style movies (if training) or normal movies (if evaluating) here
        ├── frame_extract.py        # Run this file to extract rgb frames from the movies (this will be the main input of our pipeline)
    ├── Semi_translation            
        ├── datasets
            ├── Sample/testA        # Place your input rgb images here     
        ├── pretrained_models        
            ├── shinkai-test        # Shinkai_pretrained weights
        ├── results/shinkai-test
            ├──test_Shinkai/images  # Stylyzed images generated by scenimefy goes here
        ├── train.py                # To train the scenimefy model
        ├── test.py                 # To run inference over the trained scenimefy model


```
## :zap: Dataset preparation
Download the rgb images from "https://drive.google.com/drive/folders/1PWRilXeL5OcNL3i1yMQ1suamnQou_LZ5?usp=drive_link" and place it in the LDC/data directory, if you want to generate edge maps for these rgb images

Or 

If you want direct access to the edge maps, download them from "https://drive.google.com/drive/folders/17KPK3ltYRA1i3JVTMharsr7GhO0kzDNw?usp=drive_link"

## :zap: Our pipeline
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/our_pipeline.jpg">
</div>

## :zap: Proposed EnGD algorithm
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/engd.jpg">
</div>

## :zap: Preprocessing

### Generate Edge maps
 ```bash
    cd LDC
    python3 main.py --choose_test_data=-1
```

### Generate Depth maps
 ```bash
    cd ..
    cd TokenFusion/image2image_translation
    wget https://drive.google.com/file/d/1M3gUqWuXBKt5AYldQLPkTYSZCQsyB1dt/view?usp=drive_link
    python3 eval.py --gpu 0 -c <path to pwd>
```

### Generate the Fused RGB-Depth images using our proposed EnGD algorithm
 ```bash
    cd ../..
    python3 EnGD.py
```

## :zap: Training 
 ```bash
    cd Scenimefy
    python3 train.py --name exp_shinkai  --CUT_mode CUT --model semi_cut \ 
    --dataroot ./datasets/unpaired_s2a --paired_dataroot ./datasets/pair_s2a \ 
    --checkpoints_dir ./pretrained_models \
    --dce_idt --lambda_VGG -1  --lambda_NCE_s 0.05 \ 
    --use_curriculum  --gpu_ids 0
```

## :zap: Inference
 ```bash
    cd Scenimefy
    python3 test.py --dataroot ./datasets/Sample --name shinkai-test --CUT_mode CUT  --model cut --phase test --epoch Shinkai --preprocess none
```

## :zap: Results
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/results.png">
</div>
