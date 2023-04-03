import os
import sys
import cv2
import copy
import random
import numpy as np

import json
import time
from pathlib import Path

import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl

from sklearn.metrics import f1_score
from torchvision import datasets, transforms
from pytorch_metric_learning import samplers
from torch.utils.data import Subset, Dataset, DataLoader, ConcatDataset

from s5cl_v2.coad import COAD
from s5cl_v2.methods import S5CL_V2, HiMulCon
from s5cl_v2.transforms import TransformGDC
from s5cl_v2.datasets import DatasetFromSubset
from s5cl_v2.utils import seed_everything, build_model
from s5cl_v2.swin_transformer import ConvStem, swin_tiny_patch4_window7_224

from clam.create_dataset import Coord2Data

#--------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


### ꧁-----꧂ ENVIRONMENT ꧁-----꧂ ###

#SEED = 0
#seed_everything(SEED)


### ꧁-----꧂ HELPER ꧁-----꧂ ###

class ReLabel:

    def __init__(self):
        """
        """

    def __call__(self, target):
        if target == 0:
            target = 0
        if target == 1:
            target = 0
        if target == 2:
            target = 0
        if target == 3:
            target = 1
        if target == 4:
            target = 1
        if target == 5:
            target = 1
        if target == 6:
            target = 0  
        if target == 7:
            target = 0
        if target == 8:
            target = 1
        if target == 9:
            target = 0 
        if target == 10:
            target = 0  
        if target == 11:
            target = 0 
        if target == 12:
            target = 1 
        if target == 13:
            target = 0
        if target == 14:
            target = 1
        if target == 15:
            target = 1
        return target
    
    
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


### ꧁-----꧂ MODEL ꧁-----꧂ ###

encoder = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
encoder.head = nn.Identity()
td = torch.load('./ctranspath.pth')
encoder.load_state_dict(td['model'], strict=True)
model = build_model(encoder, 768, 384, 2)


### ꧁-----꧂ DATASET ꧁-----꧂ ###

mean, std = [0.7406, 0.5331, 0.7059], [0.1279, 0.1606, 0.1191]

transform_t = TransformGDC(mean, std)
transform_v = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)]
)

root_l = "./GDC-TCGA-COAD-HE-05X-TRAIN-20X"
root_u = "./GDC_TCGA_COAD/images/512pxTiles/"
root_v = "./GDC-TCGA-COAD-HE-05X-TEST-20X-NEW"
root_e = "./slides_with_annotations/"

slide_dir_40x = './tcga_crc_slides/Sorted/40x/'
slide_dir_20x = './tcga_crc_slides/Sorted/20x/'
coord_dir_from_40x = './05x/from_40x/patch/'
coord_dir_from_20x = './05x/from_20x/patch/'

with HiddenPrints():
    
    dataset_clam_from_40x = Coord2Data(
        slide_dir_40x,
        coord_dir_from_40x,
        transforms=transform_t,
        exclude=os.listdir(root_e),
    ).get_dataset()

    dataset_clam_from_20x = Coord2Data(
        slide_dir_20x,
        coord_dir_from_20x,
        transforms=transform_t,
        exclude=os.listdir(root_e),
    ).get_dataset()

print(len(dataset_clam_from_40x), len(dataset_clam_from_20x))

dataset_u_from_40x = ConcatDataset(list(dataset_clam_from_40x.values()))
dataset_u_from_20x = ConcatDataset(list(dataset_clam_from_20x.values()))
dataset_u = ConcatDataset((dataset_u_from_40x, dataset_u_from_20x))

dataset_l = datasets.ImageFolder(root=root_l, transform=transform_t, target_transform=ReLabel())
#dataset_u = COAD(data_path=root_u, exclude=os.listdir(root_e), mode='5.0', limit=float("inf"), transform=transform_t)
#dataset_u = COAD(data_path=root_u, exclude=os.listdir(root_e), mode='5.0', limit=2, transform=transform_t)
dataset_v = datasets.ImageFolder(root=root_v, transform=transform_v, target_transform=ReLabel())

sampler = samplers.MPerClassSampler(dataset_l.targets, m=2, length_before_new_iter=len(dataset_l))

loader_l = DataLoader(dataset_l, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)
loader_u = DataLoader(dataset_u, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
loader_v = DataLoader(dataset_v, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

print(len(dataset_l), len(dataset_u), len(dataset_v)) # 101162 117957 24411
print(len(loader_l), len(loader_u), len(loader_v)) # 3161 3687 763


### ꧁-----꧂ TRAINING ꧁-----꧂ ###

N_EPOCH = 5
G_CLIP = 1.0

num_cls = 2
child_cls, parent_cls = None, None
max_steps = max(len(loader_l), len(loader_u)) * N_EPOCH

s5cl_v2 = S5CL_V2(
    model,
    max_steps,
    num_cls,
    child_cls,
    parent_cls,
    sampler,
    dataset_l,
    dataset_u,
    dataset_v,
    bsz_l=32*2,
    bsz_u=32*2,
    bsz_v=32*2,
    temp_l=0.1,
    temp_u=0.8,
    temp_p=0.1,
    temp_f=1.0, 
    ls=0.1,
    thr=0.95,
    lr=0.0001,
    wd=0.00001,
)

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1, 
    num_nodes=1, 
    precision=16, 
    max_epochs=N_EPOCH, 
    gradient_clip_val=G_CLIP
)

trainer.fit(s5cl_v2)
