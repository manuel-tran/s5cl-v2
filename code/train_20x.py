import os
import sys
import copy
import random 
import numpy as np

import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl

from tqdm.notebook import tqdm
from scipy.sparse import csr_matrix
from torchvision import datasets, transforms
from pytorch_metric_learning import samplers
from torch.utils.data import Subset, Dataset, DataLoader, ConcatDataset

from s5cl_v2.methods import HiMulConMS, S5CL_V2_MS
from s5cl_v2.transforms import TransformGDC, RandomRotate90
from s5cl_v2.datasets import make_dataset
from s5cl_v2.utils import seed_everything, build_model
from s5cl_v2.swin_transformer import ConvStem, swin_tiny_patch4_window7_224

from navigator import Router, Navigator
from navigator import create_filter, merge_and_order

from clam.create_dataset import Coord2Data

#--------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


### ꧁-----꧂ ENVIRONMENT ꧁-----꧂ ###

SEED = 42
seed_everything(SEED)


### ꧁-----꧂ HELPER ꧁-----꧂ ###

class HiddenPrints:
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        

class ReLabel:

    def __init__(self):
        """
        """

    def __call__(self, target):
        if target == 0:
            target = 1
        return target
    
    
class FilterDataset(Dataset):
    
    def __init__(self, slide, dataset, hmap_dir):
        super().__init__()
        self.slide = slide
        self.dataset = dataset
        self.hmap_dir = hmap_dir
        self.hmaps = {} 

    def loadhmap(self):
        hmaps = {} 
        
        hmap_file = os.listdir(self.hmap_dir)[0]
        downsample = int(hmap_file.split('_')[-1].split('.')[0])
        hmap_path = self.hmap_dir + hmap_file
        slide_id = hmap_file.split('.')[0]  

        hmap = np.load(hmap_path)
        hmap[hmap == 0.0] = 1
        sparse = csr_matrix(hmap)
        del hmap

        hmaps.update({slide_id : sparse}) 
        
        self.hmaps = hmaps
        return hmaps
    
    def __len__(self):
        return len(self.dataset)
                         
    def __getitem__(self, idx):
        data, coords = self.dataset[idx]  
    
        f_name = self.slide
        x_coord = coords[0]
        y_coord = coords[1]
        
        hmap_file = os.listdir(self.hmap_dir)[0]
        downsample = int(hmap_file.split('_')[-1].split('.')[0])

        x = int(x_coord / downsample)
        y = int(y_coord / downsample)

        x = int(x - ((256 * 8) / downsample))
        y = int(y - ((256 * 8) / downsample))

        psize = int((512 * 8) / downsample)
        area = psize * psize

        if (self.hmaps[hmap_file.split('.')[0]][y:y+psize, x:x+psize].sum() / area) < 0.5:
            image = data
            label = 1
        else:
            image = data
            label = 0
    
        return image, label

    
### ꧁-----꧂ MODEL ꧁-----꧂ ###

#encoder = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
#encoder.head = nn.Identity()

#td = torch.load('./ctranspath.pth')
#encoder.load_state_dict(td['model'], strict=True)
#embedder = nn.Linear(768, 384)
#model = Navigator(encoder, embedder, 384, 16).cuda()

#himulcon = HiMulConMS(model=model, child_cls=None, parent_cls=None, max_steps=0, cl=1, ls=0, tp=0, lr=0, wd=0)
#chk_path = './epoch=1-step=1584.ckpt'
#himulcon.load_state_dict(torch.load(chk_path, map_location='cuda:0')['state_dict'])
#model = copy.deepcopy(himulcon.model)

encoder = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
encoder.classifier = nn.Identity()
embedder = nn.Linear(1280, 640)
model = Navigator(encoder, embedder, 640, 16).cuda()

himulcon = HiMulConMS(model=model, child_cls=None, parent_cls=None, max_steps=0, cl=1, ls=0, tp=0, lr=0, wd=0)
chk_path = './epoch=1-step=3168.ckpt'
himulcon.load_state_dict(torch.load(chk_path, map_location='cuda:0')['state_dict'])
model = copy.deepcopy(himulcon.model)


### ꧁-----꧂ LABELED DATASET ꧁-----꧂ ###

mean, std = [0.7406, 0.5331, 0.7059], [0.1279, 0.1606, 0.1191]

transform_t = TransformGDC(mean, std)
transform_v = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize(mean, std)])

dataset_l = datasets.ImageFolder(root="./GDC-TCGA-COAD-HE-20X-TRAIN-NEW", transform=transform_t, target_transform=ReLabel())
dataset_v = datasets.ImageFolder(root="./GDC-TCGA-COAD-HE-20X-TEST-NEW", transform=transform_v, target_transform=ReLabel())

sampler = samplers.MPerClassSampler(dataset_l.targets, m=2, length_before_new_iter=len(dataset_l))

loader_l = DataLoader(dataset_l, batch_size=16, sampler=sampler, num_workers=4, pin_memory=True)
loader_v = DataLoader(dataset_v, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

print(len(dataset_l), len(dataset_v), len(loader_l), len(loader_v))


### ꧁-----꧂ UNLABELED DATASET ꧁-----꧂ ###

slide_dir = './slides_with_annotations/'
coord_dir = './clam/20x/train/patch/'
save_dir = './clam/20x/train/save/'

exclude = [
    'TCGA-AA-3562-01Z-00-DX1.e07893e6-646d-41b5-be51-9c19d51f6743.h5',
    'TCGA-AA-3697-01Z-00-DX1.AAB8DB74-F76D-4D0A-A50E-E7F97504A3C4.h5',
    'TCGA-AA-A02K-01Z-00-DX1.732DD8F9-A21A-4E97-A779-3400A6C3D19D.h5',
    
    'TCGA-A6-6142-01Z-00-DX1.e923ce20-d3c3-4d21-9e7c-d999a3742f9b.h5',
    'TCGA-AZ-6601-01Z-00-DX1.40681471-3104-48be-8b57-55dba1f432f8.h5',
    'TCGA-CK-4951-01Z-00-DX1.abdbb15c-fd40-4a55-bf54-5668b3d4ea13.h5',
    'TCGA-CM-5861-01Z-00-DX1.b900abc0-ecca-48e1-98ba-fbc99a6dae3e.h5',
    'TCGA-CM-6163-01Z-00-DX1.012a7433-73bb-4584-957b-f92c8877a114.h5',
    'TCGA-CM-6164-01Z-00-DX1.ccf5ce96-b732-4c35-b177-d3dbe2ed89cb.h5',
    'TCGA-CM-6166-01Z-00-DX1.52eaa124-7ab5-4aaf-b074-7f89a4c53804.h5',
    'TCGA-CM-6170-01Z-00-DX1.aa9c41ea-3894-4524-a94c-f44c6c53c2d0.h5'
]

exclude = [fname[:-3] for fname in exclude]

mean, std = [0.7406, 0.5331, 0.7059], [0.1279, 0.1606, 0.1191]

transform_t = TransformGDC(mean, std)
transform_v = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)])
        
with HiddenPrints():
    dataset_clam = Coord2Data(
        slide_dir,
        coord_dir,
        transforms=transform_t,
        exclude=None,
    ).get_dataset()
    
dataset_u = []
hmap_dir = './segmentation/heatmap/'

for i in tqdm(range(len(dataset_clam))):
    fname = list(dataset_clam.keys())[i]
    subset = list(dataset_clam.values())[i]
    
    if fname not in exclude:
        subset = FilterDataset(fname, subset, hmap_dir)
        _ = subset.loadhmap()
        dataset_u.append(subset)

dataset_u = ConcatDataset(dataset_u)
loader_u = DataLoader(dataset_u, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

print(len(dataset_clam), len(dataset_u), len(loader_u))


### ꧁-----꧂ HIERARCHICAL MULTILABLES ꧁-----꧂ ###

child_class_to_idx = dataset_l.class_to_idx
#child_class_to_idx = dataset.class_to_idx

child_class_to_idx = {
    'HIGH': 0,
    'LOW' : 1,
    'BLO' : 2,
    'GLA' : 3,
    'SOL' : 4,
    'DEB' : 5,
    'FAT' : 6,
    'LYM' : 7,
    'MUC' : 8,
    'NOR' : 9,
    'SMO' : 10,
    'SKE' : 11,
    'STR' : 12,
    'SUB' : 13,
    'TIL' : 14,
    'VES' : 15,
}

parent_class_to_idx = {
    'HIGH': 0,
    'LOW' : 0,
    'BLO' : 0,
    'GLA' : 1,
    'SOL' : 1,
    'DEB' : 1,
    'FAT' : 0,
    'LYM' : 0,
    'MUC' : 1,
    'NOR' : 0,
    'SMO' : 0,
    'SKE' : 0,
    'STR' : 1,
    'SUB' : 0,
    'TIL' : 1,
    'VES' : 1,
}
        
child_cls = list(child_class_to_idx.values())
parent_cls = list(parent_class_to_idx.values())


### ꧁-----꧂ TRAINING ꧁-----꧂ ###

N_EPOCH = 1
G_CLIP = 1.0
num_cls = 16
max_steps = max(len(loader_l), len(loader_u)) * N_EPOCH

s5cl_v2_ms = S5CL_V2_MS(
    model,
    max_steps,
    num_cls,
    child_cls,
    parent_cls,
    sampler,
    dataset_l,
    dataset_u,
    dataset_v,
    bsz_l=16*1,
    bsz_u=16*1,
    bsz_v=16*1,
    temp_l=0.1,
    temp_u=0.8,
    temp_p=0.1,
    temp_f=0.5, 
    ls=0.1,
    thr=0.80,
    lr=0.03,
    wd=0.0001,
)

trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, max_epochs=N_EPOCH, gradient_clip_val=G_CLIP)
trainer.fit(s5cl_v2_ms)
