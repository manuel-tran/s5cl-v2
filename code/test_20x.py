### ꧁-----꧂ IMPORTS ꧁-----꧂ ###

import os
import sys
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchmetrics
import pytorch_lightning as pl

from tqdm.notebook import tqdm
from scipy.sparse import csr_matrix
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

sys.path.append('/home/haicu/manuel.tran/phd/models/navigator/')
sys.path.append('/home/haicu/manuel.tran/phd/models/coad/')

from navigator import Router, Navigator
from navigator import create_filter, merge_and_order

from s5cl_v2.methods import S5CL_V2, S5CL_V2_MS, HiMulCon, HiMulConMS
from s5cl_v2.utils import seed_everything, build_model
from s5cl_v2.swin_transformer import ConvStem, swin_tiny_patch4_window7_224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### ꧁-----꧂ HELPERS ꧁-----꧂ ###

class ReLabel_20:

    def __init__(self):
        """
        """

    def __call__(self, target):
        if target == 0:
            target = 1
        return target
    
    
class ReLabel_05:

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
    
    
def merge_and_order(x, y, fltr):
    if len(x.size()) == 1:
        x = x.unsqueeze(dim=0)
    if len(y.size()) == 1:
        y = y.unsqueeze(dim=0)
    i, j = 0, 0
    z = torch.cat((x, y)) 
    for k in range(len(z)):
        if fltr[k] == True:        
            z[k] = x[i]
            i += 1
        else:
            z[k] = y[j]
            j += 1         
    return z


### ꧁-----꧂ DATASET ꧁-----꧂ ###

class DataSet(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        data, target = self.dataset[index]       
        return data, target, index

hmap_dir = '/home/haicu/manuel.tran/phd/temp/segmentation_env2/heatmap/'
data_dir = '/lustre/groups/haicu/workspace/manuel.tran/GDC-TCGA-COAD-HE-20X-TEST-NEW'

mean, std = [0.7406, 0.5331, 0.7059], [0.1279, 0.1606, 0.1191]
transform = transforms.Compose(
    [transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize(mean, std)]
)

dataset = datasets.ImageFolder(root=data_dir, transform=transform, target_transform=ReLabel_20())
dataset = DataSet(dataset)

dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
print(len(dataset), len(dataloader))


### ꧁-----꧂ AVERAGE METERS ꧁-----꧂ ###

test_f1 = torchmetrics.F1Score(num_classes=16, average='macro').to(device)
test_top1 = torchmetrics.Accuracy().to(device)
test_topk = torchmetrics.Accuracy(top_k=3).to(device)


### ꧁-----꧂ MODEL CHECKPOINTS ꧁-----꧂ ###

encoder = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
encoder.head = nn.Identity()
embedder = nn.Linear(768, 384)
model = Navigator(encoder, embedder, 384, 16).cuda()

s5cl_v2_ms = S5CL_V2_MS(
    model,
    max_steps=0,
    num_cls=1,
    child_cls=None,
    parent_cls=None,
    sampler=None,
    dataset_l=dataset,
    dataset_u=dataset,
    dataset_v=dataset,
    bsz_l=1,
    bsz_u=1,
    bsz_v=1,
    temp_l=1,
    temp_u=1,
    temp_p=1,
    temp_f=1, 
    ls=0,
    thr=0,
    lr=0,
    wd=0,
)

chk_path = './epoch=0-step=3402.ckpt'
s5cl_v2_ms.load_state_dict(torch.load(chk_path, map_location='cuda:0')['state_dict'])
model = copy.deepcopy(s5cl_v2_ms.model_ema)

model.to(device)
model.eval()


### ꧁-----꧂ PREPARE ROUTER ꧁-----꧂ ###                    

hmap_dir = './heatmap/'

routers = {}
start = time.time()

for hmap_file in tqdm(os.listdir(hmap_dir)):
    downsample = int(hmap_file.split('_')[-1].split('.')[0])
    hmap_path = hmap_dir + hmap_file
    slide_id = hmap_file.split('.')[0]
    
    hmap = np.load(hmap_path)
    hmap[hmap == 0.0] = 1
    
    sparse = csr_matrix(hmap)
    del hmap
    
    routers.update({slide_id : sparse}) 
    
end = time.time()
print(end-start)


### ꧁-----꧂ NAVIGATOR INFERENCE ꧁-----꧂ ###

acc = 0
with torch.no_grad():
    for batch in tqdm(dataloader):
        image = batch[0].to(device)
        label = batch[1].to(device)
        index = batch[2].to(device)
        
       
        sample_fname = [
            dataloader.dataset.dataset.samples[i][0] for i in index
        ] 
        x_coord = [int(fname[:-4].split("_")[-2]) for fname in sample_fname]
        y_coord = [int(fname[:-4].split("_")[-1]) for fname in sample_fname]
        
        fltr = []
        for idx, fname in enumerate(sample_fname):   
            for hmap_file in os.listdir(hmap_dir):
                if fname.split('/')[-1].split('_')[0] == hmap_file.split('.')[0]:
                    hmap_path = hmap_dir + hmap_file
                    downsample = int(hmap_file.split('_')[-1].split('.')[0])
                    
                    x = int(x_coord[idx] / downsample)
                    y = int(y_coord[idx] / downsample)
                    
                    x = int(x - ((256*8) / downsample))
                    y = int(y - ((256*8) / downsample))
                    
                    psize = int((512*8) / downsample)
                    area = psize * psize
              
                    fltr.append( (routers[hmap_file.split('.')[0]][y:y+psize, x:x+psize].sum() / area) < 0.5 )
      
        fltr = torch.tensor(fltr)
        _, _, crc, nrm = model(image, fltr)  
        #_, _, crc, nrm = model.module(image, fltr)  
        logit = merge_and_order(crc, nrm, fltr)
        pred = torch.argmax(logit, dim=1)
        
        test_f1(pred, label)
        test_top1(pred, label)
        test_topk(logit, label)

f1 = test_f1.compute()
top1 = test_top1.compute()
topk = test_topk.compute()

print(f"Top-1 on all data: {top1}")
print(f"Top-3 on all data: {topk}")       
print(f"F1 on all data: {f1}") 
