import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(encoder, enc_dim, emb_dim, cls_dim, dropout=0.2):

    embedder = nn.Linear(in_features=enc_dim, out_features=emb_dim, bias=True)

    classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=False),
        nn.Linear(in_features=emb_dim, out_features=cls_dim, bias=True),
    )

    model = nn.Sequential(encoder, embedder, classifier)

    return model


def multilabel(old_labels, old_cls, new_cls):
    new_labels = torch.clone(old_labels)

    for i in range(len(old_cls)):
        new_labels[new_labels == i] = new_cls[i]

    old_labels = torch.unsqueeze(old_labels, 1)
    new_labels = torch.unsqueeze(new_labels, 1)

    multi_labels = torch.cat((new_labels, old_labels), dim=1)

    return multi_labels


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
