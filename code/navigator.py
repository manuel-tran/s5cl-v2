import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import rasterio.features
import shapely.geometry

from shapely.geometry import MultiPoint


class Navigator(nn.Module):
    def __init__(self, encoder, embedder, emb_dim, cls_dim):
        super().__init__()
        self.enc = encoder
        self.emb = embedder
        self.crc = nn.Linear(emb_dim, cls_dim)
        self.nrm = nn.Linear(emb_dim, cls_dim)

    def forward(self, x, f):
        e = self.enc(x)
        z = self.emb(e)
        c = self.crc(z[np.argwhere(np.asarray(f)).squeeze()])
        n = self.nrm(z[np.argwhere(np.asarray(~f)).squeeze()])
        return (e, z, c, n)


class Router:
    def __init__(self, hmap, thrs, psize):
        self.hmap = hmap
        self.thrs = thrs
        self.psize = psize
        self.roe = None

    def segment(self):
        self.hmap[self.hmap > self.thrs] = 1
        self.hmap = self.hmap.astype(np.uint8)
        self.hmap = rasterio.features.shapes(self.hmap)

        self.roe = [
            shapely.geometry.Polygon(poly[0]["coordinates"][0])
            for poly in self.hmap if poly[1] == 1
        ]
        return self.roe

    def classify(self, x, y):
        lst = []

        patch = MultiPoint(
            [
                [x - self.psize, y - self.psize],
                [x - 0, y - self.psize],
                [x - self.psize, y - 0],
                [x - 0, y - 0],
            ]
        ).convex_hull

        for i in range(len(self.roe)):
            lst.append(self.roe[i].intersects(patch))

        return any(lst)


def create_filter(dataloader, batch, router):
    fltr = []
    for index in batch[2]:
        sample_fname, _ = dataloader.dataset.dataset.samples[index]
        x = int(sample_fname[:-4].split("_")[-2])
        y = int(sample_fname[:-4].split("_")[-1])
        fltr.append(router.classify(x, y))
    return fltr


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


def deploy_model(model, dataloader, n_classes, n_samples, device):
    model.eval()
    softmax = nn.Softmax(dim=1)

    preds = []
    probs = []

    for (image, label) in dataloader:
        image = image.to(device)
        image = image.squeeze()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = model(image)
        preds.append(torch.max(logits, 1)[1])
        probs.append(softmax(logits))

    preds = torch.cat(preds, 0)
    probs = torch.cat(probs, 0)

    return preds, probs


def calculate_scores(preds, probs):
    tgt_cls = 1
    scores = torch.clone(preds)
    scores = scores.type(torch.DoubleTensor)

    for i, pred in enumerate(preds):
        if pred == tgt_cls:
            scores[i] = probs[i][tgt_cls]
            scores[i] = 50 - 50 * scores[i]
        else:
            scores[i] = 1 - probs[i][tgt_cls]
            scores[i] = 50 + 50 * scores[i]

    scores = scores.cpu().numpy()
    scores = list([[x] for x in scores])

    return scores


def get_coordinates(dataloader):
    coords = []
    for (_, coord) in dataloader:
        coords.append(coord)
    coords = torch.cat(coords, 0)
    coords = coords.cpu().numpy()
    return coords
