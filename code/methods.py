import torch
import torchvision
import torchmetrics
import pytorch_lightning as pl

from copy import deepcopy
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy

from s5cl_v2.utils import multilabel, merge_and_order
from s5cl_v2.losses import CrossEntropy, HMLC

#-----------------------------------------------------------------------------------

class EMA(nn.Module):
    """ 
    Model Exponential Moving Average V2 from timm
    """
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

#-----------------------------------------------------------------------------------

class S5CL_V2(pl.LightningModule):
    """ 
    ꧁-----꧂ S5CL v2  ꧁-----꧂

    Abbreviations:
        x ... inputs
        y ... labels
        t ... targets
        w ... weak augmentation
        s ... strong augmentation
        l ... labeled
        u ... unlabeled
        p ... pseudo-labeled
        e ... encodings
        z ... embeddings
        ỹ ... logits
        bsz ... batch-size

    Args:
        model: of the form nn.Sequential(encoder, embedder, classifier)
        max_steps: total number of iterations for learning rate scheduler
        num_cls: number of classes in the leaf nodes
        child_cls: list of all class indices for all subcategories
        parent_cls: list of all class indices for all categories (can be None)
        sampler: sample classes from the labeled dataset
        dataset_l: labeled dataset
        dataset_u: unlabeled dataset
        dataset_v: validation dataset
        bsz_l: batch size for the labeled dataset
        bsz_u: batch size for the unlabeled dataset
        bsz_v: batch size for the validation dataset
        temp_l: fully-supervised temperature
        temp_u: self-supervised temperature
        temp_p: semi-supervised temperature
        temp_f: FixMatch pseudo-label temperature
        ls: label-smoothing in cross entropy
        thr: threshold for pseudo-label predictions
        lr: learning rate
        wd: weight decay

    Return:
        model and model_ema with training logs and checkpoints
    """

    def __init__(
        self,
        model,
        max_steps,
        num_cls, 
        child_cls,
        parent_cls,
        sampler,
        dataset_l,
        dataset_u,
        dataset_v,
        bsz_l,
        bsz_u,
        bsz_v,
        temp_l=0.1,
        temp_u=0.8,
        temp_p=0.1,
        temp_f=0.5, 
        ls=0.00,
        thr=0.80,
        lr=0.03,
        wd=0.0001,
    ):
        super().__init__()
        self.model = model
        self.model_ema = EMA(self.model, decay=0.999)
        #self.save_hyperparameters(ignore=["model"])

        self.child_cls = child_cls
        self.parent_cls = parent_cls
        
        self.max_steps = max_steps
        self.sampler = sampler
        
        self.dataset_l = dataset_l
        self.dataset_u = dataset_u
        self.dataset_v = dataset_v
        
        self.bsz_l = bsz_l
        self.bsz_u = bsz_u
        self.bsz_v = bsz_v
        
        self.temp_f = temp_f
        self.thr = thr
        
        self.lr = lr
        self.wd = wd

        self.criterion_l = HMLC(temperature=temp_l)
        self.criterion_u = HMLC(temperature=temp_u)
        self.criterion_p = HMLC(temperature=temp_p)
        self.criterion_c = CrossEntropy(label_smoothing=ls)
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc  = torchmetrics.Accuracy() 
        
        self.train_topk = torchmetrics.Accuracy(top_k=3)
        self.valid_topk = torchmetrics.Accuracy(top_k=3)
        self.test_topk  = torchmetrics.Accuracy(top_k=3)
        
        self.train_f1 = torchmetrics.F1Score(num_classes=num_cls, average='macro')
        self.valid_f1 = torchmetrics.F1Score(num_classes=num_cls, average='macro')
        self.test_f1  = torchmetrics.F1Score(num_classes=num_cls, average='macro')

    def train_dataloader(self):
        loader_l = DataLoader(
            self.dataset_l,
            batch_size=self.bsz_l,
            sampler=self.sampler,
            num_workers=4,
            pin_memory=True,
        )
        loader_u = DataLoader(
            self.dataset_u,
            batch_size=self.bsz_u,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        loaders = {"l": loader_l, "u": loader_u}
        return loaders
    
    def val_dataloader(self):
        loader_v = DataLoader(
            self.dataset_v,
            batch_size=self.bsz_v,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        #loaders = {"v": loader_v}
        return loader_v

    def configure_optimizers(self):
        #optimizer = optim.SGD(
        #    self.parameters(), lr=self.lr, weight_decay=self.wd, 
        #    momentum=0.9, nesterov=True
        #)
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd, 
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.max_steps
        )
        #warmup_scheduler = optim.lr_scheduler.LinearLR(
        #    optimizer, start_factor=0.33, total_iters=500
        #    )
        #scheduler = optim.lr_scheduler.SequentialLR(
        #    optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[500]
        #)
        return [optimizer], [scheduler]
    
    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update(self.model)
    
    def forward(self, x):
        e = self.model[0](x)
        z = self.model[1](e)
        ỹ = self.model[2](z)
        return ỹ, z

    def training_step(self, batch, batch_idx):
        (x_w_l, x_s_l), y_l = batch["l"]
        (x_w_u, x_s_u), _ = batch["u"]
        
        assert x_w_l.shape[0] == x_s_l.shape[0]
        assert x_w_u.shape[0] == x_s_u.shape[0]

        bsz_l = x_w_l.shape[0]
        bsz_u = x_w_u.shape[0]

        x = torch.cat((x_w_l, x_s_l, x_w_u, x_s_u))
        y_u = torch.arange(bsz_u)

        if self.parent_cls == None:
            t_l = y_l.unsqueeze(1)
        else:
            t_l = multilabel(y_l, self.child_cls, self.parent_cls)
        t_l = torch.cat((t_l, t_l), 0)
        
        t_u = y_u.unsqueeze(1)
        t_u = torch.cat((t_u, t_u), 0)

        ỹ, z = self.forward(x)

        z_l = z[: 2 * bsz_l]
        z_u = z[2 * bsz_l :]

        ỹ_l = ỹ[: 2 * bsz_l]
        ỹ_u = ỹ[2 * bsz_l :]

        ỹ_l_w = ỹ_l[: bsz_l]
        ỹ_u_w = ỹ_u[: bsz_u]
        ỹ_u_s = ỹ_u[bsz_u :]

        y_p = torch.softmax(ỹ_u_w.detach() / self.temp_f, dim=-1)
        prob, y_p = torch.max(y_p, dim=-1)
        t_p = y_p.unsqueeze(1)
        t_p = torch.cat((t_p, t_p), 0)

        mask_l = None
        mask_u = prob.le(self.thr).float()
        mask_p = prob.ge(self.thr).float()

        loss_l = self.criterion_l(z_l, t_l, mask_l)
        loss_u = self.criterion_u(z_u, t_u, mask_u)
        loss_p = self.criterion_p(z_u, t_p, mask_p)
        loss_ce = self.criterion_c(ỹ_l_w, y_l, mask_l)
        loss_co = self.criterion_c(ỹ_u_s, y_p, mask_p)
        loss = loss_l + loss_u + loss_p + loss_ce #+ loss_co
        
        preds = torch.argmax(ỹ_l_w, dim=1)
        logits = ỹ_l_w
  
        self.train_acc(preds, y_l)
        self.train_f1(preds, y_l)
        self.train_topk(logits, y_l)
        
        self.log("train_loss", loss, prog_bar=False)      
        self.log("train_acc", self.train_acc, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)
        self.log("train_topk", self.train_topk, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model_ema.module(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.valid_acc(preds, y)
        self.valid_f1(preds, y)
        self.valid_topk(logits, y)
        
        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", self.valid_acc, prog_bar=True)
        self.log('valid_f1', self.valid_f1, prog_bar=True)
        self.log("valid_topk", self.valid_topk, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model_ema.module(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_topk(logits, y)
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)
        self.log("test_topk", self.test_topk, prog_bar=True)

#-----------------------------------------------------------------------------------
        
class S5CL_V2_MS(pl.LightningModule):
    """ 
    ꧁-----꧂ S5CL v2 MS ꧁-----꧂

    Abbreviations:
        x ... inputs
        y ... labels
        t ... targets
        w ... weak augmentation
        s ... strong augmentation
        l ... labeled
        u ... unlabeled
        p ... pseudo-labeled
        e ... encodings
        z ... embeddings
        ỹ ... logits
        bsz ... batch-size

    Args:
        model: of the form nn.Sequential(encoder, embedder, classifier)
        max_steps: total number of iterations for learning rate scheduler
        num_cls: number of classes in the leaf nodes
        child_cls: list of all class indices for all subcategories
        parent_cls: list of all class indices for all categories (can be None)
        sampler: sample classes from the labeled dataset
        dataset_l: labeled dataset
        dataset_u: unlabeled dataset
        dataset_v: validation dataset
        bsz_l: batch size for the labeled dataset
        bsz_u: batch size for the unlabeled dataset
        bsz_v: batch size for the validation dataset
        temp_l: fully-supervised temperature
        temp_u: self-supervised temperature
        temp_p: semi-supervised temperature
        temp_f: FixMatch pseudo-label temperature
        ls: label-smoothing in cross entropy
        thr: threshold for pseudo-label predictions
        lr: learning rate
        wd: weight decay

    Return:
        model and model_ema with training logs and checkpoints
    """

    def __init__(
        self,
        model,
        max_steps,
        num_cls, 
        child_cls,
        parent_cls,
        sampler,
        dataset_l,
        dataset_u,
        dataset_v,
        bsz_l,
        bsz_u,
        bsz_v,
        temp_l=0.1,
        temp_u=0.8,
        temp_p=0.1,
        temp_f=0.5, 
        ls=0.00,
        thr=0.80,
        lr=0.03,
        wd=0.0001,
    ):
        super().__init__()
        self.model = model
        self.model_ema = EMA(self.model, decay=0.999)
        #self.save_hyperparameters(ignore=["model"])

        self.child_cls = child_cls
        self.parent_cls = parent_cls
        
        self.max_steps = max_steps
        self.sampler = sampler
        
        self.dataset_l = dataset_l
        self.dataset_u = dataset_u
        self.dataset_v = dataset_v
        
        self.bsz_l = bsz_l
        self.bsz_u = bsz_u
        self.bsz_v = bsz_v
        
        self.temp_f = temp_f
        self.thr = thr
        
        self.lr = lr
        self.wd = wd

        self.criterion_l = HMLC(temperature=temp_l)
        self.criterion_u = HMLC(temperature=temp_u)
        self.criterion_p = HMLC(temperature=temp_p)
        self.criterion_c = CrossEntropy(label_smoothing=ls)
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc  = torchmetrics.Accuracy() 
        
        self.train_topk = torchmetrics.Accuracy(top_k=3)
        self.valid_topk = torchmetrics.Accuracy(top_k=3)
        self.test_topk  = torchmetrics.Accuracy(top_k=3)
        
        self.train_f1 = torchmetrics.F1Score(num_classes=num_cls, average='macro')
        self.valid_f1 = torchmetrics.F1Score(num_classes=num_cls, average='macro')
        self.test_f1  = torchmetrics.F1Score(num_classes=num_cls, average='macro')

    def train_dataloader(self):
        loader_l = DataLoader(
            self.dataset_l,
            batch_size=self.bsz_l,
            sampler=self.sampler,
            num_workers=4,
            pin_memory=True,
        )
        loader_u = DataLoader(
            self.dataset_u,
            batch_size=self.bsz_u,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        loaders = {"l": loader_l, "u": loader_u}
        return loaders
    
    def val_dataloader(self):
        loader_v = DataLoader(
            self.dataset_v,
            batch_size=self.bsz_v,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return loader_v

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=self.lr, weight_decay=self.wd, 
            momentum=0.9, nesterov=True
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.max_steps
        )
        return [optimizer], [scheduler]
    
    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update(self.model)
    
    def forward(self, x, f):
        enc, emb, crc, nrm = self.model(x, f)
        return enc, emb, crc, nrm

    def training_step(self, batch, batch_idx):
        (x_w_l, x_s_l), y_l_20x = batch["l"]
        (x_w_u, x_s_u), y_u_5x = batch["u"]
        
        assert x_w_l.shape[0] == x_s_l.shape[0]
        assert x_w_u.shape[0] == x_s_u.shape[0]

        bsz_l = x_w_l.shape[0]
        bsz_u = x_w_u.shape[0]

        x = torch.cat((x_w_l, x_s_l, x_w_u, x_s_u))
        y_u_20x = torch.arange(bsz_u)

        if self.parent_cls == None:
            t_l = y_l_20x.unsqueeze(1)
        else:
            t_l = multilabel(y_l_20x, self.child_cls, self.parent_cls)
        t_l = torch.cat((t_l, t_l), 0)
        
        y_u_20x = y_u_20x.unsqueeze(1).cuda()
        y_u_5x = torch.unsqueeze(y_u_5x, 1).cuda()
        t_u = torch.cat((y_u_5x, y_u_20x), dim=1)
        t_u = torch.cat((t_u, t_u), 0)

        fltr_l = t_l[:, 0].type(torch.bool).cpu()
        fltr_u = t_u[:, 0].type(torch.bool).cpu()
        fltr = torch.cat((fltr_l, fltr_u), 0)
        enc, emb, crc, nrm = self.forward(x, fltr)  
        logits = merge_and_order(crc, nrm, fltr)

        z_l = emb[: 2 * bsz_l]
        z_u = emb[2 * bsz_l :]

        ỹ_l = logits[: 2 * bsz_l]
        ỹ_u = logits[2 * bsz_l :]

        ỹ_l_w = ỹ_l[: bsz_l]
        ỹ_u_w = ỹ_u[: bsz_u]
        ỹ_u_s = ỹ_u[bsz_u :]

        y_p_20x = torch.softmax(ỹ_u_w.detach() / self.temp_f, dim=-1)
        prob, y_p_20x = torch.max(y_p_20x, dim=-1)
        y_p_20x = y_p_20x.unsqueeze(1)
        t_p = torch.cat((y_u_5x, y_p_20x), dim=1)
        t_p = torch.cat((t_p, t_p), 0)

        mask_l = None
        mask_u = prob.le(self.thr).float()
        mask_p = prob.ge(self.thr).float()

        loss_l = self.criterion_l(z_l, t_l, mask_l)
        loss_u = self.criterion_u(z_u, t_u, mask_u)
        loss_p = self.criterion_p(z_u, t_p, mask_p)
        loss_ce = self.criterion_c(ỹ_l_w, y_l_20x, mask_l)
        #loss_co = self.criterion_c(ỹ_u_s, y_p_20x, mask_p)
        loss = loss_l + loss_u + loss_p + loss_ce #+ loss_co
        
        preds = torch.argmax(ỹ_l_w, dim=1)
        logits = ỹ_l_w
  
        self.train_acc(preds, y_l_20x)
        self.train_f1(preds, y_l_20x)
        self.train_topk(logits, y_l_20x)
        
        self.log("train_loss", loss, prog_bar=False)      
        self.log("train_acc", self.train_acc, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)
        self.log("train_topk", self.train_topk, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        multi_label = multilabel(y, self.child_cls, self.parent_cls)
        
        fltr = multi_label[:, 0].type(torch.bool).cpu()
        enc, emb, crc, nrm = self.model_ema.module(x, fltr)   
        logits = merge_and_order(crc, nrm, fltr)
        
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.valid_acc(preds, y)
        self.valid_f1(preds, y)
        self.valid_topk(logits, y)
        
        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", self.valid_acc, prog_bar=True)
        self.log('valid_f1', self.valid_f1, prog_bar=True)
        self.log("valid_topk", self.valid_topk, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        multi_label = multilabel(y, self.child_cls, self.parent_cls)
        
        fltr = multi_label[:, 0].type(torch.bool).cpu()
        enc, emb, crc, nrm = self.model_ema.module(x, fltr)  
        logits = merge_and_order(crc, nrm, fltr)
        
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_topk(logits, y)
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)
        self.log("test_topk", self.test_topk, prog_bar=True)
        
#-----------------------------------------------------------------------------------
        
class HiMulCon(pl.LightningModule):
    def __init__(self, model, child_cls, parent_cls, cl=10, ls=0.1, tp=0.1, lr=0.03, wd=0.0001, max_steps=1000):
        super().__init__()
        self.model = model 
        self.save_hyperparameters(ignore=["model"])
        
        self.child_cls = child_cls
        self.parent_cls = parent_cls
        
        self.lr = lr
        self.wd = wd 
        self.max_steps = max_steps
        
        self.criterion_m = HMLC(temperature=tp)
        self.criterion_c = CrossEntropy(label_smoothing=ls)
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc  = torchmetrics.Accuracy() 
        
        self.train_topk = torchmetrics.Accuracy(top_k=3)
        self.valid_topk = torchmetrics.Accuracy(top_k=3)
        self.test_topk  = torchmetrics.Accuracy(top_k=3)
        
        self.train_f1 = torchmetrics.F1Score(num_classes=cl, average='macro')
        self.valid_f1 = torchmetrics.F1Score(num_classes=cl, average='macro')
        self.test_f1  = torchmetrics.F1Score(num_classes=cl, average='macro')

    def forward(self, x):
        e = self.model[0](x)
        z = self.model[1](e)
        ỹ = self.model[2](z)
        return ỹ, z

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd, 
            #momentum=0.9, nesterov=True
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.max_steps
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        image, label = batch
        logits, encodings = self.forward(image)
        
        if self.parent_cls == None:
            multi_label = label.unsqueeze(1)
        else:
            multi_label = multilabel(label, self.child_cls, self.parent_cls)
            
        loss_m = self.criterion_m(encodings, multi_label)
        loss_c = self.criterion_c(logits, label)
        
        loss = loss_m + loss_c
        preds = torch.argmax(logits, dim=1)
        
        self.train_acc(preds, label)
        self.train_f1(preds, label)
        self.train_topk(logits, label)
        
        self.log("train_loss", loss, prog_bar=False)      
        self.log("train_acc", self.train_acc, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)
        self.log("train_topk", self.train_topk, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        logits, encodings = self.forward(image)
        
        loss = F.cross_entropy(logits, label)
        preds = torch.argmax(logits, dim=1)
        
        self.valid_acc(preds, label)
        self.valid_f1(preds, label)
        self.valid_topk(logits, label)
        
        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", self.valid_acc, prog_bar=True)
        self.log('valid_f1', self.valid_f1, prog_bar=True)
        self.log("valid_topk", self.valid_topk, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        image, label = batch
        logits, encodings = self.forward(image)
        
        loss = F.cross_entropy(logits, label)
        preds = torch.argmax(logits, dim=1)
        
        self.test_acc(preds, label)
        self.test_f1(preds, label)
        self.test_topk(logits, label)
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)
        self.log("test_topk", self.test_topk, prog_bar=True)
        
#-----------------------------------------------------------------------------------

class HiMulConMS(pl.LightningModule):
    def __init__(self, model, child_cls, parent_cls, cl=10, ls=0.1, tp=0.1, lr=0.03, wd=0.0001, max_steps=1000):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        
        self.child_cls = child_cls
        self.parent_cls = parent_cls
        
        self.lr = lr
        self.wd = wd 
        self.max_steps = max_steps
        
        self.criterion_m = HMLC(temperature=tp)
        self.criterion_c = CrossEntropy(label_smoothing=ls)
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc  = torchmetrics.Accuracy() 
        
        self.train_topk = torchmetrics.Accuracy(top_k=3)
        self.valid_topk = torchmetrics.Accuracy(top_k=3)
        self.test_topk  = torchmetrics.Accuracy(top_k=3)
        
        self.train_f1 = torchmetrics.F1Score(num_classes=cl, average='macro')
        self.valid_f1 = torchmetrics.F1Score(num_classes=cl, average='macro')
        self.test_f1  = torchmetrics.F1Score(num_classes=cl, average='macro')

    def forward(self, x, f):
        enc, emb, crc, nrm = self.model(x, f)
        return enc, emb, crc, nrm

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd, 
            #momentum=0.9, nesterov=True
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.max_steps
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        image, label = batch
        multi_label = multilabel(label, self.child_cls, self.parent_cls)
        
        fltr = multi_label[:, 0].type(torch.bool).cpu()
        enc, emb, crc, nrm = self.forward(image, fltr)   
        
        crc_label = label[np.argwhere(np.asarray( fltr)).squeeze()]
        nrm_label = label[np.argwhere(np.asarray(~fltr)).squeeze()]
        logits = merge_and_order(crc, nrm, fltr)
        
        loss_emb = self.criterion_m(emb, multi_label)
        loss_crc = self.criterion_c(crc, crc_label)
        loss_nrm = self.criterion_c(nrm, nrm_label)
        
        loss = loss_emb + loss_crc + loss_nrm 
        preds = torch.argmax(logits, dim=1)
        
        self.train_acc(preds, label)
        self.train_f1(preds, label)
        self.train_topk(logits, label)
        
        self.log("train_loss", loss, prog_bar=False)      
        self.log("train_acc", self.train_acc, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)
        self.log("train_topk", self.train_topk, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        multi_label = multilabel(label, self.child_cls, self.parent_cls)
        
        fltr = multi_label[:, 0].type(torch.bool).cpu()
        enc, emb, crc, nrm = self.forward(image, fltr)  
        logits = merge_and_order(crc, nrm, fltr)
        
        loss = F.cross_entropy(logits, label)
        preds = torch.argmax(logits, dim=1)
        
        self.valid_acc(preds, label)
        self.valid_f1(preds, label)
        self.valid_topk(logits, label)
        
        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", self.valid_acc, prog_bar=True)
        self.log('valid_f1', self.valid_f1, prog_bar=True)
        self.log("valid_topk", self.valid_topk, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        image, label = batch
        multi_label = multilabel(label, self.child_cls, self.parent_cls)
        
        fltr = multi_label[:, 0].type(torch.bool).cpu()
        enc, emb, crc, nrm = self.forward(image, fltr)  
        logits = merge_and_order(crc, nrm, fltr)
        
        loss = F.cross_entropy(logits, label)
        preds = torch.argmax(logits, dim=1)
        
        self.test_acc(preds, label)
        self.test_f1(preds, label)
        self.test_topk(logits, label)
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)
        self.log("test_topk", self.test_topk, prog_bar=True)
        
#-----------------------------------------------------------------------------------

class CCE(pl.LightningModule):
    def __init__(self, model, cl=10, ls=0.1, lr=0.03, wd=0.0001, max_steps=1000):
        super().__init__()
        #self.save_hyperparameters()
        self.model = model  
        self.lr = lr
        self.wd = wd
        self.max_steps = max_steps
        
        self.criterion = CrossEntropy(label_smoothing=ls)
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc  = torchmetrics.Accuracy() 
        
        self.train_topk = torchmetrics.Accuracy(top_k=3)
        self.valid_topk = torchmetrics.Accuracy(top_k=3)
        self.test_topk  = torchmetrics.Accuracy(top_k=3)
        
        self.train_f1 = torchmetrics.F1Score(num_classes=cl, average='macro')
        self.valid_f1 = torchmetrics.F1Score(num_classes=cl, average='macro')
        self.test_f1  = torchmetrics.F1Score(num_classes=cl, average='macro')

    def forward(self, x):
        logits = self.model(x)
        return logits

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd, 
            #momentum=0.9, nesterov=True
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.max_steps
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        image, label = batch
        logits = self.model(image)
        loss = self.criterion(logits, label)
        preds = torch.argmax(logits, dim=1)

        self.train_acc(preds, label)
        self.train_f1(preds, label)
        self.train_topk(logits, label)
        
        self.log("train_loss", loss, prog_bar=False)      
        self.log("train_acc", self.train_acc, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)
        self.log("train_topk", self.train_topk, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        logits = self.model(image)
        loss = self.criterion(logits, label)
        preds = torch.argmax(logits, dim=1)
        
        self.valid_acc(preds, label)
        self.valid_f1(preds, label)
        self.valid_topk(logits, label)
        
        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", self.valid_acc, prog_bar=True)
        self.log('valid_f1', self.valid_f1, prog_bar=True)
        self.log("valid_topk", self.valid_topk, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        image, label = batch
        logits = self.model(image)
        loss = self.criterion(logits, label)
        preds = torch.argmax(logits, dim=1)

        self.test_acc(preds, label)
        self.test_f1(preds, label)
        self.test_topk(logits, label)
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)
        self.log("test_topk", self.test_topk, prog_bar=True)
