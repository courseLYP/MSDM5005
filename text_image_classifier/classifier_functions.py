from scipy.io import loadmat
import os
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import AutoModel, BertTokenizerFast, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import optim
import time
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import statistics
from statistics import NormalDist
import random
import numpy as np
import torch.nn.functional as F

device = "cuda"
pdist = nn.PairwiseDistance(p=2)


def loss_func(txt, img):
    samepairdist=torch.tensor(0)
    diffpairdist=torch.tensor(1e-10)

    truthcounter=0
    falsecounter=0
    samepairdists = []

    for titem, tsample in enumerate(txt):
        for iitem, isample in enumerate(img):
            if titem == iitem:
                samepairdist = samepairdist+pdist(tsample, isample).sum()
                truthcounter+=1
                samepairdists.extend([pdist(tsample, isample).sum().item()])
            else:
                diffpairdist = diffpairdist+pdist(tsample, isample).sum()
                falsecounter+=1
    final_loss = samepairdist * falsecounter / diffpairdist / truthcounter

    return final_loss, samepairdists


def train(imgmodel, txtmodel, train_dataloader, optimizer):
    imgmodel.train()
    txtmodel.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):

        batch = [r.to(device) for r in batch]
        seq, mask, image_tensor = batch

        imgmodel.zero_grad()
        txtmodel.zero_grad()

        txtout = txtmodel(seq, mask)
        imgout = imgmodel(image_tensor)


        loss, samepairdist = loss_func(txtout, imgout)
        loss = loss.sum()


        del txtout
        del imgout

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)

    return avg_loss, samepairdist


def evaluate(imgmodel, txtmodel, val_dataloader):
    imgmodel.eval()
    txtmodel.eval()
    total_loss = 0
    for step, batch in enumerate(val_dataloader):
        batch = [t.to(device) for t in batch]
        seq, mask, image_tensor = batch

        with torch.no_grad():
            txtout = txtmodel(seq, mask)
            imgout = imgmodel(image_tensor)

            loss, samepairdist = loss_func(txtout, imgout)
            loss = loss.sum()
            total_loss = total_loss+loss.item()

            del txtout,imgout

    avg_loss = total_loss / len(val_dataloader)
    return avg_loss, samepairdist


def test(imgmodel, txtmodel, test_dataloader, topk=5):
    outtxt = torch.Tensor()
    outimg = torch.Tensor()

    check = torch.Tensor()
    outindex = torch.Tensor()

    for step, batch in enumerate(test_dataloader):

        imgmodel.eval()
        txtmodel.eval()
        batch = [t.to(device) for t in batch]
        seq, mask, image_tensor = batch

        with torch.no_grad():
            txtout = txtmodel(seq, mask)
            imgout = imgmodel(image_tensor)

        txtout.detach().cpu()
        imgout.detach().cpu()
        seq.detach().cpu()
        imgout.detach().cpu()

        if len(outtxt) == 0:
            inseq = seq
            inimg = image_tensor
            outtxt = txtout
            outimg = imgout
        else:
            inseq = torch.cat((inseq, seq))
            inimg = torch.cat((inimg, image_tensor))
            outtxt = torch.cat((outtxt, txtout))
            outimg = torch.cat((outimg, imgout))

    for txtitem, txtsample in enumerate(outtxt):
        pairdist = torch.Tensor()
        for imgitem, imgsample in enumerate(outimg):
            if len(pairdist) == 0:
                pairdist = torch.unsqueeze(pdist(imgsample, txtsample).sum(), 0)
            else:
                pairdist = torch.cat((pairdist, torch.unsqueeze(pdist(imgsample, txtsample).sum(), 0)))

            if imgitem == txtitem:
                truthdist = pdist(imgsample, txtsample).sum()

        topkdist, topkindex = torch.topk(pairdist, k=topk, largest=False)
        if topkdist[-1] > truthdist:
            if len(check) == 0:
                check = torch.BoolTensor([True])

            else:
                check = torch.cat((check, torch.BoolTensor([True])))
        else:
            if len(check) == 0:
                check = torch.BoolTensor([False])
            else:
                check = torch.cat((check, torch.BoolTensor([False])))

        if len(outindex) == 0:
            outindex = topkindex
        else:
            outindex = torch.cat((outindex, topkindex))

    check = torch.reshape(check, (len(inseq), 1))
    outindex = torch.reshape(outindex, (len(inseq), topk))

    print(inseq.shape, inimg.shape, outtxt.shape, outimg.shape, check.shape, outindex.shape)
    return inimg, check, outindex