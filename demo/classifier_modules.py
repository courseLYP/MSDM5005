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
import matplotlib.pyplot as plt


class BERT_Arch(nn.Module):
    def __init__(self, bert, nooffeatures):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(512,nooffeatures)
        self.softmax = nn.LogSoftmax(dim=1)  # 改、刪？

    def forward(self, sent_id, mask):
        #pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    
def classify(input_text, txtmodel_pth, imgmodel_pth, fake_image_dir, topk = 5):
    # load models
    device = torch.device("cuda")

    max_seq_len = 150
    pdist = nn.PairwiseDistance(p=2)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    txtmodel = torch.load(txtmodel_pth)
    imgmodel = torch.load(imgmodel_pth)
    
    # encode input text
    input_txt_token = tokenizer.batch_encode_plus([input_text], max_length = max_seq_len, pad_to_max_length=True, truncation=True, return_token_type_ids=False)
    input_seq = torch.tensor(input_txt_token['input_ids']).to("cuda")
    input_mask = torch.tensor(input_txt_token['attention_mask']).to("cuda")
    with torch.no_grad():
        input_txt_seq = txtmodel(input_seq, input_mask)
    input_txt_seq.detach().cpu()
    input_txt_seq.shape
    
    # load candidate images
    image_files = []
    img_tensors = []
    convert_tensor = transforms.ToTensor()
    os.chdir(fake_image_dir)
    for _, _, filesname in os.walk(fake_image_dir):
        for file in sorted(filesname):
            if not os.path.exists(file):
                continue
            img = Image.open(file)
            img_tensors.extend([convert_tensor(img)])
            image_files.append(file)

    # imaga data encode
    img_tensors = torch.stack(img_tensors).to("cuda")
    with torch.no_grad():
        imgout = imgmodel(img_tensors)
    imgout.detach().cpu()
    outimg = imgout
    
    # pairing
    pairdist = torch.Tensor()
    for image_id, img_code in enumerate(outimg):
        if len(pairdist) == 0:
            pairdist = torch.unsqueeze(pdist(img_code, input_txt_seq).sum(), 0)
        else:
            pairdist = torch.cat((pairdist, torch.unsqueeze(pdist(img_code, input_txt_seq).sum(), 0)))
    pairdist
    
    topkdist, topkindex = torch.topk(pairdist, k=topk, largest=False)
    topkdist, topkindex
    
    cadidates_image_files = [image_files[_index] for _index in topkindex]
    return cadidates_image_files, topkdist