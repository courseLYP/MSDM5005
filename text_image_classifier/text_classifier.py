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

proposed_model = input("Choose: CNN, GoogleNet, ResNet18, ResNet34")

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

device = torch.device("cuda")

# label = loadmat('e:\\imagelabels')
# label = label['labels'][0]
# print(label.min())


imgpath = './data/flowers102/images'
txtpath = './data/flowers102/text'

imgsam = []
os.chdir(imgpath)
convert_tensor = transforms.ToTensor()
for _, _, filesname in os.walk(imgpath):
    for file in filesname:
        img = Image.open(file)
        imgsam.extend([convert_tensor(img)])
txtdes = []
os.chdir(txtpath)
for _, _, filesname in os.walk(txtpath):
    for file in filesname:
        with open(file, 'r') as f:
            txtdes.extend([f.read().replace('\n',' ')])

train_txt, temp_txt, train_img, temp_img = train_test_split(txtdes, imgsam, test_size=0.5, random_state=2022)
val_txt, test_txt, val_img, test_img = train_test_split(temp_txt, temp_img, train_size=0.05, test_size=0.05, random_state=2022)

max_seq_len = 150


bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# tokenize and encode sequences
tokens_train = tokenizer.batch_encode_plus(train_txt, max_length = max_seq_len, pad_to_max_length=True, truncation=True, return_token_type_ids=False)
tokens_val = tokenizer.batch_encode_plus(val_txt, max_length = max_seq_len, pad_to_max_length=True, truncation=True, return_token_type_ids=False)
tokens_test = tokenizer.batch_encode_plus(test_txt, max_length = max_seq_len, pad_to_max_length=True, truncation=True, return_token_type_ids=False)

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])

# Stack image Tensor
train_img = torch.stack(train_img)
val_img = torch.stack(val_img)
test_img = torch.stack(test_img)

train_data = TensorDataset(train_seq, train_mask, train_img)
val_data = TensorDataset(val_seq, val_mask, val_img)
test_data = TensorDataset(test_seq, test_mask, test_img)

batchsize = 20

train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batchsize, drop_last=True)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batchsize)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batchsize)


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

class CNN_Model(nn.Module):
    def __init__(self, infeature=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(55696, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, infeature)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


noofoutfeature = 1000

if proposed_model == 'CNN':
    imgmodel = CNN_Model(noofoutfeature)

elif proposed_model == 'GoogleNet':
    imgmodel = models.googlenet(weights='DEFAULT')
    infeature_googlenet = imgmodel.fc.in_features
    imgmodel.fc = nn.Linear(infeature_googlenet, noofoutfeature)

elif proposed_model == 'ResNet18':
    imgmodel = models.resnet18(weights='DEFAULT')   
    infeature_ResNet = imgmodel.fc.in_features  
    imgmodel.fc = nn.Linear(infeature_ResNet, noofoutfeature)

elif proposed_model == 'ResNet34':
    imgmodel = models.resnet34(weights='DEFAULT')
    infeature_ResNet = imgmodel.fc.in_features  
    imgmodel.fc = nn.Linear(infeature_ResNet, noofoutfeature)


imgmodel = imgmodel.cuda()
txtmodel = BERT_Arch(bert, noofoutfeature)
txtmodel = txtmodel.cuda()

learning_rate = 1e-5

optimizer = optim.AdamW([{'params': txtmodel.parameters()},
                {'params': imgmodel.parameters()}], lr=learning_rate)

Epochs = 20
val_loss = []

for param in bert.parameters():
    param.requires_grad = True


pdist = nn.PairwiseDistance(p=2)

def loss_func(txt, img):  # 改
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


def train():
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


def evaluate():
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


def test(topk=5):
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


# def test(inclusive_dist):   
#     imgmodel.eval() 
#     txtmodel.eval() 
#     for step, batch in enumerate(test_dataloader):  
#         batch = [t.to(device) for t in batch]   
#         seq, mask, image_tensor = batch 
#         with torch.no_grad():   
#             txtout = txtmodel(seq, mask)    
#             imgout = imgmodel(image_tensor) 
#         txtout.detach().cpu()   
#         imgout.detach().cpu()   
#         if step == 0:   
#             outtxt = txtout 
#             outimg = imgout 
#         else:   
#             outtxt = torch.cat((outtxt, txtout))    
#             outimg = torch.cat((outimg, imgout))    
#         check = []  
#         for txtitem, txtsample in enumerate(outtxt):    
#             smallestdist = float('inf') 
#             for imgitem, imgsample in enumerate(outimg):    
#                 if imgitem == txtitem:  
#                     truthdist = pdist(imgsample, txtsample).sum().item()    
#                 if pdist(txtsample, imgsample).sum().item() < smallestdist: 
#                     imgcheck = imgitem  
#                     smallestdist = pdist(txtsample, imgsample).sum().item() 
#             check.append([txtitem, imgcheck, True if truthdist<= max(inclusive_dist, smallestdist) else False, truthdist, smallestdist])    
#     return check    


best_val_loss = float('inf')

modelpath = './models'
os.chdir(modelpath)

train_losses = []
eval_losses = []
elapsedtime = time.time()

for epoch in range(Epochs):
    print('Epoch %d' % (epoch+1))
    train_loss, train_samepairdist = train()
    eval_loss, eval_samepairdist = evaluate()

    if eval_loss < best_val_loss:
        best_val_loss = eval_loss
        print('Saving Model')
        torch.save(txtmodel, 'Text_Model_' + proposed_model + '.pt')
        torch.save(imgmodel, 'Image_Model_' + proposed_model + '.pt')
        mean = statistics.mean(train_samepairdist)
        sd = statistics.stdev(eval_samepairdist)
        dist = NormalDist()
        inclusive_dist = mean

    print('Training Loss: %.5f' % train_loss)
    print('Validation Loss: %.5f' % eval_loss)
    print('Classify L2 Distance: %.5f' % inclusive_dist)

    train_losses.extend([train_loss])
    eval_losses.extend([eval_loss])


running_result = 'result' + proposed_model + '.dat'

elapsedtime = time.time()-elapsedtime
print('Running Time: %d min %d sec' % (elapsedtime//60, elapsedtime % 60))

_, check, _ = test(topk=int(len(test_seq) * 0.03))
check = check.tolist()

counter = 0
for checking in check:
    if checking:
        counter += 1


# check = test(inclusive_dist)    
# counter = 0 
# for checking in check:  
#     if checking[2]: 
#         counter += 1


Falsecounter = len(check)-counter
Accuracy = counter / len(check)
print('No of Truth case: \t %d' % counter)
print('No of False case: \t %d' % Falsecounter)
print('Accuracy: \t \t %.4f' % (Accuracy*100))

result = [counter+Falsecounter, counter, Accuracy*100, elapsedtime]

Classifier = [mean, sd, inclusive_dist]
dictionary = {'Name': proposed_model, 'Tloss': train_losses, 'Eloss': eval_losses, 'Critieria': Classifier, 'Result': result}


f = open(running_result, 'wb')
pickle.dump(dictionary, f)
f.close()

x = [num for num in range(1, len(train_losses)+1)]

lines = plt.plot(x, train_losses, x, eval_losses)
plt.setp(lines[0], linewidth=1)
plt.xticks(x)
plt.legend(('Training Loss', 'Evaluation Loss'), loc='upper right')
plt.title('Training Progress Diagram (' + proposed_model + ' Model)')
plt.savefig(proposed_model + '.png')
plt.show()
plt.close()