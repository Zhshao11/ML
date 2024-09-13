#数据操作
import math
import numpy as np
#读写数据
import pandas as pd 
import os
import csv
#进度条
from tqdm import tqdm
#pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
#绘制图像
from torch.utils.tensorboard import SummaryWriter
#设置种子，保证实验可重复性
def same_seed(seed):
    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
#划分数据集
def train_vaild_split(data_set,valid_ratio,seed):
    valid_data_size=int(len(data_set)*valid_ratio)
    train_data_size=len(data_set)-valid_data_size
    train_data,valid_data = random_split(data_set,[train_data_size,valid_data_size],generator=torch.Generator().manual_seed(seed))
    return np.array(train_data),np.array(valid_data)
#选择特征（选择训练集、验证集合label(y),feature,测试集的feature）feature也就是y=b+ax的x
def get_feature(train_data,valid_data,test_data,select_all = True):
    label_train = train_data[:,-1]
    label_valid = valid_data[:,-1]
    feature_train = train_data[:,:-1]
    feature_valid = valid_data[:,:-1]
    feature_test = test_data
    if select_all:
        feature_index = list(range(feature_train.shape[1]))
    else:
        pass
    return feature_train[:,feature_index],feature_valid[:,feature_index],feature_test[:,feature_index],label_train,label_valid
#构造数据集类
class Covid19Dataset(Dataset):
    def __init__(self,features,targets = None):#读取数据，并且对数据进行处理,targets取label
        if targets is None:
            self.targets = targets
        else:
            self.targets = torch.FloatTensor(targets)
        self.features = torch.FloatTensor(features)
    def __getitem__(self, idx):#每次从数据集读取一笔数据
        if self.targets is None:
            return self.features[idx]
        else:
            return self.features[idx],self.targets[idx]
    def __len__(self):
        return len(self.features)
#构造神经网络
class My_Model(nn.Module):#My_Model继承nn.Module
    #重写构造函数
    def __init__(self,input_dim):
        super(My_Model,self).__init__()
    #容器像搭了架子，把一层一层的layer添加进去
        self.layers = nn.Sequential(
            nn.Linear(input_dim,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
    #重写前向传播函数，传入的数据经过layer输出的过程
    def forward(self,x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x
#参数设置
device = 'cuda'if torch.cuda.is_available() else 'cpu'
config = {
    'seed':5201314,
    'select_all':True,
    'valid_ratio':0.2,
    'n_epochs':3000,
    'batch_size':256,
    'learning_rate':1e-5,
    'early_stop':400,
    'save_path':'/Users/shihaozhu/Downloads/ML_HW/model.ckpt'
}
#训练过程
def trainer(train_loader,valid_loader,model,config,device):
    criterion = nn.MSELoss(reduction='mean')#损失函数
    optimizer = torch.optim.SGD(model.parameters(),lr = config['learning_rate'],momentum=0.9)
    writer =  SummaryWriter()
    if not os.path.isdir('/Users/shihaozhu/Downloads/ML_HW'):
        os.mkdir('/Users/shihaozhu/Downloads/ML_HW')
    n_epochs = config['n_epochs']
    best_loss = math.inf
    step = 0
    early_stop_count = 0
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_pbar = tqdm(train_loader,position=0,leave=True)
        #train loop
        for x,y in train_loader:
            optimizer.zero_grad()
            x,y = x.to(device),y.to(device)
            predits = model(x)
            loss = criterion(predits,y)
            loss.backward()
            optimizer.step()
            step+=1
            loss_record.append(loss.detach().item())
            #显示训练过程
            train_pbar.set_description(f'Epoch[{epoch +1}/{n_epochs}]')
            train_pbar.set_postfix({'loss':loss.detach().item()})
            train_pbar.update(1)  
        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/Train',mean_train_loss,step)
        #valid loop，验证集用于评估模型在训练过程中的表现。它不会参与模型参数的更新。
        model.eval()
        loss_record = []
        for x,y in valid_loader:
            x,y = x.to(device),y.to(device)
            with torch.no_grad():#验证的时候不需要梯度
                pred = model(x)
                loss = criterion(pred,y)
            loss_record.append(loss.item())
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch[{epoch+1}/{n_epochs}]:Train loss:{mean_train_loss:.4f},valid loss:{mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid',mean_valid_loss,step)
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(),config['save_path'])
            print('Saving model with loss{:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count+=1
        if early_stop_count>=config['early_stop']:
            print('\n Model is not improving,so we halt the training session')
            return
"准备工作"
#设置随机种子
same_seed(config['seed']) 
#读取数据
train_data = pd.read_csv('/Users/shihaozhu/Downloads/ML_HW/HW01/covid.train.csv').values
test_data = pd.read_csv('/Users/shihaozhu/Downloads/ML_HW/HW01/covid.test.csv').values
#划分数据集
train_data,valid_data = train_vaild_split(train_data,config['valid_ratio'],config['seed'])
print(train_data.shape,valid_data.shape,test_data.shape)
#选择特征
feature_train,feature_valid,feature_test,label_train,label_valid = get_feature(train_data,valid_data,test_data,config['select_all'])
print(feature_train.shape[1])
#构造数据集Dataset
train_dataset = Covid19Dataset(feature_train,label_train)
valid_dataset = Covid19Dataset(feature_valid,label_valid)
test_dataset = Covid19Dataset(feature_test)
#准备Dataloader
#Dataloader是一个迭代器，可以多线程地读取数据，并可将几笔数据组成batch
#Dataloader参数：
# batch_size 一批次的大小，并行运行批次，模型参数的变化是在每一个批次训练后进行变化，不是一个epoch才变一次，也不是逐样本变化，每个批次（batch）经过一次前向传播、反向传播和参数更新。
# 批次（batch）和epoch还有样本的关系：
# 一个epoch中包含许多批次，样本量/batch_size=batch,训练模型过程是在训练集上进行训练，然后在验证集上进行Loss的评估，最后留下loss最低的model,一个epoch里包含许多batch,model的参数会在每一个batch训练完进行更新
# 然后进行覆盖，留下最后的model当作这一轮epoch的model,然后验证集对模型进行的评估是比较不同epoch的model留下最好的model
# shuffle:是否打乱数据集，训练集，验证集要，测试集不要
# pin_memory：设为true时，把数据张量固定在GPU内存中（加速数据加载）
train_loader = DataLoader(train_dataset,batch_size = config['batch_size'],shuffle=True,pin_memory=True)
valid_loader = DataLoader(valid_dataset,batch_size = config['batch_size'],shuffle=True,pin_memory=True)
test_loader = DataLoader(test_dataset,batch_size = config['batch_size'],shuffle=False,pin_memory=True)           
#开始训练！
model = My_Model(input_dim=feature_train.shape[1]).to(device)
trainer(train_loader,valid_loader,model,config,device)
#预测
def predict(test_loader,model,device):
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds,dim=0).numpy()
    return preds
def save_pred(preds,file):
    with open(file,'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id','tested_positive']) 
        for i,p in enumerate(preds):
            writer.writerow([i,p])
#预测并保存结果
model = My_Model(input_dim=feature_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader,model,device)
save_pred(preds,'/Users/shihaozhu/Downloads/ML_HW/preds.csv')        