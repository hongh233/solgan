# -*- coding: utf-8 -*-


from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import pandas as pd
import numpy as np




class Dataset(data.Dataset):
    
    def __init__(self,mode = 'train', transform = None):
        
        self.transform = transform
        '''
        with open('mnist/train/train.txt') as f:
            
            files = f.readlines()
            self.filenames = [filename.strip('\n')[:-1] for filename in files]
            self.labels = [int(file.strip('\n')[-1:]) for file in files]
        '''
        filenames = os.listdir('./celebA')
        self.filenames = [os.path.join('./celebA', file) for file in filenames]
        
        
    def __getitem__(self,index):
        
        img = Image.open(self.filenames[index])
        return self.transform(img)
        
    def __len__(self):
        
        return len(self.filenames)
        
class Dataset_unlabel(data.Dataset):
    
    def __init__(self,transform = None, transform_aug = None):
        
        self.transform = transform
        self.transform_aug = transform_aug
        self.filenames = []
        root = './dataset/train/no_label/'
        file_dirs = os.listdir(root)
        for fd in file_dirs:
            fd = os.path.join(root, fd)
            fns = os.listdir(fd)
            for fn in fns:
                fn = os.path.join(fd, fn)
                self.filenames.append(fn)
                
        np.random.shuffle(self.filenames)
        
    def __getitem__(self,index):
        
        img = Image.open(self.filenames[index])
        return self.transform(img), self.transform_aug(img)
        
    def __len__(self):
        
        return len(self.filenames)
        
class Dataset_1D(data.Dataset):
    def __init__(self, transform = None):
        self.transform = transform
        self.data = np.array(pd.read_excel('2_data.xlsx'))
        
    def __getitem__(self,index):
        
        data = self.data[index]
        return data
        
    def __len__(self):
        
        return len(self.data)
        
        
        
def get_loader(mode = 'train/labaled/',batch_size = 128):
    
    transform = []
    transform += [T.Resize((64,64))]
    transform += [T.ToTensor()]
    transform += [T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))]
    transform = T.Compose(transform)
    path = os.path.join('dataset/', mode)
    dataset = ImageFolder(path, transform)
    data_loader = data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    
    return data_loader
    
def get_loader_unlabel(batch_size = 128):
    transform = []
    transform += [T.Resize((128,128))]
    transform += [T.ToTensor()]
    transform += [T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))]
    transform = T.Compose(transform)
    
    transform_aug = []
    transform_aug += [T.RandomHorizontalFlip(p = 0.5), T.RandomRotation(30), T.RandomVerticalFlip(p = 0.5)]
    transform_aug += [T.Resize((128,128))]
    transform_aug += [T.ToTensor()]
    transform_aug += [T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))]
    transform_aug = T.Compose(transform_aug)
    
    dataset = Dataset_unlabel(transform, transform_aug)
    data_loader = data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    
    return data_loader
    
    
def get_loader_1D(batch_size = 128):
    dataset = Dataset_1D()
    data_loader = data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    return data_loader
    
    

    

    
    



