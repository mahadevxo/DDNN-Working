import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil
import argparse

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

name = "MVCNN-Jetson"
batchSize = 8
num_models = 1000
lr = 5e-5
weight_decay = 0.001
no_pretraining = False
cnn_name = "resnet18"
num_views = 12  
train_path = "modelnet40_images_new_12x/*/train"
test_path = "modelnet40_images_new_12x/*/test"

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    pretraining = not no_pretraining

    # STAGE 1
    log_dir = name+'_stage_1'
    create_folder(log_dir)
    cnet = SVCNN(name, nclasses=40, pretraining=pretraining, cnn_name=cnn_name)
    optimizer = optim.Adam(cnet.parameters(), lr=lr, weight_decay=weight_decay)
    num_models_train = num_models*num_views
    
    train_dataset = SingleImgDataset(train_path, scale_aug=False, rot_aug=False, num_models=num_models_train, num_views=num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    
    val_dataset = SingleImgDataset(test_path, scale_aug=False, rot_aug=False, test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)
    
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    
    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1)
    
    trainer.train(10)
    
        # STAGE 2
    log_dir = name+'_stage_2'
    create_folder(log_dir)
    cnet_2 = MVCNN(name, cnet, nclasses=40, cnn_name=cnn_name, num_views=num_views)
    del cnet

    optimizer = optim.Adam(cnet_2.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    
    train_dataset = MultiviewImgDataset(train_path, scale_aug=False, rot_aug=False, num_models=num_models_train, num_views=num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=False, num_workers=8)

    val_dataset = MultiviewImgDataset(test_path, scale_aug=False, rot_aug=False, num_views=num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchSize, shuffle=False, num_workers=8)
    
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=num_views)
    trainer.train(30)