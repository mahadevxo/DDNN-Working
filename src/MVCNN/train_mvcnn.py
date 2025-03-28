import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=8)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.001)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-train_path", type=str, default="ModelNet40_12View/*/train")
parser.add_argument("-val_path", type=str, default="ModelNet40_12View/*/test")
parser.set_defaults(train=False)

def create_folder(log_dir):
    if os.path.exists(log_dir):
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
    os.mkdir(log_dir)

def load_checkpoint(model, checkpoint_path):
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            print(f"Checkpoint not found at {checkpoint_path}, starting from scratch.")

if __name__ == '__main__':
    args = parser.parse_args()
    device = 'mps' if torch.backends.mps.is_available() else \
    'cuda' if torch.cuda.is_available() else \
        'cpu'
    pretraining = not args.no_pretraining
    log_dir = args.name
    create_folder(args.name)
    with open(os.path.join(log_dir, 'config.json'), 'w') as config_f:
        json.dump(vars(args), config_f)

    # Ask user if they want to train SVCNN again
    train_svcnn = input("Do you want to train SVCNN again? (yes/no): ").strip().lower()

    if train_svcnn == 'yes':
        print("TRAINING SVCNN")
        # Train SVCNN from scratch
        log_dir = f'{args.name}_stage_1'
        create_folder(log_dir)
        cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name).to(device)

        optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        n_models_train = args.num_models * args.num_views

        train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

        val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        print(f'num_train_files: {len(train_dataset.filepaths)}')
        print(f'num_val_files: {len(val_dataset.filepaths)}')
        trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1)
        trainer.train(10)
    else:
        # Load pre-trained SVCNN model
        log_dir = f'{args.name}_stage_1'
        cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name).to(device)
        svcnn_checkpoint = 'MVCNN_stage_1/MVCNN/model-00006.pth'
        load_checkpoint(cnet, svcnn_checkpoint)

    # Train MVCNN
    log_dir = f'{args.name}_stage_2'
    create_folder(log_dir)
    cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views).to(device)
    del cnet

    mvcnn_checkpoint = 'MVCNN_stage_1/MVCNN/model-00006.pth'
    load_checkpoint(cnet_2, mvcnn_checkpoint)
    n_models_train = args.num_models * args.num_views
    print("TRAINING MVCNN")

    optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4)  # shuffle needs to be false! it's done within the trainer

    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    print(f'num_train_files: {len(train_dataset.filepaths)}')
    print(f'num_val_files: {len(val_dataset.filepaths)}')
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=args.num_views)
    trainer.train(10)