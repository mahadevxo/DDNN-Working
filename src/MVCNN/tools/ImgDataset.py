import numpy as np
import glob
import torch.utils.data
from PIL import Image
import torch
from torchvision import transforms

class MultiviewImgDataset(torch.utils.data.Dataset): #DONE FINALLY

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                         num_models=0, num_views=12, shuffle=True):
        self.class_names = ['airplane','bed','bench','bookshelf','bottle','bowl','car','chair',
                            'cone','cup','curtain','door','flower_pot','glass_box',
                            'guitar','keyboard','lamp','laptop','mantel','person','piano',
                            'plant','radio','range_hood','sink','stairs',
                            'stool','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views

        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/', 2)[0]
        self.filepaths = []

        for item in self.class_names:
            model_dirs = sorted(glob.glob(f'{parent_dir}/{item}/{set_}/*/'))
            if num_models > 0:
                model_dirs = model_dirs[:min(num_models, len(model_dirs))]
            for model_dir in model_dirs:
                x = f'{model_dir}/{model_dir.split("/")[-2]}/*.png'
                view_paths = sorted(glob.glob(x))
                # print(view_paths)
                stride = int(12 / self.num_views)
                view_paths = view_paths[::stride]
                if len(view_paths) == self.num_views:
                    self.filepaths.append(view_paths)
                else:
                    print(f"Skipping {model_dir}, found {len(view_paths)} views")

        if shuffle is True:
            np.random.shuffle(self.filepaths)

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Fixed: was 244, should be 224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Fixed: was 244, should be 224
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        paths = self.filepaths[idx]
        class_name = paths[0].split('/')[-6]
        # print(class_name)
        class_id = self.class_names.index(class_name)
        imgs = []
        for path in paths:
            im = Image.open(path).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)
        return (class_id, torch.stack(imgs), paths)


class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                         num_models=0, num_views=12):
        self.class_names = ['airplane','bed','bench','bookshelf','bottle','bowl','car','chair',
                            'cone','cup','curtain','door','flower_pot','glass_box',
                            'guitar','keyboard','lamp','laptop','mantel','person','piano',
                            'plant','radio','range_hood','sink','stairs',
                            'stool','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode

        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/', 2)[0]
        self.filepaths = []
        for item in self.class_names:
            x = f'{parent_dir}/{item}/{set_}/*/*/*.png'
            # print(x)
            all_files = sorted(glob.glob(x))
            if num_models == 0:
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models * num_views, len(all_files))])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-5]
        class_id = self.class_names.index(class_name)
        im = Image.open(path).convert('RGB')
        if self.transform:
            im = self.transform(im)
        return (class_id, im, path)