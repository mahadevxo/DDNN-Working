from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image

class SportsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, split="train"):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['data set'] == split]
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['filepaths'])
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['class id']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label