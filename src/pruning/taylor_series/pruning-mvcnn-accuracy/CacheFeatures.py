import sys
sys.path.append('../../../MVCNN/')
import os
import torch
import numpy as np
from tqdm import tqdm
from tools.ImgDataset import MultiviewImgDataset

class CacheFeatures:
    def __init__(self):
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.PRUNING_AMOUNTS = np.arange(0.0, 1.02, 0.02).tolist()
        self.NUM_VIEWS = 12
        self.TEST_DIR = '../../../MVCNN/ModelNet40-12View/*/test'
        self.FEATURE_DIR = './cached-features'
        os.makedirs(self.FEATURE_DIR, exist_ok=True)
        self.loader = self.prepare_dataset()
        self.models = self.load_pruned_models()

    # Load pruned models into dict
    def load_pruned_models(self):
        models = {}
        for p in self.PRUNING_AMOUNTS:
            path = f'./pruned-models/pruned_mvcnn_{p}.pth'
            if os.path.exists(path):
                m = torch.jit.load(path, map_location=self.device)
                m.eval()
                models[p] = m
        return models
    
    def prepare_dataset(self):
        dataset = MultiviewImgDataset(
            root_dir=self.TEST_DIR,
            num_views=self.NUM_VIEWS,
            shuffle=False,
            num_models=0,
            test_mode=True
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        return loader

# Cache features and labels
    def cache_features(self):
        with torch.no_grad():
            for idx, data in enumerate(tqdm(self.loader, desc='Caching features')):
                label = data[0].item()
                views = data[1][0]  # shape [NUM_VIEWS, C, H, W]
                for p, model in self.models.items():
                    feats = []
                    for v in range(self.NUM_VIEWS):
                        img = views[v].unsqueeze(0).to(self.device)
                        out = model.net_1(img)  # shape [1,512,7,7]
                        feats.append(out.cpu().numpy())
                    feats = np.stack(feats, axis=0)
                    # save features and label
                    np.save(os.path.join(self.FEATURE_DIR, f'sample_{idx}_prune_{p}_feats.npy'), feats)
                    np.save(os.path.join(self.FEATURE_DIR, f'sample_{idx}_label.npy'), np.array(label))
    
    def run(self):
        self.cache_features()
        print(f'Features cached in {self.FEATURE_DIR}')