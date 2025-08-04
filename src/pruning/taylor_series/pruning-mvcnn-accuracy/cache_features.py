import sys
sys.path.append('../../../MVCNN/')
import os
import torch
import numpy as np
from tqdm import tqdm
from tools.ImgDataset import MultiviewImgDataset

# Configuration
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRUNING_AMOUNTS = np.arange(0.0, 1.02, 0.02).tolist()
NUM_VIEWS = 12
TEST_DIR = '../../../MVCNN/ModelNet40-12View/*/test'
FEATURE_DIR = './cached-features'
os.makedirs(FEATURE_DIR, exist_ok=True)

# Load pruned models into dict
def load_pruned_models():
    models = {}
    for p in PRUNING_AMOUNTS:
        path = f'./pruned-models/pruned_mvcnn_{p}.pth'
        if os.path.exists(path):
            m = torch.jit.load(path, map_location=device)
            m.eval()
            models[p] = m
    return models

# Prepare dataset
dataset = MultiviewImgDataset(
    root_dir=TEST_DIR,
    num_views=NUM_VIEWS,
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

models = load_pruned_models()

# Cache features and labels
with torch.no_grad():
    for idx, data in enumerate(tqdm(loader, desc='Caching features')):
        label = data[0].item()
        views = data[1][0]  # shape [NUM_VIEWS, C, H, W]
        for p, model in models.items():
            feats = []
            for v in range(NUM_VIEWS):
                img = views[v].unsqueeze(0).to(device)
                out = model.net_1(img)  # shape [1,512,7,7]
                feats.append(out.cpu().numpy())
            feats = np.stack(feats, axis=0)
            # save features and label
            np.save(os.path.join(FEATURE_DIR, f'sample_{idx}_prune_{p}_feats.npy'), feats)
            np.save(os.path.join(FEATURE_DIR, f'sample_{idx}_label.npy'), np.array(label))

print('Feature caching complete.')