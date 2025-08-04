import sys
sys.path.append('../../../MVCNN/')
from models import MVCNN
from tools.ImgDataset import MultiviewImgDataset
import sys
import os
import torch
import numpy as np
tqdm = __import__('tqdm').tqdm

# Configuration
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRUNING_AMOUNTS = np.arange(0.0, 1.02, 0.02).tolist()
NUM_VIEWS = 12
TEST_DIR = '../../../MVCNN/ModelNet40-12View/*/test'
FEATURE_DIR = './cached-features'
os.makedirs(FEATURE_DIR, exist_ok=True)

# Load models once, cache net_1 outputs per prune amount and sample

def get_org_model():
    base = MVCNN.SVCNN(name='svcnn', nclasses=33, cnn_name='vgg11')
    w = torch.load('../../../MVCNN/MVCNN/MVCNN/model-00050.pth', map_location=device)
    base.load_state_dict(w)
    mvcnn = MVCNN.MVCNN(name='mvcnn', model=base.to(device), num_views=NUM_VIEWS, cnn_name='vgg11')
    return mvcnn.to(device)

# Load all pruned models into dict
models = {}
for p in PRUNING_AMOUNTS:
    path = f'./pruned-models/pruned_mvcnn_{p}.pth'
    if os.path.exists(path):
        models[p] = torch.jit.load(path, map_location=device)

# Prepare dataset

dataset = MultiviewImgDataset(root_dir=TEST_DIR, num_views=NUM_VIEWS, shuffle=False, num_models=0, test_mode=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                     num_workers=4, pin_memory=True)

# Iterate and cache
with torch.no_grad():
    for idx, data in enumerate(tqdm(loader, desc='Caching features')):
        label, views = data[0].item(), data[1][0]
        for p, model in models.items():
            model = model.to(device)
            feats = []
            for v in range(NUM_VIEWS):
                img = views[v].unsqueeze(0).to(device)
                out = model.net_1(img)
                feats.append(out.cpu().numpy())
            feats = np.stack(feats, axis=0)  # shape [12,512,7,7]
            np.save(os.path.join(FEATURE_DIR, f'sample_{idx}_prune_{p}.npy'), feats)
            model.to('cpu')

print('Feature caching complete.')

