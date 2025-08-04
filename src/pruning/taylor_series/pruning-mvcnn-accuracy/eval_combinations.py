import os
import numpy as np
import torch
from models import MVCNN
from tqdm import tqdm

# Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURE_DIR = './cached-features'
PRUNING_AMOUNTS = np.arange(0.0, 1.02, 0.02).tolist()
NUM_VIEWS = 12
P_MATRIX = np.random.choice(PRUNING_AMOUNTS, size=(2000, NUM_VIEWS))
LOG_CSV = 'test-log.csv'

# Load org model net_2
base = MVCNN.SVCNN(name='svcnn', nclasses=33, cnn_name='vgg11')
w = torch.load('../../../MVCNN/MVCNN/MVCNN/model-00050.pth', map_location=device)
base.load_state_dict(w)
org = MVCNN.MVCNN(name='mvcnn', model=base.to(device), num_views=NUM_VIEWS, cnn_name='vgg11').to(device)

# Helper: load features
def load_feats(idx, prune_amount):
    path = os.path.join(FEATURE_DIR, f'sample_{idx}_prune_{prune_amount}.npy')
    return torch.from_numpy(np.load(path))

# Open CSV
with open(LOG_CSV, 'w') as f:
    header = ','.join(f'prune_v{i}' for i in range(NUM_VIEWS)) + ',mean_class_acc\n'
    f.write(header)

# Evaluate combinations
num_samples = len(set(name.split('_')[1] for name in os.listdir(FEATURE_DIR)))
for comb in tqdm(P_MATRIX, desc='Evaluating combos'):
    correct, total = 0, 0
    wrong = np.zeros(33, int);
    count = np.zeros(33, int)
    for idx in range(num_samples):
        # stack features
        views = [load_feats(idx, p) for p in comb]
        stack = torch.stack(views, dim=0).to(device)
        # max pool
        feat = stack.max(dim=0)[0].view(1, -1)
        # classify
        pred = org.net_2(feat).argmax(dim=1).item()
        # true label from filename or separate array
        true = ... # load true label mapping
        total += 1; count[true] +=1
        if pred != true: wrong[true] +=1
        else: correct+=1
    mean_acc = np.mean((count - wrong)/count)
    with open(LOG_CSV,'a') as f:
        f.write(','.join(map(str,comb)) + f',{mean_acc}\n')

print('Evaluation complete.')