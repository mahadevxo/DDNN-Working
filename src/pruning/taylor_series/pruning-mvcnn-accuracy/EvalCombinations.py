import sys
sys.path.append('../../../MVCNN/')
from models import MVCNN
import os
import numpy as np
import torch
from tqdm import tqdm

# Configuration
class EvalCombinations:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FEATURE_DIR = './cached-features'
        self.PRUNING_AMOUNTS = np.arange(0.0, 1.02, 0.02).tolist()
        self.NUM_VIEWS = 12
        self.NUM_COMBOS = 2000
        self.LOG_CSV = 'test-log.csv'
        self.P_MATRIX = np.random.choice(self.PRUNING_AMOUNTS, size=(self.NUM_COMBOS, self.NUM_VIEWS))

# Load original model net_2
    def load_org_net2(self):
        base = MVCNN.SVCNN(name='svcnn', nclasses=33, cnn_name='vgg11')
        weights = torch.load('../../../MVCNN/MVCNN/MVCNN/model-00050.pth', map_location=self.device)
        base.load_state_dict(weights)
        mvcnn = MVCNN.MVCNN(name='mvcnn', model=base.to(self.device), num_views=self.NUM_VIEWS, cnn_name='vgg11')
        return mvcnn.net_2.eval()
    
    def load_feats(self, sample_idx, prune_amount):
        fname = f'sample_{sample_idx}_prune_{prune_amount}_feats.npy'
        path = os.path.join(self.FEATURE_DIR, fname)
        return torch.from_numpy(np.load(path))  # shape [NUM_VIEWS,512,7,7]

    def load_label(self, sample_idx):
        fname = f'sample_{sample_idx}_label.npy'
        path = os.path.join(self.FEATURE_DIR, fname)
        return int(np.load(path))
    
    def run(self):
        net_2 = self.load_org_net2().to(self.device)
        all_files = os.listdir(self.FEATURE_DIR)
        sample_ids = sorted(set(int(f.split('_')[1]) for f in all_files if f.endswith('_feats.npy')))
        self.NUM_SAMPLES = len(sample_ids)

        # Prepare CSV
        with open(self.LOG_CSV, 'w') as f:
            header = ','.join(f'prune_v{i}' for i in range(self.NUM_VIEWS)) + ',mean_class_acc\n'
            f.write(header)

        # Evaluate
        for combo in tqdm(self.P_MATRIX, desc='Evaluating combos'):
            wrong = np.zeros(33, int)
            count = np.zeros(33, int)

            for idx in sample_ids:
                # load and stack features
                views = [self.load_feats(idx, p)[v] for v, p in enumerate(combo)]
                stack = torch.stack(views, dim=0).to(self.device)  # [V,512,7,7]
                pooled = stack.max(dim=0)[0].view(1, -1)      # [1,512*7*7]
                pred = net_2(pooled).argmax(dim=1).item()

                true = self.load_label(idx)
                count[true] += 1
                if pred != true:
                    wrong[true] += 1

            mean_acc = np.mean((count - wrong) / count)
            # log
            with open(self.LOG_CSV, 'a') as f:
                f.write(','.join(map(str, combo.tolist())) + f',{mean_acc}\n')

        print('Evaluation complete.')