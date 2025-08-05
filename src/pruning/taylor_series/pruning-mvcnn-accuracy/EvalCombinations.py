import sys
sys.path.append('../../../MVCNN/')
from models import MVCNN
import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class EvalCombinations:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FEATURE_DIR = './cached-features'
        self.PRUNING_AMOUNTS = np.arange(0.0, 1.02, 0.02).tolist()
        self.NUM_VIEWS = 12
        self.NUM_COMBOS = 4000
        self.LOG_CSV = 'test-log.csv'
        self.P_MATRIX = np.random.choice(self.PRUNING_AMOUNTS, size=(self.NUM_COMBOS, self.NUM_VIEWS))
        self.MAX_WORKERS = 64
        self._prepare_model()
        self._cache_file_list()

    def _prepare_model(self):
        base = MVCNN.SVCNN(name='svcnn', nclasses=33, cnn_name='vgg11')
        weights = torch.load('../../../MVCNN/MVCNN/MVCNN/model-00050.pth', map_location=self.device)
        base.load_state_dict(weights)
        mvcnn = MVCNN.MVCNN(name='mvcnn', model=base.to(self.device), num_views=self.NUM_VIEWS, cnn_name='vgg11')
        self.net_2 = mvcnn.net_2.eval().to(self.device)

    def _cache_file_list(self):
        files = os.listdir(self.FEATURE_DIR)
        self.sample_ids = sorted({int(f.split('_')[1]) for f in files if f.endswith('_feats.npy')})
        self.NUM_SAMPLES = len(self.sample_ids)

    def _load_feats_and_label(self, sample_idx, combo):
        views = []
        for v, p in enumerate(combo):
            path = os.path.join(self.FEATURE_DIR,
                                f'sample_{sample_idx}_prune_{p}_feats.npy')
            arr = np.load(path)
            views.append(torch.from_numpy(arr)[v])
        stack = torch.stack(views, dim=0).to(self.device)
        pooled = stack.max(dim=0)[0].view(1, -1)
        pred = self.net_2(pooled).argmax(dim=1).item()
        true = int(np.load(os.path.join(self.FEATURE_DIR,
                                         f'sample_{sample_idx}_label.npy')))
        return true, pred

    def run(self):
        with open(self.LOG_CSV, 'w') as f:
            header = ','.join(f'prune_v{i}' for i in range(self.NUM_VIEWS)) + ',mean_class_acc\n'
            f.write(header)

        for combo in tqdm(self.P_MATRIX, desc='Evaluating combos'):
            wrong = np.zeros(33, int)
            count = np.zeros(33, int)

            # parallel sample evaluation
            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                futures = {
                    executor.submit(self._load_feats_and_label, idx, combo): idx
                    for idx in self.sample_ids
                }
                for future in as_completed(futures):
                    true, pred = future.result()
                    count[true] += 1
                    if pred != true:
                        wrong[true] += 1

            mean_acc = np.mean((count - wrong) / count)
            with open(self.LOG_CSV, 'a') as f:
                f.write(','.join(map(str, combo.tolist())) + f',{mean_acc}\n')