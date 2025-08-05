import sys
sys.path.append('../../../MVCNN/')
from models import MVCNN
import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
import queue
import threading

class EvalCombinations:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FEATURE_DIR = './cached-features'
        self.PRUNING_AMOUNTS = np.arange(0.0, 1.02, 0.02).tolist()
        self.NUM_VIEWS = 12
        self.NUM_COMBOS = 4000
        self.LOG_CSV = 'test-log.csv'
        self.P_MATRIX = np.random.choice(self.PRUNING_AMOUNTS, size=(self.NUM_COMBOS, self.NUM_VIEWS))
        self.MAX_WORKERS = 8
        self._prepare_model()
        self._cache_file_list()
        self._preload_labels()
        self._cache_features()
        
        # Setup for background file writing
        self.result_queue = queue.Queue()
        self.writing_done = threading.Event()

    def _prepare_model(self):
        base = MVCNN.SVCNN(name='svcnn', nclasses=33, cnn_name='vgg11')
        weights = torch.load('../../../MVCNN/MVCNN/MVCNN/model-00050.pth', map_location=self.device)
        base.load_state_dict(weights)
        mvcnn = MVCNN.MVCNN(name='mvcnn', model=base.to(self.device), num_views=self.NUM_VIEWS, cnn_name='vgg11')
        self.net_2 = mvcnn.net_2.eval().to(self.device)
        
        torch.backends.cudnn.benchmark = True

    def _cache_file_list(self):
        files = os.listdir(self.FEATURE_DIR)
        self.sample_ids = sorted({int(f.split('_')[1]) for f in files if f.endswith('_feats.npy')})
        self.NUM_SAMPLES = len(self.sample_ids)

    def _preload_labels(self):
        self.labels = {}
        for sample_idx in self.sample_ids:
            label_path = os.path.join(self.FEATURE_DIR, f'sample_{sample_idx}_label.npy')
            self.labels[sample_idx] = int(np.load(label_path))
    
    def _cache_features(self):
        self.feature_cache = defaultdict(dict)
        print("Caching features to memory...")
        for sample_idx in tqdm(self.sample_ids, desc="Loading features"):
            for p in self.PRUNING_AMOUNTS:
                path = os.path.join(self.FEATURE_DIR, f'sample_{sample_idx}_prune_{p}_feats.npy')
                if os.path.exists(path):
                    arr = np.load(path)
                    for v in range(self.NUM_VIEWS):
                        if v < arr.shape[0]:
                            self.feature_cache[sample_idx][p] = torch.from_numpy(arr)

    def _process_sample(self, sample_idx, combo):
        views = []
        for v, p in enumerate(combo):
            tensor = self.feature_cache[sample_idx][p]
            views.append(tensor[v])
            
        stack = torch.stack(views, dim=0)
        pooled = stack.max(dim=0)[0].view(1, -1)
        
        # Move to device only when needed
        pooled = pooled.to(self.device)
        pred = self.net_2(pooled).argmax(dim=1).item()
        true = self.labels[sample_idx]
        
        return true, pred
    
    def _file_writer_thread(self):
        """Background thread that writes results to CSV file"""
        with open(self.LOG_CSV, 'a') as f:
            while not self.writing_done.is_set() or not self.result_queue.empty():
                try:
                    # Wait for up to 1 second for new data
                    line = self.result_queue.get(timeout=1)
                    f.write(line)
                    f.flush()  # Ensure data is written to disk
                    self.result_queue.task_done()
                except queue.Empty:
                    continue

    def run(self):
        # Create CSV header
        with open(self.LOG_CSV, 'w') as f:
            header = ','.join(f'prune_v{i}' for i in range(self.NUM_VIEWS)) + ',mean_class_acc\n'
            f.write(header)
        
        # Start background writer thread
        self.writing_done.clear()
        writer_thread = threading.Thread(target=self._file_writer_thread)
        writer_thread.daemon = True
        writer_thread.start()

        with tqdm(total=len(self.P_MATRIX), desc='Evaluating combos') as pbar:
            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                for combo in self.P_MATRIX:
                    wrong = np.zeros(33, int)
                    count = np.zeros(33, int)

                    futures = {
                        executor.submit(self._process_sample, idx, combo): idx
                        for idx in self.sample_ids
                    }
                    for future in as_completed(futures):
                        true, pred = future.result()
                        count[true] += 1
                        if pred != true:
                            wrong[true] += 1

                    mean_acc = np.mean((count - wrong) / count)
                    
                    line = ','.join(map(str, combo.tolist())) + f',{mean_acc}\n'
                    self.result_queue.put(line)
                    
                    pbar.update(1)

        self.writing_done.set()
        writer_thread.join()
        print("All results written to CSV file")