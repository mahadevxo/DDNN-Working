import torch
import gc
import os
import pickle
import time

class ModelCache:
    """Cache for models and computations to avoid redundant work"""
    
    def __init__(self, cache_dir='.model_cache'):
        self.cache_dir = cache_dir
        self.memory_cache = {}
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_ranks(self, model_key, compute_fn, *args, **kwargs):
        """Get cached ranks or compute them"""
        cache_path = os.path.join(self.cache_dir, f"{model_key}_ranks.pkl")
        
        # Try to load from disk cache
        if os.path.exists(cache_path):
            try:
                print(f"Loading cached ranks from {cache_path}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        # Compute if not cached
        print("Computing ranks (not found in cache)")
        start_time = time.time()
        ranks = compute_fn(*args, **kwargs)
        print(f"Rank computation took {time.time() - start_time:.2f}s")
        
        # Cache to disk
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(ranks, f)
            print(f"Saved ranks to {cache_path}")
        except Exception as e:
            print(f"Error saving cache: {e}")
            
        return ranks

def optimize_memory_usage():
    """Aggressively optimize memory usage"""
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return
