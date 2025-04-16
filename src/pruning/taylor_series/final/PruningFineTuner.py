import torch
import gc
import sys
sys.path.append('../taylor_series/')
sys.path.append('./MVCNN')
from FilterPruner import FilterPruner
from Pruning import Pruning
from MVCNN_Trainer import MVCNN_Trainer

class PruningFineTuner:
    def __init__(self, model, test_amt=0.1, train_amt=0.1):
        self.train_path = './ModelNet40-12View/*/train'
        self.test_path = './ModelNet40-12View/*/test'
        self.num_models = 1000*12
        self.num_views = 12
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pruner = FilterPruner(self.model)
        self.test_amt = test_amt
        self.train_amt = train_amt
        self._clear_memory()
        
    def _clear_memory(self):
        """Helper method to clear memory efficiently"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def get_candidates_to_prune(self, num_filter_to_prune, get_filters=False, mvcnntrainer=None):
        self.pruner.reset()
        # Fix: Pass the existing pruner instance so that ranks accumulate
        mvcnntrainer.train_model(self.model, rank_filter=True, pruner_instance=self.pruner)
        self.pruner.normalize_ranks_per_layer()
        return self.pruner.get_pruning_plan(num_filter_to_prune, get_filters=get_filters)
    
    def total_num_filters(self):
        return sum(
            layer.out_channels
            for layer in self.model.net_1
            if isinstance(layer, torch.nn.Conv2d)
        )
    
    def get_model_size(self, model):
        total_size = sum(
            param.nelement() * param.element_size() for param in model.parameters()
        )
        # Convert to MB
        return total_size / (1024 ** 2)
    
    def get_all_ranked_filters(self):        
        self.model.train()
        for param in self.model.net_1.parameters():
            param.requires_grad = True
        self.train_epoch(rank_filter=True)
        self.pruner.normalize_ranks_per_layer()
        return self.pruner.get_sorted_filters()
    
    def prune(self, pruning_percentage=0, rank_filters=False):  # sourcery skip: extract-method, low-code-quality
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0.0)
        # Enable gradients for pruning
        for param in self.model.net_1.parameters():
            param.requires_grad = True
            
        original_filters = self.total_num_filters()
        mvcnntrainer = MVCNN_Trainer(optimizer, train_amt=self.train_amt, test_amt=self.test_amt)
        
        if rank_filters:
            return self.get_candidates_to_prune(num_filter_to_prune=int(original_filters), get_filters=True, mvcnntrainer=mvcnntrainer)
        
        num_filters_to_prune = int(original_filters * (pruning_percentage / 100.0))
        print(f"Total Filters to prune: {num_filters_to_prune} For Pruning Percentage: {pruning_percentage}")

        # Rank and get the candidates to prune
        prune_targets = self.get_candidates_to_prune(num_filters_to_prune=num_filters_to_prune, get_filters=False, mvcnntrainer=mvcnntrainer)
        print("Pruning targets", prune_targets)
        # Count the number of filters to prune per layer
        layers_pruned = {}
        for layer_index, filter_index in prune_targets:
            layers_pruned[layer_index] = layers_pruned.get(layer_index, 0) + 1
        print("Layers that will be pruned", layers_pruned)
        print("Pruning Filters")
        model = self.model
        pruner = Pruning(model)
        
        # Prune one filter at a time with memory cleanup after each
        for idx, (layer_index, filter_index) in enumerate(prune_targets):
            model = pruner.prune_conv_layers(model, layer_index, filter_index)
            if idx % 5 == 0:  # Clean up every few iterations
                self._clear_memory()

        # Convert model weights to float32 for training stability
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                if layer.weight is not None:
                    layer.weight.data = layer.weight.data.float()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.float()

        self.model = model.to(self.device)
        self._clear_memory()

        # Test and fine tune model
        finetuner = MVCNN_Trainer(optimizer=optimizer)
        model = finetuner.fine_tune(model, rank_filter=False)
    def reset(self):
        """Clear memory resources completely"""
        if hasattr(self, 'pruner'):
            self.pruner.reset()
            del self.pruner

        # Clear model explicitly
        if hasattr(self, 'model'):
            del self.model

        # Clear any other stored objects
        for attr in list(self.__dict__.keys()):
            if attr not in ['device', 'train_path', 'test_path'] and hasattr(self, attr):
                delattr(self, attr)

        # Force garbage collection and clear GPU cache
        self._clear_memory()
        
    def save_model(self, path):
        torch.save(self.model, path)
        print(f"Model saved as {path}")
        
    def __del__(self):
        self.reset()
        print("PruningFineTuner object deleted and memory cleared.")