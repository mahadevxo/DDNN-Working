import sys
sys.path.append('../../../MVCNN/')
from models import MVCNN
import numpy as np
import torch
from tqdm import tqdm

class PruneSaveModel:
    def __init__(self):
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.PRUNING_AMOUNTS = np.arange(0.0, 1.02, 0.02).tolist()
        self.NUM_VIEWS = 12
        self.BATCH_SIZE = 1
        self.VIEW_ORDER = [0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9]

    def save_model(self, model, prune_amount):
        save_path = f'./pruned-models/pruned_mvcnn_{str(prune_amount)}.pth'
        scripted_model = torch.jit.script(model)
        scripted_model.save(save_path)
        # print(f'Model saved to {save_path}')
        

    def prune_and_train(self, model, prune_amount, epochs=2):
        from PFT_MVCNN import PruningFineTuner as pft
        pruner = pft(model, quiet=False)
        
        total_filters = pruner.total_num_filters()
        filters_to_prune = int(total_filters * prune_amount)
        # print(f'Total Filters: {total_filters}')

        # print(f'Filters to Prune: {filters_to_prune}')
        
        _ = pruner.prune(pruning_amount=prune_amount, num_filters_to_prune=filters_to_prune)
        # new_filter_count = pruner.total_num_filters()
        # print(f'New Filter Count: {new_filter_count}')
        
        if prune_amount == 0.0:
            self.save_model(pruner.model, prune_amount)
            return
        
        # print("Retraining the model...")
        optimizer = torch.optim.Adam(pruner.model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        pruner.model.train()
        
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            pruner.train_epoch(optimizer=optimizer, rank_filter=False)
            scheduler.step()
            print(f"Accuracy after epoch {epoch+1}: {pruner.validate_model()}")
        
        print(f"Final Accuracy: {pruner.validate_model()}")
        del model
        self.save_model(pruner.model, prune_amount)

    def get_org_model(self):
        base_cnn = MVCNN.SVCNN(
            name='svcnn',
            nclasses=33,
            cnn_name='vgg11'
        )
        weights = torch.load('../../../MVCNN/MVCNN/MVCNN/model-00050.pth', map_location=self.device)
        base_cnn.load_state_dict(weights)
        base_cnn = base_cnn.to(self.device)
        
        mvcnn = MVCNN.MVCNN(
            name='mvcnn',
            model=base_cnn,
            num_views=12,
            cnn_name='vgg11',)
        
        return mvcnn.to(self.device)

    def prune_save_model(self):
        for prune_amount in tqdm(self.PRUNING_AMOUNTS, desc="Pruning Amounts"):
            print(f"Pruning amount: {prune_amount}")
            mvcnn = self.get_org_model()
            self.prune_and_train(mvcnn, prune_amount)
            print(f"Model pruned and saved for amount: {prune_amount}")
            print("-"*100)

    def run(self):
        self.prune_save_model()
        print("Done")