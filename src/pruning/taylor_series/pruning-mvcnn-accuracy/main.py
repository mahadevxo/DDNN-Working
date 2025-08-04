import sys
sys.path.append('../../../MVCNN/')
from models import MVCNN
import torch
import numpy as np
import os
from tqdm import tqdm
import time
from tools.ImgDataset import MultiviewImgDataset

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRUNING_AMOUNTS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]).tolist()
NUM_VIEWS = 12
BATCH_SIZE = 1

T = time.time()
with open(f"test-log-{T}.csv", 'w') as f:
    header = ",".join(
        f"prune_amount_view_{i}" for i in range(NUM_VIEWS)
        ) + ",mean_class_acc\n"
    f.write(header)

def save_model(model, prune_amount):
    save_path = f'./pruned-models/pruned_mvcnn_{str(prune_amount)}.pth'
    scripted_model = torch.jit.script(model)
    scripted_model.save(save_path)
    print(f'Model saved to {save_path}')

def prune_and_train(model, prune_amount, epochs=3):
    from PFT_MVCNN import PruningFineTuner as pft
    pruner = pft(model, quiet=False)
    
    total_filters = pruner.total_num_filters()
    filters_to_prune = int(total_filters * prune_amount)
    print(f'Total Filters: {total_filters}')

    print(f'Filters to Prune: {filters_to_prune}')
    
    _ = pruner.prune(pruning_amount=prune_amount, num_filters_to_prune=filters_to_prune)
    new_filter_count = pruner.total_num_filters()
    print(f'New Filter Count: {new_filter_count}')
    
    print("Retraining the model...")
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
    save_model(pruner.model, prune_amount)

def get_org_model():
    base_cnn = MVCNN.SVCNN(
        name='svcnn',
        nclasses=33,
        cnn_name='vgg11'
    )
    weights = torch.load('../../../MVCNN/MVCNN/MVCNN/mvcnn-00050.pth', map_location=device)
    base_cnn.load_state_dict(weights)
    base_cnn = base_cnn.to(device)
    
    mvcnn = MVCNN.MVCNN(
        name='mvcnn',
        model=base_cnn,
        num_views=12,
        cnn_name='vgg11',)
    
    return mvcnn.to(device)

def prune_save_model():
    for prune_amount in PRUNING_AMOUNTS:
        print(f"Pruning amount: {prune_amount}")
        mvcnn = get_org_model()
        prune_and_train(mvcnn, prune_amount)

def load_models():
    models = {}
    for prune_amount in PRUNING_AMOUNTS:
        model_path = f'./pruned-models/pruned_mvcnn_{prune_amount}.pth'
        if os.path.exists(model_path):
            models[str(prune_amount)] = torch.jit.load(model_path, map_location=device)
            print(f"Loaded model for pruning amount {prune_amount}")
        else:
            print(f"Model for pruning amount {prune_amount} not found.")
    return models

def save_to_csv(p, mean_class_acc):
    with open(f"test-log-{T}.csv", 'a') as f:
        f.write(','.join(map(str, p)) + f',{mean_class_acc}\n')

def test_mvcnn():
    models = load_models()
    org_model = get_org_model()
    test_path = '../../../MVCNN/ModelNet40-12View/*/test'
    dataset = MultiviewImgDataset(
        root_dir=test_path,
        num_views=12,
        shuffle=False,
        num_models=0,
        test_mode=True)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,)
    
    P_matrix = np.random.choice(PRUNING_AMOUNTS, size=(1000, 12))
    #views are as 0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9
    with torch.no_grad():
        for P in P_matrix:
            all_correct = 0
            all_samples = 0
            wrong_class = np.zeros(33, dtype=int)
            samples_class = np.zeros(33, dtype=int)
            pbar = tqdm(test_loader, desc="Testing Models")

            for batch_idx, data in enumerate(pbar):
                y = torch.zeros((NUM_VIEWS, 512, 7, 7)).to(device)
                label = data[0].to(device)
                for dataidx, p in enumerate(P):
                    model = models[str(p)]
                    model = model.to(device)
            
                    images = data[1][0][dataidx].to(device)
                    
                    x = images.unsqueeze(0).to(device)
                    
                    net_1_out = model.net_1(x)
                    
                    y[dataidx] = net_1_out.squeeze(0).to(device)

                y = y.view((int((BATCH_SIZE*NUM_VIEWS)/NUM_VIEWS),NUM_VIEWS,y.shape[-3],y.shape[-2],y.shape[-1]))
                y = torch.max(y, dim=1)[0].view(y.shape[0], -1)
                preds = org_model.net_2(y)
                preds = preds.argmax(dim=1)     
                
                batch_correct = (preds == data[0].to(device)).sum().item()
                all_correct += batch_correct
                all_samples += BATCH_SIZE
                
                label = label.item()
                
                samples_class[label] += 1
                if preds.item() != label:
                    wrong_class[label] += 1

            mean_class_acc = np.mean((samples_class - wrong_class) / samples_class)
            save_to_csv(P, mean_class_acc)
            print(f"Pruning Amount: {P}, Mean Class Accuracy: {mean_class_acc:.4f}")

if __name__ == "__main__":
    whattodo = int(input("Enter 1 to prune and save models and then test, 2 to test models: "))
    if whattodo == 1:
        prune_save_model()
        test_mvcnn()
        print("Pruning and testing completed.")
    else:
        test_mvcnn()
        print("Testing completed.")

    print(f"Took {(time.time() - T)/3600} hours to run the script.")