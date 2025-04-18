import torch
from copy import deepcopy
from MVCNN_Trainer import MVCNN_Trainer
from PruningFineTuner import PruningFineTuner
from MVCNN.models.MVCNN import SVCNN

def init_csv():
    csv_filename = 'pruning_results.csv'
    with open(csv_filename, 'w', newline='') as file:
        file.write("pruning_percentage, pre_acc, post_acc\n")
    print(f"Results file created: {csv_filename} and initialized")
    
        
def write_to_csv(pruning_percentage, pre_acc, post_acc):
    csv_filename = 'pruning_results.csv'
    with open(csv_filename, 'a') as file:
        file.write(f"{pruning_percentage},{pre_acc},{post_acc}\n")

def main():
    # Load the original model (assumed saved as 'model.pth')
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_original = SVCNN('svcnn')
    model_original.load_state_dict(torch.load('model-00030.pth', map_location='cuda'))
    model_original.to(device)
    init_csv()
        # Loop pruning percentage from 0 to 100 (step 1)
    for pruning_percentage in range(101):
        # Use a fresh copy for each pruning percentage
        model_copy = deepcopy(model_original)
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=0.001)
        trainer = MVCNN_Trainer(optimizer, train_amt=0.1, test_amt=0.5)
        
        # Evaluate pre-fine-tuning accuracy
        pre_acc = trainer.get_val_accuracy(deepcopy(model_copy))[1]
        
        # Prune and fine tune the model with the given pruning percentage
        pft = PruningFineTuner(deepcopy(model_copy), test_amt=0.1, train_amt=0.5)
        pft.prune(pruning_percentage=pruning_percentage)
        
        # Evaluate post-fine-tuning accuracy
        post_acc = trainer.get_val_accuracy(pft.model)[1]
        
        write_to_csv(pruning_percentage, pre_acc, post_acc)
        print(f"Pruning {pruning_percentage}: pre_acc = {pre_acc}, post_acc = {post_acc}")
        del trainer
    print("We done")


if __name__ == '__main__':
    main()