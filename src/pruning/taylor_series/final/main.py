import torch
from Search import Search
from MVCNN.models.MVCNN import SVCNN
def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SVCNN(name='svcnn')
    model_weights = torch.load('./model-00030.pth', map_location=device)
    model.load_state_dict(model_weights, strict=False)
    model = model.to(device)
    
    # Initialize the Search class with the model and other parameters
    search = Search(model, 
                    min_acc=50, 
                    min_size=300, 
                    acc_imp=0.5, 
                    comp_time_imp=0.0, 
                    size_imp=0.5)
    search.adam_gradient()
if __name__ == '__main__':
    main()