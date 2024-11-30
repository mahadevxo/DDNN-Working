import torch
from models import MVCNN

def load_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    svcnn_model = MVCNN.SVCNN('svcnn').to(device)
    svcnn_weights = torch.load(weights_path)
    
    svcnn_model.load_state_dict(svcnn_weights)
    
    return svcnn_model

if __name__ == "__main__":
    path = input("enter path: ").strip()
    svcnn_model = load_model(path)
    
    print(svcnn_model)