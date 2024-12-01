from torch import device
from torch import load
from torch import cuda
from models import MVCNN

def load_model(weights_path):
    device = device("cuda" if cuda.is_available() else "cpu")
    
    svcnn_model = MVCNN.SVCNN('svcnn').to(device)
    svcnn_weights = load(weights_path)
    
    svcnn_model.load_state_dict(svcnn_weights)
    
    return svcnn_model

if __name__ == "__main__":
    path = input("enter path: ").strip()
    svcnn_model = load_model(path)
    
    print(svcnn_model)