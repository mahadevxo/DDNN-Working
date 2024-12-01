from torch import device as tdevice
from torch import load
from torch import cuda as tcuda
from models import MVCNN
import sys

def load_model(weights_path):
    device = tdevice("cuda" if tcuda.is_available() else "cpu")
    
    svcnn_model = MVCNN.SVCNN('svcnn').to(device)
    svcnn_weights = load(weights_path)
    
    sys.modules.pop('MVCNN')
    del MVCNN
    sys.modules.pop('load')
    del load
    sys.modules.pop('cuda')
    del tcuda
    sys.modules.pop('device')
    del tdevice
    
    svcnn_model.load_state_dict(svcnn_weights)
    
    return svcnn_model

if __name__ == "__main__":
    path = input("enter path: ").strip()
    svcnn_model = load_model(path)
    
    print(svcnn_model)