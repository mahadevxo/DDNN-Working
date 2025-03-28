import torch
import torch.nn as nn
import os
import glob


class Model(nn.Module):

    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


    def save(self, path, epoch=0):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        torch.save(
            self.state_dict(),
            os.path.join(complete_path, f"model-{str(epoch).zfill(5)}.pth"),
        )


    def save_results(self, path, data):
        raise NotImplementedError("Model subclass must implement this method.")
        

    def load(self, path, modelfile=None):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            raise IOError(f"{self.name} directory does not exist in {path}")

        if modelfile is None:
            model_files = glob.glob(f"{complete_path}/*")
            mf = max(model_files)
        else:
            mf = os.path.join(complete_path, modelfile)

        self.load_state_dict(torch.load(mf))


