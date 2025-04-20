import sys
import torch
from PruningFineTuner import PruningFineTuner
sys.path.append('../../MVCNN')
from models import MVCNN

svcnn_weights = torch.load('../../MVCNN/trained-models/MVCNN/model-00030.pth', weights_only=True, map_location='mps')
svcnn = MVCNN.SVCNN('svcnn')
svcnn.load_state_dict(svcnn_weights, strict=False)


pruner = PruningFineTuner(model=svcnn, train_amt=0.001, test_amt=0.01)
ranks = pruner.prune(rank_filters=True)


print(ranks)