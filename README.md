# DDNN Working Directory

## Directory Structure

```
DDNN-Working
├── initialise_remote_machine.sh
├── README.md
└── src
    ├── MVCNN
    │   ├── __init__.py
    │   ├── models
    │   │   ├── __init__.py
    │   │   ├── Model.py
    │   │   └── MVCNN.py
    │   ├── MVCNN
    │   │   └── config.json
    │   ├── reorganise.py
    │   ├── tools
    │   │   ├── __init__.py
    │   │   ├── ImgDataset.py
    │   │   └── Trainer.py
    │   └── train_mvcnn.py
    ├── pruning
    │   ├── RL
    │   │   ├── ComprehensiveVGGPruner.py
    │   │   ├── diagrams.pu
    │   │   ├── GetAccuracy.py
    │   │   └── PPO.py
    │   └── taylor_series
    │       ├── FilterPruner.py
    │       ├── main.py
    │       ├── Pruning.py
    │       ├── PruningFineTuner.py
    │       ├── RewardFuntion.py
    │       ├── Search.py
    │       ├── SearchAlgorithm.py
    │       └── tests
    │           ├── create_data.ipynb
    │           ├── finding_relation.ipynb
    │           ├── local_test.py
    │           ├── multi_model_test.py
    │           ├── plot_results.ipynb
    │           ├── pre_vs_post_finetune.ipynb
    │           ├── test.py
    │           ├── testing_alexnet_jetson.py
    │           └── testing.ipynb
    └── tests
        ├── inference_time_vgg
        │   ├── vgg16_cpu.csv
        │   ├── vgg16_cuda.csv
        │   ├── vgg16_cuda.pdf
        │   ├── vgg16_features_inference.py
        │   ├── vgg16.csv
        │   └── vgg16.pdf
        ├── MVCNN
        │   └── MVCNN_delays
        └── MVCNN_delays
            ├── jetson_transmission.py
            ├── jetson.py
            ├── JetsonClient.py
            ├── mac_transmission.py
            ├── mac.py
            ├── MacServer.py
            ├── models
            │   ├── __init__.py
            │   ├── Model.py
            │   └── MVCNN.py
            └── pruning.py

16 directories, 48 files
```

---

### src/MVCNN

MVCNN contains the original MVCNN code which was used for training the model. The code was adjusted to account for the newer pytorch versions.

---

### src/pruning/RL

Tried a PPO approach to find the optimal pruning amount. Took too much time and was not feasible. So this method was abandoned.

---

### src/pruning/taylor_series

Contains the code for pruning the model using the taylor series approximation. This also adds a search algorithm based on gradient ascend.

#### ./tests

Consists of random tests that i conducted
