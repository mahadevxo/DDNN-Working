# DDNN Working Directory

## Directory Structure

```text
.
├── initialise_remote_machine.sh
├── README.md
└── src
    ├── MVCNN							[1]
    │   ├── convert_multi.py
    │   ├── convert.py
    │   ├── models
    │   │   ├── Model.py
    │   │   └── MVCNN.py
    │   ├── MVCNN
    │   │   └── model-mvcnn-00050.pth
    │   ├── pytorch3d_render.py
    │   ├── reorganise.py
    │   ├── SVCNN
    │   │   └── model-svcnn-00050.pth
    │   ├── tools
    │   │   ├── ImgDataset.py
    │   │   └── Trainer.py
    │   └── train_mvcnn.py
    └── pruning							[2]
        ├── RL							[3]
        │   ├── ComprehensiveVGGPruner.py
        │   ├── diagrams.pu
        │   ├── GetAccuracy.py
        │   └── PPO.py
        └── taylor_series					[4]
            ├── incremental-pruning				[5]
            │   ├── FilterPruner.py
            │   ├── importance.py
            │   ├── main_mvcnn.py
            │   ├── main-general-cnns.py
            │   ├── PFT_MVCNN.py
            │   ├── PFT.py
            │   ├── Pruning.py
            │   ├── reorganise-modelnet40.py
            │   ├── results
            │   │   ├── MVCNN Results.prism
            │   │   └── pruning-results-20250702-124835.csv
            │   └── Rewards.py
            ├── intermediate-tests-results
            │   ├── computation_time_alexnet.ipynb
            │   ├── intermediate-tests
            │   │   ├── FilterPruner.py
            │   │   ├── main.py
            │   │   ├── model_info.ipynb
            │   │   ├── MVCNN
            │   │   │   ├── convert_multi.py
            │   │   │   ├── convert.py
            │   │   │   ├── models
            │   │   │   │   ├── Model.py
            │   │   │   │   └── MVCNN.py
            │   │   │   ├── pytorch3d_render.py
            │   │   │   ├── reorganise.py
            │   │   │   ├── testing.ipynb
            │   │   │   ├── tools
            │   │   │   │   ├── ImgDataset.py
            │   │   │   │   └── Trainer.py
            │   │   │   ├── train_mvcnn.py
            │   │   │   └── trained-models
            │   │   │       ├── MVCNN
            │   │   │       └── SVCNN
            │   │   ├── PFT.py
            │   │   ├── Pruning.py
            │   │   ├── pytorch-pruning
            │   │   │   ├── dataset.py
            │   │   │   ├── finetune.py
            │   │   │   ├── prune.py
            │   │   │   └── README.md
            │   │   ├── reorganise.py
            │   │   ├── res1.csv
            │   │   ├── results_again.csv
            │   │   ├── results_mvcnn.csv
            │   │   ├── results_vgg11.csv
            │   │   └── tests.py
            │   ├── LearnData.ipynb
            │   ├── pre_accuracy-post_accuracy_relation.ipynb
            │   ├── pre_vs_post_finetune.ipynb
            │   ├── Rewards.ipynb
            │   └── VGG11_final_results.csv
            ├── mvcnn-optimization				[6]
            │   ├── flowchart.mmd
            │   ├── mvcnn-optimization-v1.ipynb
            │   ├── mvcnn-optimization-v2.ipynb
            │   └── view-importance-calculator.ipynb
            └── searching-algorithm-v1				[7]
                ├── curve-finding.ipynb
                ├── FilterPruner.py
                ├── infogetter.py
                ├── models
                │   ├── Model.py
                │   └── MVCNN.py
                ├── PFT.py
                ├── Pruning.py
                ├── reorganise.py
                ├── Rewards.py
                ├── searchAlgo.py
                ├── sorted_view_importance_scores.csv
                ├── tools
                │   ├── ImgDataset.py
                │   └── Trainer.py
                └── viewImportance.ipynb

25 directories, 77 files

```

1. Original MVCNN implimentation [https://github.com/jongchyisu/mvcnn_pytorch](https://github.com/jongchyisu/mvcnn_pytorch)
2. Pruning Experiments
3. Reinforement Based Approach (not feasible)
4. Using Taylor Series Approximation to Find Prunable Filters
5. Prune Models Incrementally rather than All at Once
6. **Final Optimization Algorithm**
7. V1 Searching Algorithm

---

## Final Algorithm

Final algorithm is in 

---
