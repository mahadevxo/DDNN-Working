# DDNN Working Directory

```md
.
├── README.md
├── data
│   └── transmission
│       ├── data.csv
│       ├── data.numbers
│       └── data.pdf
├── dev
│   ├── onnx_convert.py
│   └── test.py
├── get_images.py
├── jetson_socket
│   ├── JetsonClient.py
│   ├── MacServer.py
│   ├── load_model.py
│   ├── main_client.py
│   ├── main_client_no_thread.py
│   ├── main_server.py
│   ├── models
│   │   ├── MVCNN.py
│   │   ├── Model.py
│   │   └── __init__.py
│   └── preprocess_images.py
└── mvcnn_training
    ├── models
    │   ├── MVCNN.py
    │   ├── Model.py
    │   └── __init__.py
    ├── tools
    │   ├── ImgDataset.py
    │   ├── Trainer.py
    │   └── __init__.py
    └── train_mvcnn_resnet18.py

9 directories, 24 files


check data/ for results
    transmission/
        results from transmission delay testing
        data.pdf with data visualized



jetson_socket/
    run main_client.py on jetson for multithreaded process
    run main_client_no_thread.py for single threaded
    preprocess image takes value 0-400; default: 400



get_images.py
    gets 10 random images from */test
    saves it to test_set/ for inference
```
