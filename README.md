# DDNN Working Directory

```md
.
├── README.md
├── data
│   └── processing_transmission
│       ├── no tensor
│       │   ├── data.numbers
│       │   │   ├── Data
│       │   │   ├── Index.zip
│       │   │   ├── Metadata
│       │   │   │   ├── BuildVersionHistory.plist
│       │   │   │   ├── DocumentIdentifier
│       │   │   │   └── Properties.plist
│       │   │   ├── preview-micro.jpg
│       │   │   ├── preview-web.jpg
│       │   │   └── preview.jpg
│       │   ├── data_processing_transmission.pdf
│       │   ├── processing time
│       │   │   ├── data.csv
│       │   │   └── processing_time.pdf
│       │   └── transmission
│       │       ├── data.csv
│       │       ├── data.numbers
│       │       └── data.pdf
│       └── with tensor
│           ├── tranmission_processing_delay.numbers
│           └── tranmission_processing_delay.pdf
├── dev
│   ├── get_images.py
│   ├── onnx_convert.py
│   └── test.py
├── jetson_socket
│   ├── JetsonClient.py
│   ├── MacServer.py
│   ├── __pycache__
│   │   └── MacServer.cpython-310.pyc
│   ├── load_model.py
│   ├── main_client.py
│   ├── main_client_no_thread.py
│   ├── main_server.py
│   ├── models
│   │   ├── MVCNN.py
│   │   ├── Model.py
│   │   └── __init__.py
│   ├── preprocess_images.py
│   └── tranmission_processing_delay.xlsx
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

17 directories, 38 files


check data/ for results
    processing_transmission
        without sending tensor
            data_processing_transmission.pdf
            plotted in pdf
        sending tensors
            tranmission_processing_delay.pdf
            plotted in pdf



jetson_socket/
    run main_client.py on jetson for multithreaded process
    run main_client_no_thread.py for single threaded
    preprocess image takes value 0-400; default: 400



get_images.py
    gets 10 random images from */test
    saves it to test_set/ for inference
```
