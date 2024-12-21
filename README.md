# DDNN Working Directory

### Directory Structure

```md
.
├── README.md
├── data
│   └── processing_transmission
│       ├── multithread
│       │   ├── processing time.png
│       │   ├── tranmission_processing_delay.numbers
│       │   │   ├── Data
│       │   │   ├── Index.zip
│       │   │   ├── Metadata
│       │   │   │   ├── BuildVersionHistory.plist
│       │   │   │   ├── DocumentIdentifier
│       │   │   │   └── Properties.plist
│       │   │   ├── preview-micro.jpg
│       │   │   ├── preview-web.jpg
│       │   │   └── preview.jpg
│       │   ├── tranmission_processing_delay.xlsx
│       │   └── transmission time.png
│       ├── single_thread_no_tensor
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
│       └── single_thread_tensor
│           ├── single_thread.png
│           ├── tranmission_processing_delay.numbers
│           │   ├── Data
│           │   ├── Index.zip
│           │   ├── Metadata
│           │   │   ├── BuildVersionHistory.plist
│           │   │   ├── DocumentIdentifier
│           │   │   └── Properties.plist
│           │   ├── preview-micro.jpg
│           │   ├── preview-web.jpg
│           │   └── preview.jpg
│           └── tranmission_processing_delay.pdf
├── dev
│   ├── get_images.py
│   ├── onnx_convert.py
│   ├── test.py
│   └── untitled folder
├── jetson_socket
│   ├── JetsonClient.py
│   ├── MacServer.py
│   ├── __pycache__
│   │   ├── MacServer.cpython-310.pyc
│   │   └── MacServer.cpython-313.pyc
│   ├── load_model.py
│   ├── main_client_multi_thread.py
│   ├── main_client_single_thread.py
│   ├── main_server.py
│   ├── models
│   │   ├── MVCNN.py
│   │   ├── Model.py
│   │   └── __init__.py
│   ├── preprocess_images.py
│   └── tranmission_processing_delay1733547068.598926.xlsx
├── models_trained
│   ├── MVCNN-Jetson
│   │   ├── model-00000.pth
│   │   ├── model-00001.pth
│   │   ├── model-00002.pth
│   │   └── model-00004.pth
│   └── SVCNN-Jetson
│       ├── model-00000.pth
│       ├── model-00001.pth
│       ├── model-00007.pth
│       └── model-00008.pth
├── mvcnn_training
│   ├── models
│   │   ├── MVCNN.py
│   │   ├── Model.py
│   │   └── __init__.py
│   ├── tools
│   │   ├── ImgDataset.py
│   │   ├── Trainer.py
│   │   └── __init__.py
│   └── train_mvcnn_resnet18.py
└── vgg
    ├── vgg16.csv
    ├── vgg16.pdf
    └── vgg16_features_inference.py

29 directories, 67 files
```

---

### Directory Information

```md
vgg/
    contains vgg16 features infernence code
    contains csv file with results
check data/ for results
    processing_transmission
        without sending tensor
            data_processing_transmission.pdf
            plotted in pdf
        sending tensors
            tranmission_processing_delay.pdf
            plotted in pdf

models_trained/
    SVCNN-Jetson
        Has 4 trained models for SVCNN
    MVCNN-Jetson/
        Has 4 trained mdoels for MVCNN

jetson_socket/
    run main_client.py on jetson for multithreaded process
    run main_client_no_thread.py for single threaded
    preprocess image takes value 0-400; default: 400


get_images.py
    gets 10 random images from */test
    saves it to test_set/ for inference

```
