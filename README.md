# DDNN Working Directory
.
├── README.md
├── data.csv
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

7 directories, 22 files

check data.csv for results
    send_time in UTC
    time_process in UTC
    image_count
    time_received in UTC

jetson_socket/
    run main_client.py on jetson for multithreaded process
    run main_client_no_thread.py for single threaded
    preprocess image takes value 0-400; default: 400

get_images.py
    gets 10 random images from */test
    saves it to test_set/ for inference