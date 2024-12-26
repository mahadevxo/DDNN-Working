from MacServer import MacServer
from models import MVCNN
import time
import torch
import numpy as np
import pandas as pd
import base64
import os

def main():
    address = ['100.86.4.56', 4044]
    server = MacServer(address[0], int(address[1]))
    
    svcnn = MVCNN.SVCNN('svcnn')
    svcnn.load('MVCNN-Jetson_stage_2/')
    svcnn.eval()
    
    mvcnn = MVCNN.MVCNN('mvcnn', 'MVCNN-Jetson_stage_2/')
    mvcnn.eval()
    
    

    try:
        server.create_server()
        server.server_listen()
        server.connect()
    except Exception as e:
        print(f"Error setting up server: {e}")
        return
    
    time_rec = []
    
    records = []
    
    count=0
    
    df = pd.DataFrame(records,
                      columns = ['time_svcnn', 'time_sent', 'time_received', 'time_mvcnn', 'class'])
    
    while True:
        data = server.get_data()        
        if data == 'exit':
            print("Closing Server")
            server.close_sockets()
            df = pd.DataFrame(records,
                      columns = ['time_svcnn', 'time_sent', 'time_received', 'time_mvcnn', 'class'])
            df.to_excel("svcnn_mvcnn.xlsx", index=False)
            os._exit(0)
        try:
            data_rev = data.split("|")
            
            time_svcnn = data_rev[0]
            send_time = data_rev[1]
            feature_map = data_rev[2]
            
            time_r = time.time()
            
            feature_map = base64.b64decode(feature_map.encode('utf-8'))
            feature_map = np.frombuffer(feature_map, dtype=np.float32)
            
            pooled_features = torch.max(feature_map, dim=1)[0]
            
            flattened_features = pooled_features.view(pooled_features.shape[0], -1)
            output = mvcnn.net_classifier(flattened_features)
            
            predicted_class = torch.argmax(output, dim=1).item()
            
            time_fin = time.time()
            
            print(f"Predicted Class: {predicted_class}")
            
            records.append([count, time_svcnn, send_time, time_r, time_fin-time_r, predicted_class])
            
            count+=1
        
        except Exception as e:
            print('Error {e}')
        
        
        
        
        
        
        
        