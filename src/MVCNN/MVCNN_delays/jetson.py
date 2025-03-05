import torch
from models import Model, MVCNN
from JetsonClient import JetsonClient
import time
import os
import base64
import torch

def send_data(client, data):
    try:
        client.send_data(data)
    except Exception as e:
        print(f'error sending data {e}')
    if data=='exit':
        try:
            client.close_connection()
        except Exception as e:
            print(f'Error closing connection {e}')

def main():
    svcnn = MVCNN.SVCNN("svcnn")
    svcnn.load('MVCNN-Jetson_stage_1/')
    svcnn.eval()
    
    images = []
    
    for _ in range(0, 100):
        images.append(torch.randint(3, (244, 244)).float().unsqueeze(0))
        
    address = ['100.68.4.56', 4044]
    client = JetsonClient(address[0], int(address[1]))
    
    try:
        client.connect_to_server()
    except:
        print("Error")
    
    
    for image in images:
        try:
            with torch.no_grad():
                time_svcnn = time.time()
                pred = svcnn.net_features(image)
                pred = pred.cpu().numpy().tobytes()
                pred = base64.b64encode(pred).decode('utf-8')
                send_time = time.time()
                
                data = f"{send_time-time_svcnn}|{send_time}|{pred}"
                
                send_data(data)
            
            print('Inference Done')
            client.send_data('exit')
            client.close_connection()
            print("Close connection")
            os._exit(0)
        except Exception as e:
            print("Error during inference {e}")
    

if __name__ == '__main__':
    main()