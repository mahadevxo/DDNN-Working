from load_model import load_model
from preprocess_images import preprocess_images
from JetsonClient import JetsonClient
import sys

import time

def send_data(client, data):
    client.send_data(data)
    if data == 'exit':
        client.close_connection()
        
def main():
    mac_ip = input("Enter Server IP: ").strip()
    mac_port = int(input("Enter Server Port: ").strip())
    
    client = JetsonClient(mac_ip, mac_port)
    
    sys.modules.pop('JetsonClient')
    del JetsonClient
    
    try:
        client.connect_to_server()
    except:
        print("Error")
    
    path_to_weights = input("Enter Path to weights: ").strip()
    svcnn = load_model(path_to_weights)
    print("Model Loaded")

    print("Starting image preprocessing")
    images = preprocess_images()
    print("Image Preprocess Done.")
    
    sys.modules.pop('preprocess_images')
    del preprocess_images

    svcnn.eval()
    
    print("Starting Inference")
    from torch import no_grad
    with no_grad():
        for image in images:
            start_time = time.time()
            pred = svcnn(image.unqueeze(0))
            end_time = time.time()
            
            time_process = end_time - start_time
            
            send_time = time.time()
            
            data = f"{send_time}|{time_process}|{pred}"
            
            send_data(data)

if __name__ == "__main__":
    main()
