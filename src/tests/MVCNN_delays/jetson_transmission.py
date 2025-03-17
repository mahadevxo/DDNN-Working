from JetsonClient import JetsonClient
import time
import torch
from models import MVCNN
import base64

def main():
    address = ['100.86.4.56', 4044]
    client = JetsonClient(address[0], int(address[1]))
    svcnn = MVCNN.SVCNN('svcnn')
    svcnn.load('MVCNN-Jetson_stage_1/')
    
    try:
        client.connect_to_server()
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return
    
    images = []
    for _ in range(100):
        image = torch.rand(1, 3, 224, 224)
        images.append(image)
    
    for image in images:
        with torch.no_grad():
            feature_map = svcnn.net_1(image)
            
            print(feature_map.shape)
            
            feature_map = feature_map.numpy()
            feature_map = base64.b64encode(feature_map).decode('utf-8')
            
            # Ensure the feature map is correctly padded
            missing_padding = len(feature_map) % 4
            if missing_padding:
                feature_map += "=" * (4 - missing_padding)
            
            print(len(feature_map))
            
            print(f"new len {len(feature_map)}")
            
            send_time = time.time()
            client.send_data(f"{send_time}|{feature_map}")
    
    print("Done")
    client.close_connection()
    
if __name__ == "__main__":
    main()