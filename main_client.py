import sys
import time

def send_data(client, data):
    client.send_data(data)
    if data == 'exit':
        client.close_connection()
        
def main():    
    address = input("Enter Server Address: ").strip().split(':')
    
    from JetsonClient import JetsonClient
    client = JetsonClient(address[0], address[1])
    
    sys.modules.pop('JetsonClient')
    del JetsonClient
    
    try:
        client.connect_to_server()
    except:
        print("Error")
    
    path_to_weights = input("Enter Path to weights: ").strip()
    
    from load_model import load_model
    svcnn = load_model(path_to_weights)
    
    sys.modules.pop('load_model')
    del load_model
    
    print("Model Loaded")
    
    from preprocess_images import preprocess_images
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
