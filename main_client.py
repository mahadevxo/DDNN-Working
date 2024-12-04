import sys
import time
from threading import Thread
import queue

data_queue = queue.Queue()

def send_data(client, data):
    try:
        client.send_data(data)
    except Exception as e:
        print(f"Error sendind data: {e}")
    if data == 'exit':
        try:
            client.close_connection()
        except Exception as e:
            print(f"Error closing connection: {e}")
        
def worker(client):
    while True:
        item = data_queue.get()
        if item is None:
            break
        send_data(client, item)
        data_queue.task_done()
        
        
def main():
    thread = Thread(target=worker, args=(client,))
    thread.start()
    
    address =['100.86.4.56', 4044]
    
    from JetsonClient import JetsonClient
    client = JetsonClient(address[0], int(address[1]))
    
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
    
    try:
        with no_grad():
            for image in images:
                start_time = time.time()
                pred = svcnn(image.unsqueeze(0))
                end_time = time.time()
                
                time_process = end_time - start_time
                
                send_time = time.time()
                
                data = f"{send_time}|{time_process}|{pred}"
                
                data_queue.put(data)
    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        data_queue.put('exit')
        thread.join()
    
    print("Inference Done")

if __name__ == "__main__":
    main()
