import time
from threading import Thread
import queue
from JetsonClient import JetsonClient
from load_model import load_model
from preprocess_images import preprocess_images
from torch import no_grad
import base64
import sys

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
    address =['100.86.4.56', 4044]
    client = JetsonClient(address[0], int(address[1]))
    
    thread = Thread(target=worker, args=(client,))
    thread.start()
    
    try:
        client.connect_to_server()
    except:
        print("Error")
    
    path_to_weights = "/home/mahadev/Desktop/DDNN/DDNN-Working/models_trained/SVCNN-Jetson/model-00008.pth"
    
    # _ = input(f"Path to Weights >>> {path_to_weights}")
    
    svcnn = load_model(path_to_weights).to("cuda")
    
    print("Model Loaded")
    
    print("Starting image preprocessing")
    images = preprocess_images()
    print("Image Preprocess Done.")

    svcnn.eval()
    
    print("Starting Inference")
    
    try:
        with no_grad():
            image_count = 0
            for image in images:
                image = image.to("cuda")
                
                start_time = time.time()
                pred = svcnn(image.unsqueeze(0))
                end_time = time.time()
                
                time_process = end_time - start_time
                
                # convert pred to bytes
                pred = pred.cpu().numpy().tobytes()
                pred = base64.b64encode(pred).decode('utf-8')
                
                send_time = time.time()
                
                data = f"{image_count},{pred},{time_process},{send_time}"
                
                if image_count == 3:
                    print(sys.getsizeof(data))
                # print(data)
                image_count += 1
                
                data_queue.put(data)
                
                #print(f"Send Time: {send_time}")
        print("Inference Done")
        data_queue.put(None)
        thread.join()
        exit()
    
    except Exception as e:
        print(f"Error during inference: {e}")
        data_queue.put(None)
        thread.join()
        exit()

if __name__ == "__main__":
    main()
