import time
from JetsonClient import JetsonClient
from load_model import load_model
from preprocess_images import preprocess_images
from torch import no_grad

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
        
def main():    
    address =['100.86.4.56', 4044]
    client = JetsonClient(address[0], int(address[1]))
    
    try:
        client.connect_to_server()
    except:
        print("Error")
    
    path_to_weights = "/home/mahadev/Desktop/DDNN/work/SVCNN-Jetson/model-00008.pth"
    
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
                _ = svcnn(image.unsqueeze(0))
                end_time = time.time()
                
                time_process = end_time - start_time
                
                send_time = time.time()
                
                data = f"{send_time},{time_process},{image_count}"
                image_count += 1                
                
                client.send_data(data)
                
                print(f"send time: {send_time}")
                
                
    except Exception as e:
        print(f"Error during inference: {e}")
    
    print("Inference Done")
    client.close_connection()
    exit()

if __name__ == "__main__":
    main()
