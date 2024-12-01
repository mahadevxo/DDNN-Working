import time
import torch
import os
import socket
import glob
from torchvision.models import vgg11

class Model(torch.nn.Module):

    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name


    def save(self, path, epoch=0):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        torch.save(self.state_dict(), 
                os.path.join(complete_path, 
                    "model-{}.pth".format(str(epoch).zfill(5))))


    def save_results(self, path, data):
        raise NotImplementedError("Model subclass must implement this method.")
        

    def load(self, path, modelfile=None):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            raise IOError("{} directory does not exist in {}".format(self.name, path))

        if modelfile is None:
            model_files = glob.glob(complete_path+"/*")
            mf = max(model_files)
        else:
            mf = os.path.join(complete_path, modelfile)

        self.load_state_dict(torch.load(mf))
        
class SVCNN(Model):

    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11'):
        super(SVCNN, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).cuda()
        self.net_1 = vgg11(pretrained=self.pretraining).features
        self.net_2 = vgg11(pretrained=self.pretraining).classifier
            
        self.net_2._modules['6'] = torch.nn.Linear(4096,40)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0],-1))


class MVCNN(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12):
        super(MVCNN, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).cuda()
        self.net_1 = model.net_1
        self.net_2 = model.net_2

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1])) #(8,12,512,7,7)
        return self.net_2(torch.max(y,1)[0].view(y.shape[0],-1))

mean = torch.FloatTensor([0.485, 0.456, 0.406]).cuda()
std = torch.FloatTensor([0.229, 0.224, 0.225]).cuda()

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def load_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    svcnn_model = MVCNN.SVCNN('svcnn').to(device)
    svcnn_weights = torch.load(weights_path)
    
    svcnn_model.load_state_dict(svcnn_weights)
    
    return svcnn_model

class JetsonClient:
    def __init__(self, mac_ip, mac_port):
        self.mac_ip = mac_ip
        self.mac_port = mac_port
        self.jetson_socket = None

    def connect_to_server(self):
        self.jetson_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.jetson_socket.connect((self.mac_ip, self.mac_port))
        print(f"Connected to server at {self.mac_ip}:{self.mac_port}")

    def send_data(self, message):
        if self.jetson_socket:
            self.jetson_socket.send(message.encode())
            print("Data sent")
        else:
            print("Not connected to the server.")

    def close_connection(self):
        if self.jetson_socket:
            self.jetson_socket.close()
            print("Connection closed.")
        else:
            print("No active connection to close.")
            

            
def preprocess_images():

    import torchvision.transforms as transforms
    import os
    import PIL.Image as Image

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        

    image_folder = '/home/mahadev/Desktop/DDNN/work/test_set'

    count = 0
    test_images = []

    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith('.png'):
                # Load the image
                image_path = os.path.join(root, file)
                image = Image.open(image_path).convert('RGB')
                image = preprocess(image)
                test_images.append(image)

                count += 1

    print(f'Processed {count} images')

    return(test_images)


def send_data(client, data):
    client.send_data(data)
    if data == 'exit':
        client.close_connection()
        
def main():
    path  = input("Enter Path to SVCNN Model: ").strip()
    svcnn = load_model(path)
    print("Model Loaded")
    mac_ip = input("Enter Server IP: ").strip()
    mac_port = int(input("Enter Server Port: ").strip())
    
    client = JetsonClient(mac_ip, mac_port)
    
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

    svcnn.eval()
    
    print("Starting Inference")
    
    with torch.no_grad():
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