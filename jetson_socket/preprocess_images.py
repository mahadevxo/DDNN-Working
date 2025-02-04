# SVCNN
def preprocess_images(count_arg=400):

    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    import os
    import PIL.Image as Image

    preprocess = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_folder = '/home/mahadev/Desktop/DDNN/DDNN-Working/test_set'

    count = 0
    test_images = []

    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith('.png'):
                if count == count_arg:
                    break 
                # Load the image
                image_path = os.path.join(root, file)
                image = Image.open(image_path).convert('RGB')
                image = preprocess(image)
                test_images.append(image)

                count += 1

    print(f'Processed {count} images')

    return(test_images)