import tensorflow as tf
class Inference:
    def __init__(self):
        print("Inference class initialized")
        
    def load_imagenet(self):
        print("Loading ImageNet")
        img_height = 224
        img_width = 224
        return tf.keras.utils.image_dataset_from_directory(
            'imagenet-sample-images/',
            image_size=(img_height, img_width),
            batch_size=32,
        )
    
    def run_inference(self, model, x_test, y_test):
        print("Running inference")
        # Run inference on the 100 images
        accuracy = model.evaluate(x_test, y_test)
        print("Accuracy: ", accuracy)
        print("Inference completed")