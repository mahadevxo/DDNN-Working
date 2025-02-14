from tensorflow.keras import applications as apps # type: ignore
class Models:
    def __init__(self):
        print("Models class initialized")
    
    def get_model(self, model_name):
        if model_name == '0':
            return apps.VGG16(
                weights='imagenet',
                include_top=True,
            )
        elif model_name == '1':
            return apps.VGG19(
                weights='imagenet',
                include_top=True,
            )
        elif model_name == '2':
            return apps.resnet50(
                weights='imagenet',
                include_top=True,
            )
        elif model_name == '3':
            return apps.mobilenet_v2(
                weights='imagenet',
                include_top=True,
            )
        else:
            raise ValueError('Invalid model name')
        
    def save_model(self, model, model_name):
        model.save(f'{model_name}.h5')
        print(f"Model saved as {model_name}.h5")