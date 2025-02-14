from inference import Inference
from models import Models
from pruning import Pruning

def main():
    model_sel = "[0] VGG16 \n[1] VGG19 \n[2] ResNet50 \n[3] MobileNetV2 \nSelect model: "
    model_sel = input(model_sel)
    
    model = Models().get_model(model_sel)
    
    opt = int(input("[0] Prune and Save \n[1] Prune and Run Inference"))
    if opt == 0:
        model = Pruning().prune_selection(model, 0.5)
        Models().save_model(model, model_sel)
    elif opt == 1:
        prune = Pruning()
        model = prune.prune_model(model, 0.5)
        x_test, y_test = Inference().load_imagenet()
        Inference().run_inference(model, x_test, y_test)
    else:
        print("Invalid option")
    
    print("Done")
    
if __name__ == "__main__":
    main()