from saved_models.vgg16_fcn.svhn10vgg import *
from saved_models.vgg16_fcn.cifar10vgg import cifar10vgg
from tensorflow.keras.models import Model, load_model

SAVED_MODELS_DIR = "./saved_models/"

def load_custom_model_for_ds(in_ds_name, model_type):
    model = None
    #**********************************************************
    if  in_ds_name=="MNIST" and model_type=="LeNet":
        model = load_model(SAVED_MODELS_DIR+'LeNet.h5')
        if not model is None:
            print('The weights of LeNet model was loaded.')
    #**********************************************************
    elif in_ds_name=="CIFAR10" and model_type=="VGG":
        model_handle = cifar10vgg(False)
        model = model_handle.get_model()
        if not model is None:
            print('The weights of CIFAR10-VGG16 model was loaded.')
        #**********************************************************
    elif in_ds_name=="SVHN" and model_type=="VGG":
        model_handle = svhn10vgg(False)
        model = model_handle.get_model()
        if not model is None:
            print('The weights of SVHN-VGG16 model was loaded.')
    #**********************************************************
    elif in_ds_name=="CIFAR10" and model_type=="ResNet":
        model = load_model(SAVED_MODELS_DIR+'cifar10_ResNet44v1_model.138.h5')
        if not model is None:
            print('The ResNet-V1-44 model for CIFAR10 was loaded.')   
    
    elif in_ds_name=="SVHN" and model_type=="ResNet":
        model = load_model(SAVED_MODELS_DIR+'svhn_resnet.h5')
        if not model is None:
            print('The ResNet-V1-20 model for SVHN was loaded.')
            
    return model 

