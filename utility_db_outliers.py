import numpy as np
import tensorflow.keras as keras
from utility import get_dataset_hdf5
import scipy.io as sio

OOD_ADRESS = "./ood_datasets/"

def load_dataset(db_name):
    ds_names     = ["MNIST", "FASHION_MNIST", "OMNIGLOT_RESIZED_28", "MNIST_GAUSSIAN_NOISE_ODIN","MNIST_UNIFORM_NOISE",\
                    "CIFAR10", "SVHN", "TINYIMAGENET_RESIZED_32", "LSUN_RESIZED","ISUN_PATCHED", "SVHN_CROPPED",   \
                    "G_255","U_255", "MNIST_LIKE_PERLIN_NOISE", "CIFAR_LIKE_PERLIN_NOISE", "LSUN"]
    train_data, train_labels, test_data, test_labels  = [],[],[],[]
    
    if not db_name in ds_names:
        print("The name of dataset is not valid")
        return None
    
    if db_name=="MNIST":
        (train_data, train_labels),(test_data, test_labels) = __load_mnist()
        
    elif db_name=="CIFAR10":
        (train_data, train_labels),(test_data, test_labels) = __load_cifar10()
        
    elif db_name=="SVHN":
        (train_data, train_labels),(test_data, test_labels) = __load_svhn()
    else:
        (train_data, train_labels),(test_data, test_labels) = \
                                            (None,None), (get_dataset_hdf5("test_data", OOD_ADRESS+db_name),None)             
    
    return (train_data, train_labels),(test_data, test_labels)

def __load_mnist():
    """
    """
    (train_data, train_labels),(test_data, test_labels) = np.asarray(keras.datasets.mnist.load_data())
    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
    test_data  = test_data.reshape(test_data.shape[0], 28, 28, 1)    
    train_data = train_data / 255.0
    test_data = test_data / 255.0    
    return (train_data, train_labels),(test_data, test_labels)


def __load_cifar10():
    """
    """
    (train_data, train_labels),(test_data, test_labels) =np.asarray(keras.datasets.cifar10.load_data())
    train_labels = train_labels[:,0]
    test_labels  = test_labels [:,0]
    return (train_data, train_labels),(test_data, test_labels)

def __load_svhn():
    train = sio.loadmat('./datas/svhn_train.mat')
    test = sio.loadmat('./datas/svhn_test.mat')
    x_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
    x_test = np.transpose(test['X'], axes=[3, 0, 1, 2])

    train_labels = np.reshape(train['y'], (-1,)) - 1
    test_labels = np.reshape(test['y'], (-1,)) - 1

    # cast pixels to floats, normalize to [0, 1] range
    train_data = x_train.astype('float32')
    test_data = x_test.astype('float32')

    return (train_data, train_labels), (test_data, test_labels)