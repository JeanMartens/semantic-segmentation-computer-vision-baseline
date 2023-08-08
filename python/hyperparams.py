import albumentations as A
from torchvision import transforms
import numpy as np
import cv2

class Hyperparams:
    
    #Training Params
    lr = 1e-3
    num_epochs = 2
    batch_size_train = 4
    batch_size_valid = 4
    weight_decay = 1e-5
    
    original_img_shape = (32,32,3)
    img_shape = (64,64,3)


    #Model params
    archi_name = 'Unet'
    encoder_name = "efficientnet-b0"


    #Folds params
    num_splits = 5
    splits_to_train = [1,2,3,4,5]
    splits_to_oof = [1,2,3,4,5]
    
    random_state = 19

    normalise_transform = transforms.Compose([
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    augment_transform = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
    ], p=0.6)


