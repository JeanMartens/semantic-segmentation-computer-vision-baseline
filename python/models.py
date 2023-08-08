import torch 
import torch.nn as nn
import segmentation_models_pytorch as smp

from hyperparams import Hyperparams

def create_model(accelerate):
    if Hyperparams.archi_name=='Unet':
        model = smp.Unet(encoder_name=Hyperparams.encoder_name,
                                       encoder_weights="imagenet",
                                       in_channels=3,
                                       classes=1,)
        
    if Hyperparams.archi_name=='UnetPlusPlus':
        model = smp.UnetPlusPlus(encoder_name=Hyperparams.encoder_name,
                                       encoder_weights="imagenet",
                                       in_channels=3,
                                       classes=1,)
        
        
    return accelerate.prepare(model)

def model_naming_function(metric_score, epoch, Hyperparams):
    return f'me_{metric_score:.3f}_ep_{epoch}_en_{Hyperparams.encoder_name}_lr_{Hyperparams.lr}_si_{Hyperparams.img_shape[0]}.pt'.replace(",", "" )