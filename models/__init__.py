from .vanilla_torch_models import resnet18v, resnet34v, resnet50v, ViT_B_16, ViT_L_16
from .vanilla_torch_models import efficientnet_b0v, efficientnet_b7v, mobilenet_v2v, densenet121v, regnet_y_400mfv
                                


model_dict = {

    'resnet18': resnet18v,
    'resnet34': resnet34v,
    'resnet50': resnet50v,
    'vit_b16': ViT_B_16,
    'vit_l16': ViT_L_16,
    'efficientnet_b0': efficientnet_b0v,
    'efficientnet_b7': efficientnet_b7v,
    'mobilenet_v2': mobilenet_v2v,
    'densenet121': densenet121v,
    'regnet_y_400mf': regnet_y_400mfv
    }


