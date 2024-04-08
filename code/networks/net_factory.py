from networks.unet_3D import unet_3D
from networks.unet import *

def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "attention_unet":
        net = unet_3D(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
    return net
