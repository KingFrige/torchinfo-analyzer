import torch
from .yolof.build import build_yolof
from .fcos.build import build_fcos
from .retinanet.build import build_retinanet


# build object detector
def build_model(version, 
                cfg, 
                device, 
                num_classes=80, 
                trainable=False, 
                pretrained=None,
                eval_mode=False):
    print('==============================')
    print('Build {} ...'.format(version.upper()))

    if version in ['yolof-r18', 'yolof-r50', 'yolof-r50-DC5',
                        'yolof-rt-r50', 'yolof-r101', 'yolof-r101-DC5']:
        return build_yolof(version, cfg, device, num_classes, trainable, pretrained, eval_mode)

    elif version in ['fcos-r18', 'fcos-r50', 'fcos-r101', 'fcos-rt-r18', 'fcos-rt-r50']:
        return build_fcos(version, cfg, device, num_classes, trainable, pretrained, eval_mode)

    elif version in ['retinanet-r18', 'retinanet-r50', 'retinanet-r101',
                          'retinanet-rt-r18', 'retinanet-rt-r50']:
        return build_retinanet(version, cfg, device, num_classes, trainable, pretrained, eval_mode)
