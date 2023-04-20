#!/bin/env python3

import argparse
import torch
from torchstat import stat
import torchvision.models as models

from model_param_analyzer import Model_param_analyzer
model_param_analyzer_obj = Model_param_analyzer()

def get_classifaction_param():
  from classifaction.MobileNetV1 import MobileNetV1
  model_param_analyzer_obj.get_model_param(model=MobileNetV1(ch_in=3, n_classes=1000), model_name='MobileNetV1', feature_map=(1,3,224,224))

  model_param_analyzer_obj.get_model_param(model=models.MobileNetV2(), model_name='MobileNetV2', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.mobilenet_v3_large(), model_name='mobilenet_v3_large', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.mobilenet_v3_small(), model_name='mobilenet_v3_small', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.resnet101(), model_name='resnet101', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.resnet152(), model_name='resnet152', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.resnet18(), model_name='resnet18', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.resnet34(), model_name='resnet34', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.resnet50(), model_name='resnet50', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.resnext101_32x8d(), model_name='resnext101_32x8d', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.resnext101_64x4d(), model_name='resnext101_64x4d', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.resnext50_32x4d(), model_name='resnext50_32x4d', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.GoogLeNet(), model_name='inception_v1', feature_map=(1,3,224,224))

  from classifaction.InceptionV2 import InceptionV2
  model_param_analyzer_obj.get_model_param(model=InceptionV2(), model_name='inception_v2', feature_map=(1,3,224,224))

  model_param_analyzer_obj.get_model_param(model=models.Inception3(), model_name='Inception3', feature_map=(1,3,224,224))

  from classifaction.inception_v4 import inception_v4_resnet_v2
  model_param_analyzer_obj.get_model_param(model=inception_v4_resnet_v2(), model_name='inception_v4_resnet_v2', feature_map=(1,3,224,224))

  from classifaction.xception import xception_backbone
  model_param_analyzer_obj.get_model_param(model=xception_backbone(3, 16), model_name='xception', feature_map=(1,3,299,299))

  model_param_analyzer_obj.get_model_param(model=models.DenseNet(), model_name='DenseNet', feature_map=(1,3,224,224))

  from classifaction.senet.se_resnet import se_resnet50
  model_param_analyzer_obj.get_model_param(model=se_resnet50(), model_name='se_resnet50', feature_map=(1,3,224,224))
  
  from classifaction.senet.se_inception import se_inception_v3
  model_param_analyzer_obj.get_model_param(model=se_inception_v3(), model_name='se_inception_v3', feature_map=(1,3,299,299))

  from classifaction.ECANet.eca_resnet import eca_resnet50
  model_param_analyzer_obj.get_model_param(model=eca_resnet50(), model_name='eca_resnet50', feature_map=(1,3,224,224))
  from classifaction.ECANet.eca_mobilenetv2 import eca_mobilenet_v2
  model_param_analyzer_obj.get_model_param(model=eca_mobilenet_v2(), model_name='eca_mobilenet_v2', feature_map=(1,3,224,224))

  from classifaction.shufflenet import ShuffleNet
  model_param_analyzer_obj.get_model_param(model=ShuffleNet(2, 1000), model_name='shufflenet_v1', feature_map=(1,3,224,224))

  model_param_analyzer_obj.get_model_param(model=models.shufflenet_v2_x0_5(), model_name='shufflenet_v2_x0_5', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.shufflenet_v2_x1_0(), model_name='shufflenet_v2_x1_0', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.shufflenet_v2_x1_5(), model_name='shufflenet_v2_x1_5', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.shufflenet_v2_x2_0(), model_name='shufflenet_v2_x2_0', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.vgg11(), model_name='vgg11', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.vgg11_bn(), model_name='vgg11_bn', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.vgg13(), model_name='vgg13', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.vgg13_bn(), model_name='vgg13_bn', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.vgg16(), model_name='vgg16', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.vgg16_bn(), model_name='vgg16_bn', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.vgg19(), model_name='vgg19', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.vgg19_bn(), model_name='vgg19_bn', feature_map=(1,3,224,224))

  from classifaction.Lenet import Lenet5
  model_param_analyzer_obj.get_model_param(model=Lenet5(), model_name='Lenet5', feature_map=(1,3,32,32))

  model_param_analyzer_obj.get_model_param(model=models.alexnet(), model_name='alexnet', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.efficientnet_b0(), model_name='efficientnet_b0', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.efficientnet_b1(), model_name='efficientnet_b1', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.efficientnet_b2(), model_name='efficientnet_b2', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.efficientnet_b3(), model_name='efficientnet_b3', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.efficientnet_b4(), model_name='efficientnet_b4', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.efficientnet_b5(), model_name='efficientnet_b5', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.efficientnet_b6(), model_name='efficientnet_b6', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.efficientnet_b7(), model_name='efficientnet_b7', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.efficientnet_v2_l(), model_name='efficientnet_v2_l', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.efficientnet_v2_m(), model_name='efficientnet_v2_m', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.efficientnet_v2_s(), model_name='efficientnet_v2_s', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.squeezenet1_0(), model_name='squeezenet1_0', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.squeezenet1_1(), model_name='squeezenet1_1', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.wide_resnet101_2(), model_name='wide_resnet101_2', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.wide_resnet50_2(), model_name='wide_resnet50_2', feature_map=(1,3,224,224))

  from classifaction.dpn import dpn107
  model_param_analyzer_obj.get_model_param(model=dpn107(), model_name='dpn107', feature_map=(1,3,224,224))

  model_param_analyzer_obj.get_model_param(model=models.convnext_base(), model_name='convnext_base', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.convnext_large(), model_name='convnext_large', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.convnext_small(), model_name='convnext_small', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.convnext_tiny(), model_name='convnext_tiny', feature_map=(1,3,224,224))

  from classifaction.swin.swin_mlp import SwinMLP
  model_param_analyzer_obj.get_model_param(model=SwinMLP(), model_name='swinmlp', feature_map=(1,3,224,224))
  from classifaction.swin.swin_transformer_v2 import SwinTransformerV2
  model_param_analyzer_obj.get_model_param(model=SwinTransformerV2(), model_name='swintransformerv2', feature_map=(1,3,224,224))

  from classifaction.wavemlp import WaveMLP_T
  model_param_analyzer_obj.get_model_param(model=WaveMLP_T(), model_name='wavemlp_t', feature_map=(1,3,224,224))

  from classifaction.acmix.ResNet_ImageNet import ACmix_ResNet
  model_param_analyzer_obj.get_model_param(model=ACmix_ResNet(), model_name='acmix_resnet', feature_map=(1,3,224,224))
  from classifaction.acmix.swin_transformer_acmix import SwinTransformer_acmix
  model_param_analyzer_obj.get_model_param(model=SwinTransformer_acmix(), model_name='swintransformer_acmix', feature_map=(1,3,224,224))

def get_object_detection_param():
  model_param_analyzer_obj.get_model_param(model=models.detection.ssd300_vgg16(), model_name='ssd300_vgg16', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.detection.ssdlite320_mobilenet_v3_large(), model_name='ssdlite320_mobilenet_v3_large', feature_map=(1,3,224,224))

  from object_detection.yolov1.ResNet18_YOLOv1 import ResNet18_YOLOv1
  model_param_analyzer_obj.get_model_param(model=ResNet18_YOLOv1(), model_name='resnet18_yolov1', feature_map=(1,3,448,448))

  from object_detection.yolov2.yolov2 import Yolov2
  model_param_analyzer_obj.get_model_param(model=Yolov2(), model_name='yolov2', feature_map=(1,3,416,416))

  from object_detection.yolov3.yolo import YoloBody
  model_param_analyzer_obj.get_model_param(model=YoloBody([[ 10,  13],[ 16,  30],[ 33.,  23],[ 30,  61],[ 62,  45],[ 59, 119],[116,  90],[156, 198],[373, 326]], 9), model_name='yolov3', feature_map=(1,3,416,416))

  from object_detection.yolov4.yolo import YoloBody
  model_param_analyzer_obj.get_model_param(model=YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 9), model_name='yolov4', feature_map=(1,3,416,416))

  from object_detection.yolov5.yolo import YoloBody
  model_param_analyzer_obj.get_model_param(model=YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 9, 's', backbone = 'cspdarknet',           input_shape = [416, 416]), model_name='yolov5-cspdarknet',           feature_map=(1,3,416,416))
  model_param_analyzer_obj.get_model_param(model=YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 9, 's', backbone = 'convnext_tiny',        input_shape = [416, 416]), model_name='yolov5-convnext_tiny',        feature_map=(1,3,416,416))
  model_param_analyzer_obj.get_model_param(model=YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 9, 's', backbone = 'convnext_small',       input_shape = [416, 416]), model_name='yolov5-convnext_small',       feature_map=(1,3,416,416))
  model_param_analyzer_obj.get_model_param(model=YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 9, 's', backbone = 'swin_transfomer_tiny', input_shape = [416, 416]), model_name='yolov5-swin_transfomer_tiny', feature_map=(1,3,416,416))

  from object_detection.yolov6.models.yolo import Model
  from object_detection.yolov6.utils.config import Config
  cfg = Config.fromfile("./src/object_detection/yolov6/configs/yolov6l.py")
  model_param_analyzer_obj.get_model_param(model=Model(cfg, channels=3, num_classes=80, fuse_ab=False, distill_ns=False), model_name='yolov6', feature_map=(1,3,416,416))

  from object_detection.yolov7.yolo import YoloBody
  model_param_analyzer_obj.get_model_param(model=YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 9, 'l'), model_name='yolov7', feature_map=(1,3,416,416))

  from object_detection.yolof.models import build_model
  from object_detection.yolof.config import build_config
  for version in ['yolof-r18', 'yolof-r50', 'yolof-r50-DC5', 'yolof-rt-r50', 'fcos-r18', 'fcos-r50', 'fcos-rt-r18', 'fcos-rt-r50', 'retinanet-r18', 'retinanet-r50', 'retinanet-rt-r18', 'retinanet-rt-r50']:
    cfg = build_config(version)
    model_param_analyzer_obj.get_model_param(model=build_model(version=version, cfg=cfg, device=torch.device("cpu"), num_classes=80, trainable=False), model_name=version, feature_map=(1,3,416,416))

  from classifaction.Res2Net.dla import res2net_dla60, res2next_dla60
  from classifaction.Res2Net.res2net import res2net50
  from classifaction.Res2Net.res2next import res2next50
  model_param_analyzer_obj.get_model_param(model=res2net_dla60(pretrained=False), model_name='res2net_dla60', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=res2next_dla60(pretrained=False), model_name='res2next_dla60', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=res2net50(pretrained=False), model_name='res2net50', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=res2next50(pretrained=False), model_name='res2next50', feature_map=(1,3,224,224))

  model_param_analyzer_obj.get_model_param(model=models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False), model_name='fasterrcnn_mobilenet_v3_large_320_fpn', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.detection.fasterrcnn_mobilenet_v3_large_fpn(), model_name='fasterrcnn_mobilenet_v3_large_fpn', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.detection.fasterrcnn_resnet50_fpn(), model_name='fasterrcnn_resnet50_fpn', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.detection.fasterrcnn_resnet50_fpn_v2(), model_name='fasterrcnn_resnet50_fpn_v2', feature_map=(1,3,224,224))

  print("please add model: ST-GCN")
  print("please add model: Centernet")
  print("please add model: AutoAssign")

  model_param_analyzer_obj.get_model_param(model=models.detection.maskrcnn_resnet50_fpn_v2(), model_name='maskrcnn_resnet50_fpn_v2', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.detection.maskrcnn_resnet50_fpn(), model_name='maskrcnn_resnet50_fpn', feature_map=(1,3,224,224))

  print("please add model: PVANet")
  print("please add model: R-FCN")
  print("please add model: RetinaFace")
  print("please add model: goturn")
  print("please add model: PP-YOLOE")
  model_param_analyzer_obj.get_model_param(model=models.vit_b_16(), model_name='vit_b_16', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.vit_b_32(), model_name='vit_b_32', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.vit_h_14(), model_name='vit_h_14', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.vit_l_16(), model_name='vit_l_16', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.vit_l_32(), model_name='vit_l_32', feature_map=(1,3,224,224))

  model_param_analyzer_obj.get_model_param(model=models.detection.retinanet_resnet50_fpn(), model_name='retinanet_resnet50_fpn', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.detection.retinanet_resnet50_fpn_v2(), model_name='retinanet_resnet50_fpn_v2', feature_map=(1,3,224,224))

  print("please add model: GCNet")
  print("please add model: siameseNet")
  print("please add model: BiT")

def get_semantic_segmentation_param():
  print("please add model: segnet")

  model_param_analyzer_obj.get_model_param(model=models.segmentation.fcn_resnet101(), model_name='fcn_resnet101', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=models.segmentation.fcn_resnet50(), model_name='fcn_resnet50', feature_map=(1,3,224,224))

  print("please add model: pspnet")
  print("please add model: icnet")
  print("please add model: bisenet")
  print("please add model: erfnet")
  print("please add model: enet")
  print("please add model: mdnet")
  print("please add model: unet")
  print("please add model: dilated residual networks")
  print("please add model: deeplab")
  print("please add model: apcnet")
  print("please add model: cgnet")
  print("please add model: contextnet")
  print("please add model: dabnet")
  print("please add model: danet")
  print("please add model: denseaspp")
  print("please add model: dfanet")
  print("please add model: dunet")
  print("please add model: encnet")
  print("please add model: fastscnn")
  print("please add model: fpenet")
  print("please add model: hardnet")
  print("please add model: lednet")
  print("please add model: linknet")
  print("please add model: ocrnet")
  print("please add model: psanet")
  print("please add model: refinenet")
  print("please add model: 3d-unet")
  print("please add model: dnlnet")

def get_nlp_param():
  print("please add model: bert")
  print("please add model: gpt-2")
  print("please add model: transformer")
  print("please add model: nmt")
  print("please add model: glm")
  print("please add model: cpm")
  print("please add model: ernie")
  print("please add model: t5")
  print("please add model: distillbert")
  print("please add model: roberta")
  print("please add model: roformer")
  print("please add model: xlnet")
  print("please add model: albert")
  print("please add model: bvrbert")
  print("please add model: rtcbert")

def get_face_recognition_param():
  print("please add model: arcface")
  print("please add model: cosface")
  print("please add model: facenet")
  print("please add model: deepid3")
  print("please add model: mtcnn-net")

def get_super_resolution_param():
  print("please add model: basicvsr")
  print("please add model: basicvsr++")
  print("please add model: esrgan")
  print("please add model: liif")
  print("please add model: realbasicvsr")
  print("please add model: ttsr")
  print("please add model: ttvsr")
  print("please add model: rcan")
  print("please add model: srresnet")
  print("please add model: vdsr")
  print("please add model: srcnn")

def get_pose_estimation_param():
  print("please add model: openpose")
  print("please add model: HRNet")
  print("please add model: DHRNet")

def get_speech_param():
  print("please add model: ESPNet")
  print("please add model: Conformer")
  print("please add model: RNN-T")
  print("please add model: DFSMN")
  print("please add model: GRU")
  print("please add model: RNN")
  print("please add model: DeepSpeech")
  print("please add model: Tacotron2")
  print("please add model: VQMIVC")
  print("please add model: WaveGlow")

def get_recommeder_systems_param():
  print("please add model: DeepFM")
  print("please add model: Wide&Deep")
  print("please add model: NCF")
  print("please add model: DLRM")
  print("please add model: deep_interest_net")
  print("please add model: xDeepFM")
  print("please add model: MMOE")
  print("please add model: ipnn")
  print("please add model: kpnn")
  print("please add model: opnn")
  print("please add model: CVR-Net")
  print("please add model: DCNv2")
  print("please add model: DAE")
  print("please add model: MLP")

def get_cv_ocr_param():
  print("please add model: SAR")
  print("please add model: SATRN")
  print("please add model: CRNN+CTC")
  print("please add model: PSENet")
  print("please add model: PP_COR")
  print("please add model: CRNN")

def get_all_param():
  get_classifaction_param()
  get_object_detection_param()
  get_semantic_segmentation_param()
  get_nlp_param()
  get_face_recognition_param()
  get_super_resolution_param()
  get_pose_estimation_param()
  get_recommeder_systems_param()
  get_cv_ocr_param()

# Add more model type
model_type_map = {
  "classifaction"         :get_classifaction_param,
  "object_detection"      :get_object_detection_param,
  "semantic_segmentation" :get_semantic_segmentation_param,
  "nlp"                   :get_nlp_param,
  "face_recognition"      :get_face_recognition_param,
  "super_resolution"      :get_super_resolution_param,
  "pose_estimation"       :get_pose_estimation_param,
  "recommeder_systems"    :get_recommeder_systems_param,
  "cv_ocr"                :get_cv_ocr_param,
  "all"                   :get_all_param,
}

def analysis_models(nnType="classifaction"):
  model_type_map[nnType]()

  model_param_analyzer_obj.analysis_frequency()
  model_param_analyzer_obj.draw_analysis_statictis(nnType)

def main():
  parser = argparse.ArgumentParser(description='manual to this script')
  parser.add_argument('--nnType', type=str, default = 'classifaction')
  args = parser.parse_args()
  nnType = args.nnType

  analysis_models(nnType)

if __name__ == "__main__":
    main()
