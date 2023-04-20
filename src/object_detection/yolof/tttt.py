import torch

from models import build_model
from config import build_config

for version in ['yolof-r18', 'yolof-r50', 'yolof-r50-DC5', 'yolof-rt-r50', 'fcos-r18', 'fcos-r50', 'fcos-rt-r18', 'fcos-rt-r50', 'retinanet-r18', 'retinanet-r50', 'retinanet-rt-r18', 'retinanet-rt-r50']:
  cfg = build_config(version)
  model = build_model(version=version, cfg=cfg, device=torch.device("cpu"), num_classes=80, trainable=False)
  
  print(model)
