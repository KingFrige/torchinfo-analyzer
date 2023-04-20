#!/bin/env python3

import torch
from torch import nn
import torchvision.models as models

from model_param_analyzer import Model_param_analyzer
model_param_analyzer_obj = Model_param_analyzer()

def main():
  from object_detection.yolov4.yolo import YoloBody
  model_param_analyzer_obj.get_model_param(model=YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 9), model_name='yolov4', feature_map=(1,3,224,224))

  model_param_analyzer_obj.analysis_frequency()
  model_param_analyzer_obj.show_analysis_statictis()
  model_param_analyzer_obj.draw_analysis_statictis('yolov4')

if __name__ == "__main__":
    main()
