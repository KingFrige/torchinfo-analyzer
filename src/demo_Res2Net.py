#!/bin/env python3

import torch
import torchvision.models as models

from model_param_analyzer import Model_param_analyzer
model_param_analyzer_obj = Model_param_analyzer()

def main():
  from classifaction.Res2Net.dla import res2net_dla60, res2next_dla60
  from classifaction.Res2Net.res2net import res2net50
  from classifaction.Res2Net.res2next import res2next50
  
  model_param_analyzer_obj.get_model_param(model=res2net_dla60(pretrained=False), model_name='res2net_dla60', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=res2next_dla60(pretrained=False), model_name='res2next_dla60', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=res2net50(pretrained=False), model_name='res2net50', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=res2next50(pretrained=False), model_name='res2next50', feature_map=(1,3,224,224))

  model_param_analyzer_obj.analysis_frequency()
  model_param_analyzer_obj.show_analysis_statictis()
  model_param_analyzer_obj.draw_analysis_statictis('res2Net')

if __name__ == "__main__":
    main()
