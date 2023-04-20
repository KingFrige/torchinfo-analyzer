#!/bin/env python3

import torch
import torchvision.models as models

from model_param_analyzer import Model_param_analyzer
model_param_analyzer_obj = Model_param_analyzer()

def main():
  from stable_diffusion.stable_diffusion_pytorch import CLIP, Encoder, Decoder, Diffusion

  # clip = CLIP().to('cpu')
  # encoder = Encoder().to('cpu')
  # decoder = Decoder().to('cpu')
  diffusion = Diffusion().to('cpu')
  
  # model_param_analyzer_obj.get_model_param(model=clip, model_name='stable_diffusion_clip', feature_map=(1,3,224,224))
  # model_param_analyzer_obj.get_model_param(model=encoder, model_name='stable_diffusion_encoder', feature_map=(1,3,224,224))
  # model_param_analyzer_obj.get_model_param(model=decoder, model_name='stable_diffusion_decoder', feature_map=(1,3,224,224))
  model_param_analyzer_obj.get_model_param(model=diffusion, model_name='stable_diffusion_diffusion', feature_map=(1,3,224,224))

  model_param_analyzer_obj.analysis_frequency()
  model_param_analyzer_obj.show_analysis_statictis()
  model_param_analyzer_obj.draw_analysis_statictis('stable_diffusion')

if __name__ == "__main__":
    main()
