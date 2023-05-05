#!/bin/env python3

import argparse
import torch
from torchstat import stat
import torchvision.models as models

from model_param_analyzer import Model_param_analyzer
model_param_analyzer_obj = Model_param_analyzer()

def get_AI_Benchmark_parm():
  print("please add model: AIBenchmark")

def get_MLMark_param():
  print("please add model: MLMark")

def get_MLPerf_param():
  print("please add model: MLPerf")

def get_DAWNBench_param():
  print("please add model: bawnbench")

def get_all_param():
  get_AI_Benchmark_parm()
  get_MLMark_param()
  get_MLPerf_param()
  get_DAWNBench_param()

# Add more model type
model_type_map = {
  "AIBenchmark" :get_AI_Benchmark_parm,
  "MLMark"      :get_MLMark_param,
  "MLPerf"      :get_MLPerf_param,
  "BAWNBench"   :get_DAWNBench_param,
  "all"         :get_all_param,
}

def analysis_models(bmType="AIBenchmark"):
  model_type_map[bmType]()

  model_param_analyzer_obj.analysis_frequency()
  model_param_analyzer_obj.draw_analysis_statictis(bmType)

def main():
  parser = argparse.ArgumentParser(description='manual to this script')
  parser.add_argument('--bmType', type=str, default = 'AIBenchmark')
  args = parser.parse_args()
  bmType = args.bmType

  analysis_models(bmType)

if __name__ == "__main__":
    main()
