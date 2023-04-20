#!/bin/env python3

import torch
import torchvision.models as models
import matplotlib.pyplot as plt
from utils.torchinfo.torchinfo import summary
from utils.torchinfo.write_table import write_table

class Model_param_analyzer():
  def __init__(self) -> None:
    self.layers_table_list = []
    self.operator_freq_dict = {}
    self.fearturemap_freq_dict = {}
    self.kernel_freq_dict = {}

  def layers_key_frequency(self, model_layers_table:dict, key_index:int=0):
    result_dict = {}
    for num, row in enumerate(model_layers_table):
      if(num == 0) or (row[key_index] == '--'): # remove HEADER_TITLES and '--'
        continue
      result_dict[row[key_index]] = result_dict.get(row[key_index], 0) + 1

    return result_dict

  def show_freq_statictis(self, operator_freq_dict, kernel_freq_dict, fearturemap_freq_dict):
    divider = "=" * 50
    summary_str = f'\n     operator_freq_dict: {len(operator_freq_dict)}\n\n'
    sorded_operator_freq = sorted(operator_freq_dict.items(), key=lambda d: d[1], reverse=False)
    for key,value in sorded_operator_freq:
      summary_str += f"{key:{30}}: {value:{10}}\n"
    summary_str += divider+ '\n'

    summary_str += f'\n     kernel_freq_dict: {len(kernel_freq_dict)}\n\n'
    sorded_kernel_freq = sorted(kernel_freq_dict.items(), key=lambda d: d[1], reverse=False)
    for key,value in sorded_kernel_freq:
      summary_str += f"{key:{30}}: {value:{10}}\n"
    summary_str += divider+ '\n'

    summary_str += f'\n     fearturemap_freq_dict: {len(fearturemap_freq_dict)}\n\n'
    sorted_fearturemap_freq = sorted(fearturemap_freq_dict.items(), key=lambda d: d[1], reverse=False)
    for key,value in sorted_fearturemap_freq:
      summary_str += f"{key:{30}}: {value:{10}}\n"
    summary_str += divider+ '\n'

    print(summary_str)

  def get_model_param(self, model, model_name, feature_map):
    txt_file_name = 'output/' + model_name + '-pytorch-param.txt'
    xlsx_file_name = 'output/pytorch-models-param.xlsx'

    print_log = open(txt_file_name, "w")

    import sys
    savedStdout = sys.stdout
    sys.stdout = print_log

    model_stat = summary(
      model,
      feature_map,
      depth=10,
      device='cpu',
      verbose=1,
      col_names=["kernel_size","input_size", "output_size", "num_params", "mult_adds"],
      row_settings=["depth"],
    )

    model_layers_table = model_stat.layers_table

    layers_operator_freq_dict = self.layers_key_frequency(model_layers_table, 0)
    layers_kernel_freq_dict = self.layers_key_frequency(model_layers_table, 1)
    layers_fearturemap_freq_dict = self.layers_key_frequency(model_layers_table, 2)
    self.show_freq_statictis(layers_operator_freq_dict, layers_kernel_freq_dict, layers_fearturemap_freq_dict)

    write_table(model_layers_table, xlsx_file_name, model_name)
    self.layers_table_list.append(model_layers_table)

    sys.stdout = savedStdout
    print_log.close()

  def key_frequency(self, frequency_dict:dict, key_index:int=0):
    for table in self.layers_table_list:
      layers_frequency_dict = self.layers_key_frequency(table, key_index)
      for key in layers_frequency_dict:
        frequency_dict[key] = frequency_dict.get(key, 0) + layers_frequency_dict.get(key, 0)

  def analysis_frequency(self)->dict:
    self.key_frequency(frequency_dict=self.operator_freq_dict, key_index=0)
    self.key_frequency(frequency_dict=self.kernel_freq_dict, key_index=1)
    self.key_frequency(frequency_dict=self.fearturemap_freq_dict, key_index=2)

  def show_analysis_statictis(self):
    self.show_freq_statictis(self.operator_freq_dict, self.kernel_freq_dict, self.fearturemap_freq_dict)

  def draw_from_dict(self, data_dict, title='title', output_dir='.'):
    by_value = sorted(data_dict.items(),key = lambda item:item[1],reverse=True)
    x = []
    y = []
    for d in by_value:
      x.append(d[0])
      y.append(d[1])

    fig_height = len(data_dict)*0.4 + 2
    plt.figure(figsize=(20, fig_height))
    plt.barh(x, y, height=0.5, linewidth=0.5)
    data_sum = sum(y)
    for i, v in enumerate(y):
      plt.text(v, i, str('{:.2f}%'.format(v/sum(y)*100)), color='blue', va='center', fontweight='bold')
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('Type')
    fig_path = output_dir + '/' + title + '-hist.png'
    plt.savefig (fig_path)

  def draw_analysis_statictis(self, prefix):
    self.draw_from_dict(self.operator_freq_dict,    prefix+'_operators_frequence',  'output')
    self.draw_from_dict(self.fearturemap_freq_dict, prefix+'_featuremap_frequency', 'output')
    self.draw_from_dict(self.kernel_freq_dict,      prefix+'_kernel_frequency',     'output')

def main():
  data_dict = {'ReLU':289, 'Conv2d':341, 'BatchNorm2d':1565, 'Linear':655, 'AdaptiveAvgPool2d':1337, 'Padding':226, 'MaxPool2d':399, 'Hardswish':967, 'Softmax':405}
  model_param_analyzer_obj = Model_param_analyzer()
  model_param_analyzer_obj.draw_from_dict(data_dict, 'operators_frequency')

if __name__ == "__main__":
    main()
