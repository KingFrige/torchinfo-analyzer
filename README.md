# README

## start

```bash
$ git clone <repo-path>

$ cd torchinfo-analyzer

$ git submodule update --init --recursive
```

## require

### way 1: Using pip without a virtual environment

```bash
$ pip install -r requirements.txt
```

### way 2: Using a virtual environment with *virtualenv*

- Create your virtual environment for `python3`:

```bash
$ python3 -m venv venv
```
   
- Activate your virtualenv:

```bash
$ source venv/bin/activate
```

- Install dependencies using the `requirements.txt`:

```bash
$ pip install -r requirements.txt
```

## function

- model parameter generate
- operators analysis: type / kernel / featuremap size
- scenario related operator analysis
- chart display

## usage

```bash
$ make analysis-demo-info

$ make analysis-classifaction-models-info

$ make analysis-torch-models-info
```

## reference

1. [torchinfo](https://github.com/TylerYep/torchinfo)
1. [NNparser](https://github.com/alexhegit/NNparser)
1. [pytorch_model_summary](https://github.com/ceykmc/pytorch_model_summary)
1. [chainer_computational_cost](https://github.com/belltailjp/chainer_computational_cost)
1. [torchstat](https://github.com/Swall0w/torchstat)
1. [pytorch-summary](https://github.com/sksq96/pytorch-summary)
1. [pytorch_modelsummary](https://github.com/jacobkimmel/pytorch_modelsummary)
1. [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch)

1. [pandas追加sheet](https://blog.csdn.net/gf1321111/article/details/130041260)
