#!/bin/bash

source ~/anaconda3/bin/activate research

python train.py --samples 3 41
python train.py --samples 42 82
python train.py --samples 83 123
