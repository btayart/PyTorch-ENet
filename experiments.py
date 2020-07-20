#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 00:09:38 2020

@author: btayart
"""


#%% Redo all training with ignore_unlabeled set to True
#Train the ENet model with vanilia CamVid
from main import run_training
from custom import CustomArgs
import torch
args = CustomArgs(resume=False, batch_size=6, print_step=False,
                  ignore_unlabeled = True,
                  model_type="ENet", name="ENet_allclasses")
run_training(args)
torch.cuda.empty_cache()
# BEST VALIDATION
# Epoch: 170
# Mean IoU: 0.651799197633767

args = CustomArgs(resume=False, batch_size=6, print_step=False, weight_decay=2e-3,
                  ignore_unlabeled = True,
                  model_type="ENet", name="ENet_allclasses_wd2")
run_training(args)
torch.cuda.empty_cache()
# BEST VALIDATION
# Epoch: 270
# Mean IoU: 0.6653864847519912

# Train the ENet model on CamVid with 'people' classes dropped
args = CustomArgs(resume=False, batch_size=6, print_step=False, weight_decay=2e-3,
                  ignore_unlabeled = True,
                  model_type="ENet", name="ENet_8classes",
                  dataset="camvid_8class")
run_training(args)
torch.cuda.empty_cache()
# BEST VALIDATION
# Epoch: 150
# Mean IoU: 0.7106002417121681

#%% Redo all training with ignore_unlabeled set to True
from main import run_training
from custom import CustomArgs
import torch
# Train the BiSeNEtv2 model with vanilia CamVid
args = CustomArgs(resume=False, batch_size=16, print_step=False, weight_decay=2e-2,
                  ignore_unlabeled = True,
                  model_type="BiSeNet", name="BiSeNet_allclasses")
run_training(args)
torch.cuda.empty_cache()
# BEST VALIDATION
# Epoch: 180
# Mean IoU: 0.5579522411781617

args = CustomArgs(resume=False, batch_size=16, print_step=False, weight_decay=1e-1,
                  ignore_unlabeled = True,
                   model_type="BiSeNet", name="BiSeNet_allclasses_wd2")
run_training(args)
torch.cuda.empty_cache()
# BEST VALIDATION
# Epoch: 290
# Mean IoU: 0.5505752787346246

args = CustomArgs(resume=False, batch_size=16, print_step=False, weight_decay=5e-3,
                  ignore_unlabeled = True,
                  model_type="BiSeNet", name="BiSeNet_allclasses_wd3")
run_training(args)
torch.cuda.empty_cache()
# BEST VALIDATION
# Epoch: 100
# Mean IoU: 0.5905692669089511


# Train the BiSeNEtv2 model on CamVid with 'people' classes dropped
args = CustomArgs(resume=False, batch_size=16, print_step=False, weight_decay=5e-3,
                  ignore_unlabeled = True,
                  model_type="BiSeNet", name="BiSeNet_8classes",
                  dataset="camvid_8class")
run_training(args)
torch.cuda.empty_cache()
# BEST VALIDATION
# Epoch: 100
# Mean IoU: 0.6728652056299841

#%% Train models for MC dropout
from main import run_training
from custom import CustomArgs
import torch

# Train the ENet model on CamVid with 'people' classes dropped and 15% dropout
args = CustomArgs(resume=False, batch_size=6, print_step=False, weight_decay=2e-3,
                  ignore_unlabeled = True,
                  model_type="ENet", name="ENet_dropout", dropout=0.15,
                  dataset="camvid_8class", )
run_training(args)
torch.cuda.empty_cache()
# BEST VALIDATION
# Epoch: 290
# Mean IoU: 0.7016420531064288

# Train the BiSeNEtv2 model on CamVid with 'people' classes dropped and 15% dropout
args = CustomArgs(resume=False, batch_size=16, print_step=False, weight_decay=5e-3,
                  ignore_unlabeled = True,
                  model_type="BiSeNet", name="BiSeNet_dropout", dropout=0.15,
                  dataset="camvid_8class")
run_training(args)
torch.cuda.empty_cache()
# BEST VALIDATION
# Epoch: 200
# Mean IoU: 0.7064606623353822

#%% Train another 4 models for aggregation
from main import run_training
from custom import CustomArgs
import torch
# Train the ENet model on CamVid with 'people' classes dropped
args = CustomArgs(resume=False, batch_size=6, print_step=False, weight_decay=2e-3,
                  ignore_unlabeled = True,
                  model_type="ENet", name="ENet_8classes",
                  dataset="camvid_8class")
for ii in range(1,5):
    args.name="ENet_8classes%i"%ii
    run_training(args)
    torch.cuda.empty_cache()
# Mean IoU: 0.7117726044143295
# Mean IoU: 0.7197424765042213
# Mean IoU: 0.7004683844529975
# Mean IoU: 0.7098324303419927