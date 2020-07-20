#!/usr/bin/env python3
# _*_ coding: utf_8 _*_
"""
Created on Wed Jul  8 00:00:16 2020

@author: btayart
"""


class CustomArgs():
    def __init__(self,
                 mode="train",
                 resume=True,
                 batch_size=10,
                 epochs=300,
                 learning_rate=5e-4,
                 lr_decay=0.1,
                 lr_decay_epochs=100,
                 weight_decay=2e-4,
                 dataset="camvid",
                 dataset_dir="data/camvid",
                 height=360, width=480,
                 weighing = "ENet",
                 
                 ignore_unlabeled = False,
                 workers=4,
                 print_step=True,
                 imshow_batch=True,
                 device="cuda",
                 name = "ENet",
                 save_dir = "save",
                 
                 model_type="ENet",
                 dropout = None
                 ):
        # Hyperparameters
        self.mode = mode
        self.resume = resume
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lr_decay_epochs = lr_decay_epochs
        self.weight_decay = weight_decay
        # Dataset
        self.dataset=dataset
        self.dataset_dir = dataset_dir
        self.height = height
        self.width = width
        self.weighing = weighing
        self.ignore_unlabeled = ignore_unlabeled
        # Settings
        self.workers = workers
        self.print_step = print_step
        self.imshow_batch = imshow_batch
        self.device = device
        # Storage settings
        self.name = name
        self.save_dir = save_dir
        # Custom
        self.model_type = model_type
        self.dropout = dropout