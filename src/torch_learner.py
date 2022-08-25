import os
import sys
import random
import itertools
import pickle
import copy
import traceback


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from livelossplot import PlotLosses
from tqdm import tqdm
from pandarallel import pandarallel
from collections import defaultdict

import pymorphy2
import nltk

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from keras.preprocessing.sequence import pad_sequences

from .utils import *

class TorchLearner:
    def __init__(self, 
                 model, 
                 loss_function, 
                 train_dataset,
                 val_dataset,
                 metrics: dict,
                 optimizer = torch.optim.Adam,
                 batch_size: int = 64,
                 shuffle_train: bool = True,
                 lr: float = 1e-4,
                 device: str = None,
                 ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = torch.device(device)

        self.model = model
        self.model.to(self.device)

        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        
        self.loss_function = loss_function

        self.train_size = len(train_dataset)
        self.val_size = len(val_dataset)
        self.batch_size = batch_size
        self.metrics = metrics
        self.metrics["loss"] = self.loss_function
        
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle_train,
        )

        self.val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
        )
        
        self.best_val_loss = float('inf')
        self.best_epoch_i = 0
        self.best_model = None
        
        self.last_epoch_i = -1
        self.history = {"epoch": []}
        liveplot_groups = {}
        for k in self.metrics.keys():
            self.history[k + "_train"] = []
            self.history[k + "_val"] = []
        
            liveplot_groups[k] = [k + "_train", k + "_val"]
        
        self.liveplot = PlotLosses(groups=liveplot_groups)
        
    def fit_epoch(self, steps_per_epoch = None):
        self.model.train()
        batch_n = 0
        
        metrics = defaultdict(float)
        for batch_i, (batch_x, batch_y) in enumerate(self.train_dataloader):
            if steps_per_epoch is not None:
                if batch_i > steps_per_epoch:
                    break
                    
            batch_x = copy_data_to_device(batch_x, self.device)
            batch_y = copy_data_to_device(batch_y, self.device)
            
            pred = self.model(batch_x)
            loss = self.loss_function(pred, batch_y)

            for metric_name, metric_f in self.metrics.items():
                metrics[metric_name] += float(metric_f(pred, batch_y))
            
            self.model.zero_grad() 
            loss.backward() 
            self.optimizer.step()

            batch_n += 1
        
        for metric_name, metric_value in metrics.items():
            metrics[metric_name] /= batch_n
        
        return metrics
    
    def eval(self, steps_per_epoch = None):
        self.model.eval()
        batch_n = 0
        
        metrics = defaultdict(float)
        with torch.no_grad():
            for batch_i, (batch_x, batch_y) in enumerate(self.val_dataloader):
                if steps_per_epoch is not None:
                    if batch_i > steps_per_epoch:
                        break

                batch_x = copy_data_to_device(batch_x, self.device)
                batch_y = copy_data_to_device(batch_y, self.device)

                pred = self.model(batch_x)
                loss = self.loss_function(pred, batch_y)

                for metric_name, metric_f in self.metrics.items():
                    metrics[metric_name] += float(metric_f(pred, batch_y))
                
                batch_n += 1
                
        for metric_name, metric_value in metrics.items():
            metrics[metric_name] /= batch_n
            
        return metrics
                
    def fit(self, 
            epoch: int = 10, 
            step_per_epoch_train: int = None,
            step_per_epoch_val: int = None, 
            early_stopping_patience: int = 10):
           
        if step_per_epoch_train is None:
            step_per_epoch_train = self.train_size // self.batch_size
        
        if step_per_epoch_val is None:
            step_per_epoch_val = self.val_size // self.batch_size        
        
        for epoch_i in range(epoch):
            try:
                train_metrics = self.fit_epoch(step_per_epoch_train)
                val_metrics = self.eval(step_per_epoch_val)
                self.last_epoch_i += 1

                liveplot_update_dict = {}
                for k in self.metrics.keys():
                    liveplot_update_dict[k + "_train"] = train_metrics[k]
                    liveplot_update_dict[k + "_val"] = val_metrics[k]
                    
                self.liveplot.update(liveplot_update_dict)
                self.liveplot.draw()
                
                self.history["epoch"].append(self.last_epoch_i)                
                for k in self.metrics.keys():
                    self.history[k + "_train"].append(train_metrics[k])
                    self.history[k + "_val"].append(val_metrics[k])
                                                
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_epoch_i = epoch_i
                    self.best_val_loss = val_metrics["loss"]
                    self.best_model = copy.deepcopy(self.model)    
                                                
                elif epoch_i - self.best_epoch_i > early_stopping_patience:
                    print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(
                        early_stopping_patience))
                    break

            except KeyboardInterrupt:
                print('Досрочно остановлено пользователем')
                break

            except Exception as ex:
                print('Ошибка при обучении: {}\n{}'.format(ex, traceback.format_exc()))
                break