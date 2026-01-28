import torch
import pandas as pd
import time
import os
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Metric
from torch import Tensor
from typing import Optional,Any
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict
from .history import History
from .batch_results import BatchResults

class Trainer:

    def __init__(self,
        model : Optional[Module] = None,
        criteron : Optional[Module] = None,
        optimizer : Optional[Optimizer] = None,
        scheduler : Optional[LRScheduler] = None,
        metrics : Optional[dict[str, Metric]] = None,
        score_metric : Optional[str] = None,
        device : Optional[torch.device] = None,
        weights_folder : Optional[str] = None,
        save_weights_every : Optional[int] = None,
        save_best_weights : bool = False
    ) -> None:
        
        self.model = model
        self.criteron = criteron
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics if metrics is not None else {}
        self.score_metric = score_metric
        self.device = device
        self.weights_folder = weights_folder
        self.save_weights_every = save_weights_every
        self.history = self._init_history()
        self.save_best_weights = save_best_weights

        self.best_weights = None
        self.best_epoch = None
        self.last_best_score = None

    def _init_history(self) -> History:

        return History(
            splits=['train', 'val'], 
            metrics=['epoch','time','loss'] + (list(self.metrics.keys()) if self.metrics is not None else [])
        )

    def set_model(self, model : Module):
        self.model = model
        return self
    
    def set_criteron(self, criteron : Module):
        self.criteron = criteron
        return self
    
    def set_optimizer(self, optimizer : Optimizer):
        self.optimizer = optimizer
        return self
    
    def set_scheduler(self, scheduler : Optional[LRScheduler]):
        self.scheduler = scheduler
        return self
    
    def set_score_metric(self, name : str):

        if name not in self.metrics.keys():
            raise Exception(f"name must be one of : {','.join(list(self.metrics.keys()))},but found : {name} instead.")
        
        self.score_metric = name

        return self
    
    def set_device(self, device : torch.device):
        self.device = device
        return self
    
    def set_weights_folder(self, weights_folder : str):
        self.weights_folder = weights_folder
        return self
    
    def set_save_weights_every(self, save_weights_every : int):
        self.save_weights_every = save_weights_every
        return self
    
    def add_metric(self, name : str, metric : Metric):
        self.metrics[name] = metric
        self.history = self._init_history()
        return self
    
    def _format(self, dict_ : dict) -> str:

        r = []

        for key,value in dict_.items():
            r.append(f"{key} = {value}")

        return ','.join(r)
    
    def train_on_batch(self, batch : tuple[Any, Any], results : BatchResults) -> tuple[dict, Tensor]:

        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        ### Zero the gradints
        self.optimizer.zero_grad()

        ### Forward Step
        y_hat = self.model(x)

        ### Calculate the loss
        loss = self.criteron(y_hat, y)

        ### Backward
        loss.backward()

        ### Update the weights
        self.optimizer.step()

        ### Update the learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        ### Update the metrics
        step_results = results.update(y_hat, y)

        return step_results,loss
    
    def eval_on_batch(self, batch : tuple[Any, Any], results : BatchResults) -> tuple[dict, Tensor]:

        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        ### Forward Step
        y_hat = self.model(x)

        ### Calculate the loss
        loss = self.criteron(y_hat, y)

        ### Update the metrics
        step_results = results.update(y_hat, y)

        return step_results,loss
    
    def train_on_loader(self, loader : DataLoader) -> tuple[dict[str, float],float]:

        self.model = self.model.train()

        results = BatchResults(self.metrics)
        running_loss = 0.0

        t_object = tqdm(loader)

        for batch in t_object:

            step_results,loss = self.train_on_batch(batch, results)
            running_loss += loss.item()

            lr = f"lr={self.scheduler.get_last_lr()[0]}" if self.scheduler is not None else ""
            dsc = self._format(step_results) + f",loss={loss.item()},{lr}" 
            t_object.set_description(dsc)

        running_loss /= len(loader)

        return results.compute(),running_loss
    
    def eval_on_loader(self, loader : DataLoader) -> tuple[dict[str, float],float]:

        self.model = self.model.eval()

        results = BatchResults(self.metrics)
        running_loss = 0.0

        with torch.inference_mode():

            for batch in tqdm(loader):

                _,loss = self.eval_on_batch(batch, results)
                running_loss += loss.item()

        running_loss /= len(loader)

        return results.compute(),running_loss
    
    def train(self, train_dataloader : DataLoader,val_dataloader : DataLoader, epochs : int) -> None:

        for epoch in range(epochs):

            train_tic = time.time()
            train_results, train_loss = self.train_on_loader(train_dataloader)
            train_toc = time.time()

            print()

            val_tic = time.time()
            val_results, val_loss = self.eval_on_loader(val_dataloader)
            val_toc = time.time()

            train_results['loss'] = train_loss
            val_results['loss'] = val_loss

            msg = f"\nEpoch {epoch+1} : Train Results : {self._format(train_results)}"
            msg += f" \n Validation Results : {self._format(val_results)}\n"

            print(msg)

            train_results['epoch'] = epoch
            train_results['time'] = (train_toc - train_tic)
            val_results['epoch'] = epoch
            val_results['time'] = (val_toc - val_tic)

            self.history.update(train_results, 'train')
            self.history.update(val_results, 'val')

            if self.save_best_weights and (self.last_best_score is None or val_results[self.score_metric] > self.last_best_score):

                self.best_epoch = epoch + 1
                self.last_best_score = val_results[self.score_metric]
                self.best_weights = OrderedDict()

                for name,weight in self.model.state_dict().items():
                    self.best_weights[name] = weight.cpu()

            if (self.save_weights_every is not None) and (epoch + 1) % self.save_weights_every == 0:
                path = os.path.join(self.weights_folder, f"{time.time()}_epoch={epoch+1}.pt")
                print(f"\nSaving weights to : {path}\n")
                torch.save(self.model.state_dict(), path)