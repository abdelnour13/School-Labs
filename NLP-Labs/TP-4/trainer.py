import torch
import os
import time
from tqdm import tqdm
from torch import nn
from torchmetrics import Metric
from typing import Any,Optional,Literal
from collections import OrderedDict

class Trainer:
    """
        a class that allows training models and recording a history
        of different scores & the loss over time.
    """

    def __init__(self,
        optimizer:Optional[torch.optim.Optimizer] = None,
        loss:Optional[torch.nn.Module] = None,
        scheduler : Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        metrics : Optional[dict[str, Metric]] = None,
        device : str =  'cpu',
        save_best_weights : bool = False,
        score_metric : Optional[str] = None,
        log_every : int = 1,
        save_every : int = 10,
        weights_folder : Optional[str] = None,
        progress_level : Literal['epoch','iteration'] = 'iteration'
    ):
        """
            The constructor of the Trainer class.

            - optimizer : the optimizer used to update the weights of type torch.optim.Optimizer.
            - loss : the loss function of type torch.nn.Module.
            - metrics : dictionary of metrics to compute each epoch.
            - device : the device you wish to train the model in.
        """
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.metrics = metrics if metrics is not None else {}
        self.history = self.init_history()
        self.model = None
        self.device = device
        self.save_best_weights = save_best_weights
        self.best_weights = None
        self.last_best_score = None
        self.best_epoch = None
        self.log_every = log_every
        self.save_every = save_every
        self.weights_folder = weights_folder
        self.progress_level = progress_level

        self.set_score_metric(score_metric)

    def set_save_every(self, save_every : int):
        self.save_every = save_every
        return self
    
    def set_weights_folder(self, weights_folder : str):
        self.weights_folder = weights_folder
        return self

    def set_log_every(self, log_every : int):
        self.log_every = log_every
        return self

    def set_optimizer(self, optimizer : torch.optim.Optimizer):
        """
            sets the optimizer to train the model on.
        
            - optimizer : torch.optim.Optimizer.

            Retuns : 
            - self.
        """
        self.optimizer = optimizer
        return self

    def set_loss(self, loss : torch.nn.Module):
        """
            sets the loss of the model.
        
            - loss : torch.nn.Module.

            Retuns : 
            - self.
        """
        self.loss = loss
        return self

    def set_device(self, device : str):
        """
            sets the device of the model.
        
            - device : either cpu or cuda.

            Retuns : 
            - self.
        """
        self.device = device
        return self
    
    def set_score_metric(self, score_metric : str):

        if self.save_best_weights:

            if score_metric is None:
                raise Exception(f"save best weights requires 'metric' to not be None.")
            
            if score_metric not in list(self.metrics.keys()) + ['loss']:
                raise Exception(f"""
                    save best weights requires 'metric' must be one of {','.join(self.metrics.keys())}
                    found : {score_metric}
                """)      
            
            self.score_metric = score_metric

        return self  

    def add_metric(self,name:str,metric:Metric):
        """
            add a metric to the list of metrics already specified.

            - name : the name of metric,of type str.
            - metric : the metric to add,of type torchmetrics.Metric.

            Retuns : 
            - .
        """
        self.metrics[name] = metric
        self.history['train'][name] = []
        self.history['val'][name] = []
        return self
    
    def set_scheduler(self, scheduler : Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """
            add a metric to the list of metrics already specified.

            - name : the name of metric,of type str.
            - metric : the metric to add,of type torchmetrics.Metric.

            Retuns : 
            - .
        """
        self.scheduler = scheduler
        return self
    
    def set_save_best_weights(self, save_best_weights : bool):
        self.save_best_weights = save_best_weights
        return self

    def init_history(self) -> dict:
        """
            Returns :

            - a dictionary of arrays to store the history of the metrics. 
        """

        history = {
            'train' : {},
            'val' : {}
        }

        for name in list(self.metrics.keys()) + ['loss','epoch','time']:
            history['train'][name] = []
            history['val'][name] = []

        return history
    
    def reset(self) -> None:
        """
            resets the model by clearing the history.

            Retuns : 
            - None
        """
        self.history = self.init_history()

    def compute_metrics(self,
        y:torch.Tensor,
        y_hat:torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
            compute the different metrics.

            - y : torch.Tensor.
            - y_hat : torch.Tensor.

            Retuns : 
            - None.
        """

        result = {}

        for name,metric in self.metrics.items():
            result[name] = metric(y_hat, y).item()

        return result
    
    def add_dicts(self, dict1 : dict, dict2 : dict) -> dict[str,torch.Tensor]:
        """
            adds the values of tow dictionaries together.

            Returns : 
            - a new dictionary containing the results.
        """

        result = {}

        for key,value in dict1.items():
            result[key] = value + dict2[key]

        return result
    
    def div_dict(self, dict_ : dict, by : Any):
        """
            divides the values of dictionary by a given value.

            Returns : 
            - a new dictionary containing the results.
        """

        result = {}

        for key,value in dict_.items():
            result[key] = dict_[key] / by

        return result
    
    def get_results_dict(self) -> dict[str,float] :
        """
            Returns :
            - a new dictionary containing the keys of the metrics with a default value of zero.
        """
        
        results = {}

        for name in list(self.metrics.keys()) + ['loss']:
            results[name] = 0.0

        return results


    def append_to_history(self, results : dict[str,torch.Tensor], to : str = 'train') -> None:
        """
            appends a dictionary of results to the history.

            - results : dictionary of metrics.
            - to : whether to append to the validationsor the training history,must be string either 'train' ot 'val'            
        
            Retruns : 
            - None.
        """
        
        for key,value in results.items():
            self.history[to][key].append(value)


    def train_on_batch(self,X_batch : torch.Tensor,y_batch : torch.Tensor) -> tuple[float,torch.Tensor]:
        """
            applies one training step on a batch

            - X_batch : torch.Tensor.
            - y_batch : torch.Tensor.

            Returns :
            - float : the loss.
            - torch.Tensor : the predictions.
        """

        # 1- Forward pass
        y_hat = self.model(X_batch)

        # 2- Calculate the loss
        loss = self.loss(y_hat, y_batch)

        # 3- zero the gradients
        self.optimizer.zero_grad()

        # 4- Backward pass
        loss.backward()

        # 5- Optimizer step
        self.optimizer.step()

        return loss.item(),y_hat

    def test_on_batch(self,X_batch:torch.Tensor,y_batch:torch.Tensor) -> tuple[float,torch.Tensor]:
        """
            applies one validation step on a batch.

            - X_batch : torch.Tensor.
            - y_batch : torch.Tensor.

            Returns :
            - float : the loss.
            - torch.Tensor : the predictions.
        """

        # 1- Forward pass
        y_hat = self.model(X_batch)

        # 2- Calculate the loss
        loss = self.loss(y_hat, y_batch)

        return loss.item(),y_hat

    def check(self) -> None:

        if self.optimizer is None:
            raise Exception("optimizer can not be None.")
        
        if self.loss is None:
            raise Exception("loss can not be None.")
        
    def format(self, dict_ : dict,prefix : str = '') -> str:
        return ','.join(
            [f"{prefix}{key} = {value}" for key,value in dict_.items()]
        )

    def train(self,
        model:nn.Module,
        train_dataloader : torch.utils.data.DataLoader,
        val_dataloader : Optional[torch.utils.data.DataLoader] = None,
        epochs : int = 100
    ) -> None:
        """
            trains the model for number of epochs.

            - model : the model to train, of type torch.nn.Module.
            - train_dataloader : the loader to load the training batches,of type torch.utils.data.DataLoader.
            - val_dataloader : the loader to load the validation batches,of type torch.utils.data.DataLoader.
            - epochs : the number of epochs to train the model,of type int.

            Returns : 
            - None
        """

        # check if the parameters are valid.
        self.check()

        self.model = model

        t_epochs = tqdm(range(epochs)) if self.progress_level == 'epoch' else range(epochs)
        for epoch in t_epochs:

            # put the model in training mode
            self.model.train()

            train_results = self.get_results_dict()
            val_results = self.get_results_dict()

            train_tic = time.time()

            ### Training loop
            t = tqdm(train_dataloader) if self.progress_level == 'iteration' else train_dataloader
            for i,(X_batch,y_batch) in enumerate(t):
                # put the data in the right device
                X_batch,y_batch = X_batch.to(self.device),y_batch.to(self.device)
                loss, y_hat = self.train_on_batch(X_batch,y_batch)
                train_batch_results = self.compute_metrics(y_batch,y_hat)
                train_batch_results['loss'] = loss

                if i % self.log_every == 0 and self.progress_level == 'iteration':
                    t.set_description(self.format(train_batch_results))

                train_results = self.add_dicts(train_results, train_batch_results)

            train_toc = time.time()

            ### Testing loop
            if val_dataloader is not None:
                # put the model in training mode
                self.model.eval()

                with torch.inference_mode():
                    
                    val_tic = time.time()

                    for X_batch,y_batch in val_dataloader:
                        # put the data in the right device
                        X_batch,y_batch = X_batch.to(self.device),y_batch.to(self.device)
                        loss, y_hat = self.test_on_batch(X_batch,y_batch)
                        val_batch_results = self.compute_metrics(y_batch,y_hat)
                        val_batch_results['loss'] = loss
                        val_results = self.add_dicts(val_results, val_batch_results)

                    val_toc = time.time()

            # adjust the metrics
            train_results = self.div_dict(train_results, by=len(train_dataloader))
            
            if val_dataloader is not None:

                val_results = self.div_dict(val_results, by=len(val_dataloader))

                if self.save_best_weights:

                    if self.last_best_score is None or self.last_best_score < val_results[self.score_metric]:

                        self.last_best_score = val_results[self.score_metric]
                        self.best_epoch = epoch

                        self.best_weights = OrderedDict()

                        for k,v in self.model.state_dict().items():
                            self.best_weights[k] = v.cpu()

            # append to the history
            self.append_to_history(train_results, to='train')
            self.history["train"]["epoch"].append(epoch)
            self.history["train"]['time'].append((train_toc - train_tic))

            if val_dataloader is not None:
                self.append_to_history(val_results, to='val')
                self.history["val"]["epoch"].append(epoch)
                self.history["val"]['time'].append((val_toc - val_tic))

            # update the learning rate if a scheduler is defined
            if self.scheduler is not None:
                self.scheduler.step()

            msg = f"Epoch {epoch+1} : " + self.format(train_results) + "," + self.format(val_results, prefix='val_') + "\n"

            if (epoch+1) % self.save_every == 0 and self.weights_folder is not None:
                state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(self.weights_folder, f"epoch-{epoch}.pt"))

            if self.progress_level == 'iteration':
                print(msg)
            else:
                t_epochs.set_description(msg)