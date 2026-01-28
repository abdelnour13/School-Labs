from torchmetrics import Metric
from torch import Tensor
from typing import Any

class BatchResults:

    def __init__(self, metrics : dict[str, Metric]):

        self.metrics = metrics
        self.results = {}
        self.iters = 0
        self.exipred = False

        for name in metrics.keys():
            self.results[name] = 0.0

    def update(self, y_hat : Tensor, y : Tensor) -> dict[str, float]:

        if self.exipred:
            raise Exception("This object can no longer be used to accumelate results,instantiate a new object.")

        results = {}

        for name,metric in self.metrics.items():
            results[name] = metric(y_hat, y)    
            self.results[name] += results[name].item()

        self.iters += 1

        return results

    def __getitem__(self, key : Any):
        return self.results[key]

    def compute(self) -> dict[str, float]:

        for name in self.metrics.keys():
            self.results[name] /= self.iters

        self.exipred = True

        return self.results