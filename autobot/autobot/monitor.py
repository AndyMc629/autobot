# Monitor will be a class that:
# 1. Will be configured to calculate model performance metrics inside a simulation (using truth and predicted data)
# 2. Will be run inside a simulation (simplest case at the end, more complex cases in 'real' time)
# 3. Will generate alert objects (dicts/jsons) for each inference event or metric calculation that can be consumed downstream

import pandas as pd
import numpy as np

class Monitor(object):
    def __init__(self, predictions, truth, metrics, **kwargs):
        self.predictions = predictions
        self.truth = truth
        self.metrics = metrics
        self.kwargs = kwargs

    #validate that predictions is a tuple, list, pandas dataframe or numpy array
    @property
    def predictions(self):
        return self._predictions

    @predictions.setter
    def predictions(self, predictions):
        if not isinstance(predictions, (tuple, list, pd.DataFrame, np.ndarray)):
            raise ValueError("predictions must be a tuple, list, pandas dataframe or numpy array")
        self._predictions = predictions

    #validate that truth is a tuple, list, pandas dataframe or numpy array
    @property
    def truth(self):
        return self._truth

    @truth.setter
    def truth(self, truth):
        if not isinstance(truth, (tuple, list, pd.DataFrame, np.ndarray)):
            raise ValueError("truth must be a tuple, list, pandas dataframe or numpy array")
        self._truth = truth
        
    #define your metrics in a list with each entry being a callable with truth, prediction and **kwargs as arguments
    @property
    def metrics(self):
        return self._metrics
    
    @metrics.setter
    def metrics(self, metrics):
        if not isinstance(metrics, list):
            raise ValueError("metrics must be a list of callables")
        
        for metric in metrics:
            if not callable(metric):
                raise ValueError("metrics must be a list of callables")
            
        self._metrics = metrics
        
    def calculate_metrics(self, *args, **kwargs):
        results = {}
        for metric in self.metrics:
            metric_kwargs = self.kwargs.copy()
            metric_kwargs.update(kwargs)
            results[metric.__name__] = metric(self.truth, self.predictions, *args, **metric_kwargs) # TODO: Should I validate the .__name__ method in the setter?
        return results

# # Example of using monitor
# from sklearn.metrics import precision_score, recall_score, f1_score #note: these all require the 'average' parameter
# from sklearn.preprocessing import StandardScaler
# from model import Model
# from sklearn.linear_model import RidgeClassifier
# from sklearn.datasets import load_iris

# iris = load_iris()
# X = StandardScaler().fit_transform(iris.data)  # iris.data
# y = iris.target
# ridge_classifier= RidgeClassifier()
# model = Model(ridge_classifier)
# model.fit(X=X, y=y)

# predictions = model.predict(X)
# monitor = Monitor(predictions, y, [precision_score, recall_score, f1_score], average='macro')

# # Print monitor results
# print(f"Monitor results: {monitor.calculate_metrics()}")