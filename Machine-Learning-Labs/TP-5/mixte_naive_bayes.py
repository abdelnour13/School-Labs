import numpy as np
import pandas as pd

class MixteNaiveBayes:
    
    def __init__(self, categorical):
        self.model = None
        self.categorical = categorical
        self.classes = None
        self.columns = None

    def get_params(self):
        return {
            'model': self.model,
            'categorical': self.categorical,
            'classes': self.classes
        }
        
    def fit(self, X, Y):
        
        self.model = []
        self.classes, counts = np.unique(Y, return_counts = True)

        self.columns = X.columns if isinstance(X,pd.DataFrame) else np.arange(X.shape[1])
        
        for i in self.columns:

            if isinstance(X,pd.DataFrame):
                column = X[i]
            else:
                column =  X[:,i]
                        
            if i in self.categorical:
                p = self.__handle_categorical_attribute__(column, Y)
            else:
                p = self.__handle_numerical_attribute__(column, Y)
                
            self.model.append(p)
        
        probabilities = dict(zip(self.classes, counts / counts.sum()))
        self.model.append(probabilities)
        
            
    def predict(self, X):
        
        p = self.predict_proba(X)
        return self.classes[np.argmax(p, axis = 0)]
    
    def predict_proba(self, X):
        print("hi")
        if self.model is None:
            raise Exception("fit was not called")
        
        y_hat = []
        
        for x in X:
            probailities = []
            for group in self.classes:
                p = 1
                for i in range(X.shape[1]):
                    if self.columns[i] in self.categorical:
                        p = p * self.model[i][group][x[i]]
                    else:
                        std = self.model[i][group]['std']
                        mean = self.model[i][group]['mean']
                        p = p * self.__gauss__(mean, std, x[i])
                p = p * self.model[-1][group]
                probailities.append(p)
            probailities = np.array(probailities)
            probailities = probailities / probailities.sum()
            y_hat.append(probailities)

        return np.array(y_hat)
        
    def __gauss__(self, mean, std, x):
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (np.sqrt(2 * np.pi) * std)
            
    def __handle_categorical_attribute__(self, column, Y):
        
        probabilities = {}
        values = np.unique(column)
        
        for group in self.classes:
            mask = Y == group
            _Y = Y[mask]
            _column = column[mask]
            probabilities[group] = {}
            for value in values:
                probabilities[group][value] = _Y[_column == value].shape[0] / _Y.shape[0]
                
        return probabilities
    
    def __handle_numerical_attribute__(self, column, Y):
        
        probabilities = {}
        
        for group in self.classes:
            
            _column = column[Y == group]
            
            mean = np.mean(_column)
            std = np.std(_column, ddof=1)
            
            probabilities[group] = {
                'mean': mean,
                'std': std
            }
            
        return probabilities