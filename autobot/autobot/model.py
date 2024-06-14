class Model:
    def __init__(self, model):
        self.model = model
        self._ensure_predict_method()
        self._ensure_fit_method()

    def _ensure_predict_method(self):
        if not hasattr(self.model, 'predict'):
            raise ValueError(f"The model {type(self.model)} does not have a 'predict' method.")
        
    def _ensure_fit_method(self):
        if not hasattr(self.model, 'fit'):
            raise ValueError(f"The model {type(self.model)} does not have a 'fit' method.")
        
    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

# from sklearn.linear_model import LogisticRegression

# logreg = LogisticRegression()
# model = Model(logreg)

# model.fit(X=[[1, 2, 3], [4, 5, 6]], y =[0, 1]) 

# # Example usage:
# import torch.nn as nn
# import torch

# # PyTorch model (with a custom predict method)
# class MyPyTorchModel(nn.Module):
#     def __init__(self):
#         super(MyPyTorchModel, self).__init__()
#         self.linear = nn.Linear(10, 1)

#     def forward(self, x):
#         return self.linear(x)

#     def predict(self, x):
#         with torch.no_grad():
#             return self.forward(x)

# pytorch_model = MyPyTorchModel()
# model = Model(pytorch_model)
# print(model.predict(torch.randn(5, 10)))