
from autobot.model import Model

class Simulation(object):
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
    
    # validate that model is of type Model using property and setter.
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if not isinstance(model, Model):
            raise ValueError("model must be of type Model")
        self._model = model

    # Not sure I need this now.
    def __call__(self, *args, **kwargs):
        return self.model(*args, **self.kwargs)
    
    def get_prediction_input(self, *args, **kwargs):
        if 'prediction_input' in kwargs:
            return kwargs['prediction_input']
        elif 'prediction_input' in self.kwargs:
            return self.kwargs['prediction_input']
        else:
            raise ValueError("No prediction input data provided")
            
    def simulate_prediction(self, *args, **kwargs):
        prediction_input = self.get_prediction_input(*args, **kwargs)
        predictions = self.model.predict(prediction_input)        
        return predictions
    
# from model import Model
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import load_iris

# iris = load_iris()
# X = StandardScaler().fit_transform(iris.data)  # iris.data
# y = iris.target
# logreg = LogisticRegression()
# model = Model(logreg)
# model.fit(X=X, y=y)

# sim = Simulation(model)
# print(sim.simulate_prediction(prediction_input=X))