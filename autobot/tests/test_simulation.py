from autobot.simulation import Simulation
from autobot.model import Model
import pytest

@pytest.fixture
def data():
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)  # iris.data
    y = iris.target
    return [X,y]

@pytest.fixture
def model(data):
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    model = Model(logreg)
    model.fit(X=data[0] , y=data[1])
    return model

def test_simple_simulation(model, data):
    sim = Simulation(model)
    print(sim.simulate_prediction(prediction_input=data[0]))
    assert(True)

def test_simulation_with_kwargs(model, data):
    sim = Simulation(model, prediction_input=data[0])
    print(sim.simulate_prediction())
    assert(True)
    
def test_simple_simulation_with_valid_model(data, model):
    # Test case: Verify that the simulation can be created with a valid model and simulate_prediction returns the expected result.
    sim = Simulation(model)
    predictions = sim.simulate_prediction(prediction_input=data[0])
    assert(len(predictions) == len(data[0]))

def test_simple_simulation_with_invalid_model(data):
    # Test case: Verify that a ValueError is raised when an invalid model is provided.
    with pytest.raises(ValueError):
        sim = Simulation("invalid_model")
        sim.simulate_prediction(prediction_input=data[0])

def test_simple_simulation_with_no_input_data(model):
    # Test case: Verify that a ValueError is raised when no input data is provided.
    sim = Simulation(model)
    with pytest.raises(ValueError):
        sim.simulate_prediction()
        
def test_truth_data(model, data):
    sim = Simulation(model, truth=data[1])
    assert(len(sim.simulate_truth()) == len(data[1]))
    assert(sim.simulate_truth().all() == data[1].all())

def test_truth_data_without_truth_data(model):
    sim = Simulation(model)
    with pytest.raises(ValueError):
        sim.simulate_truth()
        
def test_truth_data_supplied_as_kwargs_at_call(model, data):
    sim = Simulation(model)
    assert(len(sim.simulate_truth(truth=data[1])) == len(data[1]))
    assert(sim.simulate_truth(truth=data[1]).all() == data[1].all())
    
def test_truth_data_with_invalid_model(data):
    with pytest.raises(ValueError):
        sim = Simulation("invalid_model", truth=data[1])
        sim.simulate_truth()
