#Tests for the monitor class
import pytest
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error
from autobot.monitor import Monitor


@pytest.fixture
def predictions():
    return np.random.random(100)

@pytest.fixture
def truth():
    return np.random.random(100)

@pytest.fixture
def metrics():
    return [mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error]
    

def test_monitor_instantiation(predictions, truth, metrics):
    monitor = Monitor(predictions, truth, metrics)
    assert isinstance(monitor, Monitor)
    
def test_monitor_results(predictions, truth, metrics):
    monitor = Monitor(predictions, truth, metrics)
    results = monitor.calculate_metrics()
    assert isinstance(results, dict)
    
def test_monitor_invalid_predictions(predictions):
    with pytest.raises(ValueError):
        monitor = Monitor('invalid_predictions', truth, metrics)
        
def test_monitor_invalid_truth(truth):
    with pytest.raises(ValueError):
        monitor = Monitor(predictions, 'invalid_truth', metrics)
        
def test_monitor_invalid_metrics(metrics):
    with pytest.raises(ValueError):
        monitor = Monitor(predictions, truth, 'invalid_metrics')   
