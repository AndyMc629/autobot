from autobot.alert import Alert
from autobot.monitor import Monitor
from autobot.model import Model
from autobot.simulation import Simulation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score

import logging

def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#TODO: Revamp the whole api as it's too confusing, simplify drastically!
def main():
    configure_logging()
    
    logging.info("Defining model...")
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)  # iris.data
    y = iris.target
    logreg = LogisticRegression()
    
    logging.info("Fitting autobot wrapped model...")
    model = Model(logreg)
    model.fit(X=X, y=y)
    
    logging.info("Simulating prediction...")
    simulation = Simulation(model)
    simulation.simulate_prediction(prediction_input=X)
    logging.info("Simulating truth...")
    simulation.simulate_truth()

    logging.info("Calculating monitoring metrics...")      
    monitor = Monitor(simulation, truth=y, metrics=f1_score)
    monitor.calculate_metrics()

    logging.info("Sending alert...")
    alert = Alert(monitor)
    alert.send_alert()

    logging.info("Simulation complete.")

if __name__=="__main__":
    main()
    