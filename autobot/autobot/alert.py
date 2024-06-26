# A data class that defines the configuration for triggering an MLOps alert
from dataclasses import dataclass, field
from typing import Callable, List, Any
from sklearn.metrics import precision_score, recall_score, f1_score

@dataclass
class Alert:
    name: str
    metric: Callable[[Any, Any], float]
    threshold: float
    alert_type: str
    description: str = ""
    enabled: bool = True

# Example usage
def example_usage():
    alerts = [
        Alert(
            name="Low Precision Alert",
            metric=precision_score,
            threshold=0.70,
            alert_type="log",
            description="Triggered when precision falls below 70%",
            enabled=True
        ),
        Alert(
            name="High F1 Score Alert",
            metric=f1_score,
            threshold=0.90,
            alert_type="email",
            description="Triggered when F1 score exceeds 90%"
        )
    ]

    # Example predictions and truth
    predictions = [0, 1, 1, 0, 1, 1, 0, 0]
    truth = [0, 1, 0, 0, 1, 0, 0, 1]

    for alert in alerts:
        if not alert.enabled:
            continue

        metric_value = alert.metric(truth, predictions, average='macro' if alert.metric in {precision_score, recall_score, f1_score} else None)

        if alert.metric in {precision_score, recall_score, f1_score} and metric_value is not None and metric_value < alert.threshold:
            print(f"Alert: {alert.name} - {alert.description} - Metric Value: {metric_value}")

# Example call
example_usage()