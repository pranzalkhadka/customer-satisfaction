from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/home/pranjal/Downloads/customer-satisfaction/data/olist_customers_dataset.csv")


#mlflow ui --backend-store-uri "file:/home/pranjal/.config/zenml/local_stores/1c986883-7c2f-476f-bc3e-d1544a5304dd/mlruns"