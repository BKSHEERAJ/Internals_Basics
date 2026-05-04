import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "genomeflow-sequencing-hours"
MODEL_NAME = "genomeflow_model"


def register_and_promote():
    client = MlflowClient()

    # Get experiment
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise Exception("Experiment not found")

    # Get best run (lowest RMSE)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1
    )

    if not runs:
        raise Exception("No runs found")

    best_run = runs[0]
    run_id = best_run.info.run_id

    # Register model from run artifacts
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    # Transition to Staging
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Staging"
    )

    print("Model registered and moved to Staging")
    print("Version:", result.version)


if __name__ == "__main__":
    register_and_promote()