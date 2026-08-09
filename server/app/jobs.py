def run_training_job(experiment_id: str) -> None:
    from server.trainer.run import run_experiment

    run_experiment(experiment_id)
