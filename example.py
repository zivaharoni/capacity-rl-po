import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print("MESSAGE", e)
import mlflow
from models.example_model import ActorModel, IsingModel
from algorithms.example_trainer import POTrainer
from utils.config import process_config
from utils.utils import get_args



def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    config = process_config(get_args())

    print("MLflow Version:", mlflow.version.VERSION)
    mlflow.set_tracking_uri('/common_space_docker/storage_1TSSD/ziv/capacity-rl-po/mlruns')
    print("Tracking URI:", mlflow.tracking.get_tracking_uri())

    experiment_name = config.exp_name
    print("experiment_name:", experiment_name)
    mlflow.set_experiment(experiment_name)

    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    print("experiment_id:", experiment_id)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_param("channel_cardinality", config.channel_cardinality)
        mlflow.log_param("lr", config.learning_rate)
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("hidden_size", config.hidden_size)
        mlflow.log_param("num_epochs", config.num_epochs)
        mlflow.log_param("unroll_steps", config.unroll_steps)

        # create an instance of the model you want
        actor = ActorModel(config)

        # create your data generator§
        env = IsingModel(config)

        # create tensorboard logger
        # logger = Logger(config)

        # create trainer and pass all the previous components to it
        trainer = POTrainer(actor, env, config)

        #load model if exists
        # model.load(sess)

        # here you train your model
        trainer.train()

        if config.save_model:
            actor.model.save("./tmp/model")
            mlflow.log_artifact("./tmp/model")


if __name__ == '__main__':
    main()
