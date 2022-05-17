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
from models.example_model import ActorModel, IsingModel, TrapdoorModel, RLL_0_1
from algorithms.example_trainer import POTrainer
from utils.config import process_config
from utils.utils import get_args



def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    config = process_config(get_args())

    experiment_name = config.exp_name
    print("experiment_name:", experiment_name)
    # create your data generator
    if config.channel == "ising":
        env = IsingModel(config)
    elif config.channel == "trapdoor":
        env = TrapdoorModel(config)
    elif config.channel == "rll":
        env = RLL_0_1(config, eps=0.5)
    else:
        raise ValueError("invalid channel name")
    # create an instance of the model you want
    actor = ActorModel(config, env)


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


if __name__ == '__main__':
    main()
