# Computing the Feedback Capacity of Finite State Channels using Reinforcement Learning

This repository contains an implementation of feedback capacity estimator of an unifilar finite-state-channel using reinforcement learning, specifically policy optimization technique.

## Prerequisites

The code is compatible with a tensorflow 2.0 environment.
If you use a docker, you can pull the following docker image

```
docker pull tensorflow/tensorflow:latest-gpu-py3
```


## Running the code

The parameters of the channels are in the file `example.json`. Modify this file to set your own values for the parameters.

To run the code:
```
python ./example.py --exp_name <simulation_name> --config ./configs/example.json
```


## Authors

* **Ziv Aharoni** 
* **Oron Sabag** 
* **Haim Permuter** 


## License

This project is licensed under Apache License 2.0 - see the [LICENSE.md](LICENSE.md) file for details

