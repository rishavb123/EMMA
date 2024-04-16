# External Model Motivated Agent (EMMA)

The goal of this research is to create an agent that optimizes an external model as quickly as possible while still performing well on its given task. Further, the agent should be able to quickly adapt to the changes to the environment, quickly relearning any missing information in the external model.

## Installation

To create a conda environment for this package use the following commands to do so

```bash
conda create -n emma_env python=3.11 -y
conda activate emma_env
```

Install the version of pytorch that is compatible with your hardware using the pip or conda command from their [website](https://pytorch.org/get-started/locally/).

Then to install this package and its dependencies use the following commands:

```bash
git clone https://github.com/rishavb123/EMMA.git
cd EMMA
pip install -e .
```

## Configs

The configs for this experiment runner should all be based off the `base_emma` config (located in the `configs` folder). This config is based on the `RLConfig` specified [here](https://github.com/rishavb123/ExperimentLab/blob/main/experiment_lab/experiments/rl/config.py) and is formally specified in the `emma/experiment.py` file.

Within this config, it is possible to construct experiments with different poi field models, poi exploration algorithms, different poi embedding learners (and different state samplers for them), etc. These objected should be specified as listed [here](https://hydra.cc/docs/advanced/instantiate_objects/overview/).

## Run Script

To run an EMMA experiment, use the following command:

```bash
./scripts/run.sh --config-name {insert config name here}
```

The entry point is a hydra script, so any hydra overrides will work with this run command. An example of this is,

```bash
./scripts/run.sh --config-name dev poi_emb_learner.poi_emb_size=0 policy_cls=stable_baselines3.ppo.MlpPolicy
```

to run a `dev` experiment with `poi_emb_size` set to 0 and using an `MlpPolicy`.

## POI Algorithms

### POI Field Models

To create a new POI field models, go to the `emma/poi/poi_field.py` file and create a new subclass of `POIFieldModel`. The important method for this class to fill in is the `calculate_poi_values` method. This function takes in a tensor representing the model input tensor with shape `(batch_size, *inp_shape)` and should return a numpy array of shape `(batch_size, )` with a poi value for each input. This function will be called by the poi exploration method to modify agent behavior. Feel free to look at the other classes in the file for examples.

### POI Exploration Algorithm

To create a new POI exploration algorithm, go to the `emma/poi/poi_exploration.py` file and create a new subclass of `POIPPO`. This is a subclass of the `PPO` class from `stable_baselines3`. This class can be changed in any way to modify agent behavior based on the poi model. Note that the poi values can be accessed by calling the `self.poi_model.calculate_poi_values` function. To construct the model input, use 

```python
model_inp = (
    self.poi_model.external_model_trainer.rollout_to_model_input(
        env=env,
        rollout_buffer=rollout_buffer,
        info_buffer=self.info_buffer,
    )
)
```

which makes use of the info buffer that `POIPPO` adds.

Feel free to look at the other classes in the file for examples.