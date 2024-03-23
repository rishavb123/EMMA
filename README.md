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
