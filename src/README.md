# CS 169 Final Project

Generates actors powered by feed-forward neural networks, trained by genetic algorithms. In particular, targets the Gym/MuJoCo Half-Cheetah environment.

## Installation

First, install Gym and MuJoCo according to their installation instructions, found at https://github.com/openai/gym#installation and https://github.com/openai/mujoco-py#install-mujoco.

Next, optionally install joblib using `pip install joblib==0.12`. **Do not install joblib 0.13, as it causes a memory leak which can crash the computer.** Installing joblib lets the algorithm train in parallel, dramatically accelerating evolution.

## Execution

Once installed, run the Half-Cheetah environment by executing `example_mujo_nn.py`.
