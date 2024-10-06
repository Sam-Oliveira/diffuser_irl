# Inverse Reinforcement Learning using Diffusion models in Trajectory Space

MSc Thesis on using the [Diffuser](https://arxiv.org/abs/2205.09991) for Inverse Reinforcement Learning.  
This repo results from a fork from the original [Diffuser repository](https://github.com/jannerm/diffuser). 
Development of our method for all environments was done in the original maze2d branch, and merged into the main branch at the end of the project.
The "cluster" branch contains the code for machines with CUDA.

<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1M3ydOSOkQenYHwZZajntXkzwea3SkFg4" width="60%" title="Value-Guided Diffuser Diagram">
</p>

**Updates**
- 06/10/2024: Merged development into main branch (from maze2d branch).

## Installation

#### Conda environment

```
conda env create -f environment.yml
conda activate diffuser
pip install -e .
```

#### Mujoco Installation

Download mujoco210 from https://github.com/google-deepmind/mujoco/releases/tag/2.1.0 , extract it and copy it to ~/.mujoco/mujoco210. Download mujoco key file from https://www.roboti.us/file/mjkey.txt and add it to "~/.mujoco"

Run the folllowing three commands:

'''
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
'''

Then add the following three lines to '.bashrc'.

'''
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_PRELOAD=$LD_PRELOAD:~/miniconda3/envs/diffuser/lib/libstdc++.so.6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
'''

## Running Code

The 'scripts/' folder contains all the scripts to generate the results presented in the MSc Thesis. The 'u_maze' folder presents the scripts for experiments on the U-Maze Maze2D environment, the 'large_maze' contains the scripts for experiments on the Large Maze Maze2d environment, the 'locomotion' contains the scripts for the Mujoco Locomotion environments (including HalfCheetah), and 'evaluations' contains code for performance (reward and ERC) analysis, and visualisation of learnt behaviour.

#### Diffuser Training

To train the Base Diffuser, run the appropriate script for your choice of environment.

'''
python scripts/{CHOICE_OF_ENV}/train.py
'''

For HalfCheetah, you can add a dataset flag such as '--dataset halfcheetah-medium-replay-v2' for your choice of dataset.

### Reward Model Learning

To learn a reward model, firstly run the set-up script 'initiate_value.py', followed by either 'guided_learning_reward.py' for MSE Loss or 'guided_learning_mmd.py' for MMD Loss.

'''
python scripts/{CHOICE_OF_ENV}/initiate_value.py
python scripts/{CHOICE_OF_ENV}/guided_learning_reward.py  # MSE Loss
python scripts/{CHOICE_OF_ENV}/guided_learning_mmd.py  # MMD Loss
'''

### Guided Planning with Learnt Reward Model

To create trajectories/rollouts obtained from planning (using the Diffuser) with a learnt reward model, run:

'''
python scripts/{CHOICE_OF_ENV}/guided_learnt_reward.py # To generate only 1 trajectory
python scripts/{CHOICE_OF_ENV}/parallel/guided_learnt_reward.py # To generate multiple trajectories
'''

## Acknowledgements

This project was done as part of my MSc Thesis for the MSc in Machine Learning at University College London (UCL). This work was done as part of [Ilija Bogunovic's Group](https://ilijabogunovic.com/), and under the supervision of Dr Bogunovic, William Bankes and Lorenz Wolf.