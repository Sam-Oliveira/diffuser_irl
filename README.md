# Inverse Reinforcement Learning using Diffusion models in Trajectory Space

MSc Thesis on using the [Diffuser](https://arxiv.org/abs/2205.09991) for Inverse Reinforcement Learning.
Training and visualizing of diffusion models from [Planning with Diffusion for Flexible Behavior Synthesis](https://diffusion-planning.github.io/).
This branch has the Maze2D experiments and will be merged into main shortly.

<p align="center">
    <img src="https://diffusion-planning.github.io/images/diffuser-card.png" width="60%" title="Diffuser model">
</p>

**Updates**
- 12/09/2022: Diffuser (the RL model) has been integrated into 🤗 Diffusers (the Hugging Face diffusion model library)! See [these docs](https://huggingface.co/docs/diffusers/using-diffusers/rl) for how to run Diffuser using their pipeline.
- 10/17/2022: A bug in the value function scaling has been fixed in [this commit](https://github.com/jannerm/diffuser/commit/3d7361c2d028473b601cc04f5eecd019e14eb4eb). Thanks to [Philemon Brakel](https://scholar.google.com/citations?user=Q6UMpRYAAAAJ&hl=en) for catching it!

## Quickstart

Load a pretrained diffusion model and sample from it in your browser with [scripts/diffuser-sample.ipynb](https://colab.research.google.com/drive/1YajKhu-CUIGBJeQPehjVPJcK_b38a8Nc?usp=sharing).


## Installation

```
conda env create -f environment.yml
conda activate diffuser
pip install -e .
```

## Using pretrained models

### Downloading weights

Download pretrained diffusion models and value functions with:
```
python scripts/train.py --dataset hopper-medium-replay-v2 \
    --horizon 512 --n_diffusion_steps 200
```

The default hyperparameters are listed in [`config/locomotion.py`](config/locomotion.py).
You can override any of them with runtime flags, eg `--batch_size 64`.


## Docker

1. Build the image:
```
docker build -f Dockerfile . -t diffuser
```

2. Test the image:
```
docker run -it --rm --gpus all \
    --mount type=bind,source=$PWD,target=/home/code \
    --mount type=bind,source=$HOME/.d4rl,target=/root/.d4rl \
    diffuser \
    bash -c \
    "export PYTHONPATH=$PYTHONPATH:/home/code && \
    python /home/code/scripts/train.py --dataset halfcheetah-medium-expert-v2 --logbase logs"
```

## Singularity

1. Build the image:
```
singularity build --fakeroot diffuser.sif Singularity.def
```

2. Test the image:
```
singularity exec --nv --writable-tmpfs diffuser.sif \
        bash -c \
        "pip install -e . && \
        python scripts/train.py --dataset halfcheetah-medium-expert-v2 --logbase logs"
```


## Running on Azure

#### Setup

1. Tag the Docker image (built in the [Docker section](#Docker)) and push it to Docker Hub:
```
export DOCKER_USERNAME=$(docker info | sed '/Username:/!d;s/.* //')
docker tag diffuser ${DOCKER_USERNAME}/diffuser:latest
docker image push ${DOCKER_USERNAME}/diffuser
```

3. Update [`azure/config.py`](azure/config.py), either by modifying the file directly or setting the relevant [environment variables](azure/config.py#L47-L52). To set the `AZURE_STORAGE_CONNECTION` variable, navigate to the `Access keys` section of your storage account. Click `Show keys` and copy the `Connection string`.

4. Download [`azcopy`](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10): `./azure/download.sh`

#### Usage

Launch training jobs with `python azure/launch.py`. The launch script takes no command-line arguments; instead, it launches a job for every combination of hyperparameters in [`params_to_sweep`](azure/launch_train.py#L36-L38).


#### Viewing results

To rsync the results from the Azure storage container, run `./azure/sync.sh`.

To mount the storage container:
1. Create a blobfuse config with `./azure/make_fuse_config.sh`
2. Run `./azure/mount.sh` to mount the storage container to `~/azure_mount`

To unmount the container, run `sudo umount -f ~/azure_mount; rm -r ~/azure_mount`


## Reference
```
@inproceedings{janner2022diffuser,
  title = {Planning with Diffusion for Flexible Behavior Synthesis},
  author = {Michael Janner and Yilun Du and Joshua B. Tenenbaum and Sergey Levine},
  booktitle = {International Conference on Machine Learning},
  year = {2022},
}
```


## Acknowledgements

The diffusion model implementation is based on Phil Wang's [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) repo.
The organization of this repo and remote launcher is based on the [trajectory-transformer](https://github.com/jannerm/trajectory-transformer) repo.
