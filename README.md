<div style="display: flex; align-items: center; background-color: rgba(0,0,0,0.62); padding: 5px;">
    <div>
        <h1 style="color: white; margin: 0; font-size: 27px;">SayCanPay: Heuristic Planning with Large Language Models using Learnable Domain Knowledge</h1>
    </div>
</div>

<p align="center">
    <a href="https://rishihazra.github.io/SayCanPay/" target="_blank">
        <img alt="Documentation" src="https://img.shields.io/website/https/rishihazra.github.io/SayCanPay?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="//arxiv.org/abs/2308.12682" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-2308.12682-red">
    </a>
</p>

<p align="center">
  <img src="merged-saycanpay.gif" alt="merged-saycanpay">
</p>

 Official Code of our AAAI 2024 paper.

[//]: # (### [[Preprint]]&#40;https://arxiv.org/pdf/2308.12682.pdf&#41; | [[Website]]&#40;https://rishihazra.github.io/SayCanPay/&#41;)

## Quickstart

### 1. Install all requirements.

Using Docker: Download and run the docker image. Ensure you have [Docker](https://docs.docker.com/desktop/install/ubuntu/) installed.
```shell
docker pull rishihazra/llm-dgx:torch-2.0.0-latest
docker run -it rishihazra/llm-dgx:torch-2.0.0-latest
```
You can also build it from scratch using docker/Dockerfile.

Alternatively, you can create a conda environment and install all requirements.
```shell
conda create -n saycanpay_env python=3
source activate saycanpay_env
pip install -r docker/requirements.txt
```

### 2. Now define environment paths
```shell
export RAVENS_ROOT=$(pwd)/llm_planning/ravens
export BabyAI=$(pwd)/llm_planning/babyai
export VIRTUALHOME=$(pwd)/llm_planning/virtualhome/src/virtualhome
export PLANNER=$(pwd)/llm_planning/planLM
cd $PLANNER
```
To use the language interface in BaByAI with our high-level actions, run this additional line:
```shell
cp -f ../babyai/unlock.py /opt/conda/lib/python3.8/site-packages/minigrid/envs/babyai/unlock.py  # for python 3.8
```

## Data Split Generation
* To generate data, say for instance for BabyAI (using multiprocessing)
```shell
python3 babyai_inference.py \  # change 
        planner.agent_type=oracle \  # generates oracle trajectories
        domain.mode=train \  # data split type
        save_data=True \  # to save or not to save data
        parallel=True \  # set False if multiprocessing is not required
        n=400  # number of trajectories in the split
python3 babyai_inference.py \
        planner.agent_type=oracle \
        domain.mode=test-success \
        save_data=True \
        parallel=True \
        n=100
python3 babyai_inference.py \
        planner.agent_type=oracle \
        domain.mode=test-optimal \
        save_data=True \
        parallel=True \
        n=100
python3 babyai_inference.py \
        planner.agent_type=oracle \
        domain.mode=test-generalize \
        save_data=True \
        parallel=True \
        n=100
```

For Ravens:
```shell
python3 ravens_inference.py \
        planner.agent_type=oracle \
        task=towers-of-hanoi-seq \  # put-block-in-bowl
        mode=train \
        save_data=True \
        parallel=True \
        n=800  # 100 for test splits
```

For VirtualHome, we provide a sub-set of processed crowdsourced plans in $VIRTUALHOME/data/oracle-plans.

## Train Can, Pay models
Training framework is Distributed Data Parallel over multi-GPU.

```shell
python3 train/babyai_train.py \  # ravens_train.py, virtualhome_train.py
        train.task=pickup \  # towers-of-hanoi-seq (Ravens), put-block-in-bowl (Ravens), remove for VirtualHome
        train.model=can \  # pay for Pay model
        train.max_epochs=30 \  # 20 for Pay model
        train.batch_size=60
```

## For Inference (Plan Generation)

```shell
python3 ravens_inference.py \  # virtualhome_inference.py, # babyai_inference.py
        task=put-block-in-bowl \
        mode=test-success \  # test-optimal, test-generalize
        planner.agent_type=lm \  
        planner.model_name=flan_t5 \  # vicuna
        planner.decoding_type=beam_action \  # greedy_token, greedy_action
        planner.decoding_score=say_can_pay  # say, say_can (remove for greedy_token decoding_type)
```

## References
[1] [Transporter Networks: Rearranging the Visual World for Robotic Manipulation, Zeng et al., CoRL 2020](https://arxiv.org/abs/2010.14406)  
[2] [BabyAI: First Steps Towards Grounded Language Learning With a Human In the Loop, Chevalier-Boisvert et al., ICLR 2019](https://openreview.net/pdf?id=rJeXCo0cYX)  
[3] [Virtualhome: Simulating household activities via programs, Puig et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Puig_VirtualHome_Simulating_Household_CVPR_2018_paper.pdf)

### To cite our paper:
```bibtex
@inproceedings{hazra2024saycanpay,
  title={SayCanPay: Heuristic Planning with Large Language Models using Learnable Domain Knowledge},
  author={Hazra, Rishi and Dos Martires, Pedro Zuidberg and De Raedt, Luc},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={18},
  pages={20123--20133},
  year={2024}
}
```


