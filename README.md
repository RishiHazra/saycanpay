# SayCanPay: Heuristic Planning with Large Language Models using Learnable Domain Knowledge
![combined.gif](..%2F..%2FDownloads%2Fcombined.gif)

## Quickstart

* Build and run the docker image using docker/Dockerfile using
```shell
docker build -t saycanpay_env -f docker/Dockerfile .
docker run saycanpay_env
```
Alternatively, you can create a conda environment and install all requirements.
```shell
conda create -n saycanpay_env python==3
source activate saycanpay_env
pip install -r docker/requirements.txt
```

* Now define environment paths
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


