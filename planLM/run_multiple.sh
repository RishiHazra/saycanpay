#!/bin/bash

########################################################
#              BabyAI: Pickup
########################################################
# generate train-test files
#python3 babyai_inference.py \
#        planner.agent_type=oracle \
#        domain.mode=train \
#        save_data=True \
#        parallel=True \
#        n=400
#python3 babyai_inference.py \
#        planner.agent_type=oracle \
#        domain.mode=test-success \
#        save_data=True \
#        parallel=True \
#        n=50
#python3 babyai_inference.py \
#        planner.agent_type=oracle \
#        domain.mode=test-optimal \
#        save_data=True \
#        parallel=True \
#        n=50
#python3 babyai_inference.py \
#        planner.agent_type=oracle \
#        domain.mode=test-generalize \
#        save_data=True \
#        parallel=True \
#        n=50
#
## run train can-pay
#python3 train/babyai_train.py \
#        train.task=pickup \
#        train.model=can \
#        train.max_epochs=30 \
#        train.batch_size=60
#python3 train/babyai_train.py \
#        train.task=pickup \
#        train.model=pay \
#        train.max_epochs=20 \
#        train.batch_size=60

############################################################
# babyai
# vicuna: test-success
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-success \
#        planner.decoding_type=greedy_token
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-success \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-success \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-success \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-success \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-success \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-success \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay

# babyai
# vicuna: test-optimal
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-optimal \
#        planner.decoding_type=greedy_token
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-optimal \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-optimal \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-optimal \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-optimal \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-optimal \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-optimal \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay

# babyai
# vicuna: test-generalize
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-generalize \
#        planner.decoding_type=greedy_token
python3 babyai_inference.py \
        planner.agent_type=lm \
        planner.model_name=vicuna \
        domain.mode=test-generalize \
        planner.decoding_type=greedy_action \
        planner.decoding_score=say
python3 babyai_inference.py \
        planner.agent_type=lm \
        planner.model_name=vicuna \
        domain.mode=test-generalize \
        planner.decoding_type=greedy_action \
        planner.decoding_score=say_can
python3 babyai_inference.py \
        planner.agent_type=lm \
        planner.model_name=vicuna \
        domain.mode=test-generalize \
        planner.decoding_type=greedy_action \
        planner.decoding_score=say_can_pay
python3 babyai_inference.py \
        planner.agent_type=lm \
        planner.model_name=vicuna \
        domain.mode=test-generalize \
        planner.decoding_type=beam_action \
        planner.decoding_score=say
python3 babyai_inference.py \
        planner.agent_type=lm \
        planner.model_name=vicuna \
        domain.mode=test-generalize \
        planner.decoding_type=beam_action \
        planner.decoding_score=say_can
python3 babyai_inference.py \
        planner.agent_type=lm \
        planner.model_name=vicuna \
        domain.mode=test-generalize \
        planner.decoding_type=beam_action \
        planner.decoding_score=say_can_pay

##########################################################
# babyai
# flan_t5: test-success
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-success \
#        planner.decoding_type=greedy_token
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-success \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-success \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-success \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-success \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-success \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-success \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay


# babyai
# flan_t5: test-optimal
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-optimal \
#        planner.decoding_type=greedy_token
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-optimal \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-optimal \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-optimal \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-optimal \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-optimal \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 babyai_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-optimal \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay

# babyai
# flan_t5: test-generalize
python3 babyai_inference.py \
        planner.agent_type=lm \
        planner.model_name=flan_t5 \
        domain.mode=test-generalize \
        planner.decoding_type=greedy_token
python3 babyai_inference.py \
        planner.agent_type=lm \
        planner.model_name=flan_t5 \
        domain.mode=test-generalize \
        planner.decoding_type=greedy_action \
        planner.decoding_score=say
python3 babyai_inference.py \
        planner.agent_type=lm \
        planner.model_name=flan_t5 \
        domain.mode=test-generalize \
        planner.decoding_type=greedy_action \
        planner.decoding_score=say_can
python3 babyai_inference.py \
        planner.agent_type=lm \
        planner.model_name=flan_t5 \
        domain.mode=test-generalize \
        planner.decoding_type=greedy_action \
        planner.decoding_score=say_can_pay
python3 babyai_inference.py \
        planner.agent_type=lm \
        planner.model_name=flan_t5 \
        domain.mode=test-generalize \
        planner.decoding_type=beam_action \
        planner.decoding_score=say
python3 babyai_inference.py \
        planner.agent_type=lm \
        planner.model_name=flan_t5 \
        domain.mode=test-generalize \
        planner.decoding_type=beam_action \
        planner.decoding_score=say_can
python3 babyai_inference.py \
        planner.agent_type=lm \
        planner.model_name=flan_t5 \
        domain.mode=test-generalize \
        planner.decoding_type=beam_action \
        planner.decoding_score=say_can_pay

########################################################
#              Ravens: Tower-of-hanoi
########################################################
# generate train-test files
#export RAVENS_ROOT=$(pwd)/saycanpay/ravens
#export PLANNER=$(pwd)/saycanpay/planLM
#cd $PLANNER

#python3 ravens_inference.py \
#        planner.agent_type=oracle \
#        task=towers-of-hanoi-seq \
#        mode=train \
#        save_data=True \
#        parallel=True \
#        n=800
#python3 ravens_inference.py \
#        planner.agent_type=oracle \
#        task=towers-of-hanoi-seq \
#        mode=test-success \
#        save_data=True \
#        parallel=True \
#        n=50
#python3 ravens_inference.py \
#        planner.agent_type=oracle \
#        task=towers-of-hanoi-seq \
#        mode=test-optimal \
#        save_data=True \
#        parallel=True \
#        n=50
#python3 ravens_inference.py \
#        planner.agent_type=oracle \
#        task=towers-of-hanoi-seq \
#        mode=test-generalize \
#        save_data=True \
#        parallel=True \
#        n=50
#
# run train can-pay
#python3 train/ravens_train.py \
#        train.task=towers-of-hanoi-seq \
#        train.model=can \
#        train.max_epochs=40 \
#        train.batch_size=60
#        wandb.task=hanoi
#python3 train/ravens_train.py \
#        train.task=towers-of-hanoi-seq \
#        train.model=pay \
#        train.max_epochs=30 \
#        train.batch_size=60
#        wandb.task=hanoi

############################################################
# ravens: towers-of-hanoi-seq
# vicuna: test-success
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_token
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay

# ravens: towers-of-hanoi-seq
# vicuna: test-optimal
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_token
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay


# ravens: towers-of-hanoi-seq
# vicuna: test-generalize
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_token
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay

############################################################
# ravens: towers-of-hanoi-seq
# flan_t5: test-success
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_token
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay

# ravens: towers-of-hanoi-seq
# flan_t5: test-optimal
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_token
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay

# ravens: towers-of-hanoi-seq
# flan_t5: test-generalize
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_token
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=towers-of-hanoi-seq \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay


########################################################
#              Ravens: Put-Block-in-Bowl
########################################################
# generate train-test files
#export RAVENS_ROOT=$(pwd)/saycanpay/ravens
#export PLANNER=$(pwd)/saycanpay/planLM
#cd $PLANNER

#python3 ravens_inference.py \
#        planner.agent_type=oracle \
#        task=put-block-in-bowl \
#        mode=train \
#        save_data=True \
#        parallel=True \
#        n=800
#python3 ravens_inference.py \
#        planner.agent_type=oracle \
#        task=put-block-in-bowl \
#        mode=test-success \
#        save_data=True \
#        parallel=True \
#        n=50
#python3 ravens_inference.py \
#        planner.agent_type=oracle \
#        task=put-block-in-bowl \
#        mode=test-optimal \
#        save_data=True \
#        parallel=True \
#        n=50
#python3 ravens_inference.py \
#        planner.agent_type=oracle \
#        task=put-block-in-bowl \
#        mode=test-generalize \
#        save_data=True \
#        parallel=True \
#        n=50
#
# run train can-pay
#python3 train/ravens_train.py \
#        train.task=put-block-in-bowl \
#        train.model=can \
#        train.max_epochs=50 \
#        train.batch_size=50
#        wandb.task=blocks
#python3 train/ravens_train.py \
#        train.task=put-block-in-bowl \
#        train.model=pay \
#        train.max_epochs=40 \
#        train.batch_size=60
#        wandb.task=blocks

#############################################

# ravens: put-block-in-bowl
# vicuna: test-success
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_token
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay

# ravens: put-block-in-bowl
# vicuna: test-optimal
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_token
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay

# ravens: put-block-in-bowl
# vicuna: test-generalize
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_token
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay

############################################################

# ravens: put-block-in-bowl
# flan_t5: test-success
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_token
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-success \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay

# ravens: put-block-in-bowl
# flan_t5: test-optimal
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_token
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-optimal \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay


# ravens: put-block-in-bowl
# flan_t5: test-generalize
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_token
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 ravens_inference.py \
#        task=put-block-in-bowl \
#        mode=test-generalize \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay

########################################################
#              VirtualHome
########################################################
# generate train-test files
#export VIRTUALHOME=$(pwd)/saycanpay/virtualhome/src/virtualhome
#export PLANNER=$(pwd)/saycanpay/planLM
#cd $PLANNER

## run train can-pay
#python3 train/virtualhome_train.py \
#        train.model=can \
#        train.max_epochs=30 \
#        train.batch_size=30
#python3 train/virtualhome_train.py \
#        train.model=pay \
#        train.max_epochs=20 \
#        train.batch_size=30

############################################################
# virtualhome
# vicuna: test
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test \
#        planner.decoding_type=greedy_token
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay

# virtualhome
# vicuna: test-generalize
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-generalize \
#        planner.decoding_type=greedy_token
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-generalize \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-generalize \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-generalize \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-generalize \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-generalize \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=vicuna \
#        domain.mode=test-generalize \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay
############################################################

# virtualhome
# flan_t5: test
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test \
#        planner.decoding_type=greedy_token
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay

# virtualhome
# flan_t5: test-generalize
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-generalize \
#        planner.decoding_type=greedy_token
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-generalize \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-generalize \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-generalize \
#        planner.decoding_type=greedy_action \
#        planner.decoding_score=say_can_pay
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-generalize \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-generalize \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can
#python3 virtualhome_inference.py \
#        planner.agent_type=lm \
#        planner.model_name=flan_t5 \
#        domain.mode=test-generalize \
#        planner.decoding_type=beam_action \
#        planner.decoding_score=say_can_pay
