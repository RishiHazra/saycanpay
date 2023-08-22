import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from abc import ABC, abstractmethod
import re
from transformers import BertTokenizer, BertModel, LlamaForCausalLM, \
    LlamaTokenizer, GenerationConfig, T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
from torch import optim, nn
import lightning.pytorch as pl
from torchmetrics import MetricCollection, Accuracy, F1Score, MeanSquaredError
from data_utils import process_action, ExceededMaxSteps


# roberta = RobertaModel.from_pretrained("roberta-base")
# roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")


class CanModel(pl.LightningModule):
    def __init__(self, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # adding a special token to concat the context and the action
        self.bert_tokenizer.add_special_tokens({'additional_special_tokens': ['[NXT]']})
        self.bert_encoder.resize_token_embeddings(len(self.bert_tokenizer))

        self.classifier = \
            nn.Sequential(nn.Linear(self.bert_encoder.config.hidden_size, 128),
                          nn.Dropout(),
                          nn.ReLU(),
                          nn.Linear(128, 1),
                          nn.Sigmoid()
                          )
        metrics = \
            MetricCollection([Accuracy(task='binary'),
                              F1Score(task='binary')])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

    def encode(self, text):
        """ :returns: roberta pooled encodings[b, 768]"""
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        pooled = self.bert_encoder(**inputs).pooler_output
        return pooled

    def nce_loss(self, pos_probs, neg_probs, neg_sample_probs):
        pos_probs = pos_probs.view(-1)
        neg_probs = neg_probs.view(-1)
        neg_sample_probs = neg_sample_probs.view(-1)
        loss = -torch.log(pos_probs / (pos_probs + neg_probs + neg_sample_probs)).mean()
        return loss

    def configure_optimizers(self):
        # Define two sets of parameters
        classifier_params = list(self.classifier.parameters())
        base_model_params = list(self.bert_encoder.parameters())

        # Create an optimizer that has different learning rates for the classifier and the base model
        optimizer = optim.AdamW([
            {'params': classifier_params, 'lr': self.lr},  # e.g. learning rate for the classifier
            {'params': base_model_params, 'lr': 1e-5}  # e.g. learning rate for the base model
        ], weight_decay=self.weight_decay)
        # optimizer = optim.Adam(self.can_model.parameters(),
        #                        lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        context_pos_action = [c + ' ' + a
                              for c, a in zip(train_batch['context'],
                                              train_batch['pos_action'])]
        context_neg_action = [c + ' ' + a
                              for c, a in zip(train_batch['context'],
                                              train_batch['neg_action'])]
        context_neg_sample_action = [c + ' ' + a
                                     for c, a in zip(train_batch['context'],
                                                     train_batch['neg_sample_action'])]
        pos_probs = self.classifier(self.encode(context_pos_action))
        neg_probs = self.classifier(self.encode(context_neg_action))
        neg_sample_probs = self.classifier(self.encode(context_neg_sample_action))

        loss = self.nce_loss(pos_probs, neg_probs, neg_sample_probs)
        y = torch.cat((torch.ones_like(pos_probs),
                       torch.zeros_like(neg_probs),
                       torch.zeros_like(neg_sample_probs)), dim=0)
        all_probs = torch.cat((pos_probs, neg_probs, neg_sample_probs), dim=0)
        self.train_metrics.update(all_probs, y)
        self.log("can-train-loss", loss, on_epoch=True,
                 batch_size=len(train_batch), prog_bar=True,
                 sync_dist=True)
        return loss

    def training_epoch_end(self, output):
        train_acc, train_f1 = \
            list(self.train_metrics.compute().values())
        self.log('can-train-acc', train_acc, sync_dist=True)
        self.log('can-train-f1', train_f1, sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, val_batch, batch_idx):
        # print(f'Context: {val_batch["context"]} | Pos: {val_batch["pos_action"]} | '
        #       f'Neg: {val_batch["neg_action"]} | Neg Sample: {val_batch["neg_sample_action"]}')
        context_pos_action = [c + ' [NXT] ' + a
                              for c, a in zip(val_batch['context'],
                                              val_batch['pos_action'])]
        context_neg_action = [c + ' [NXT] ' + a
                              for c, a in zip(val_batch['context'],
                                              val_batch['neg_action'])]
        context_neg_sample_action = [c + ' [NXT] ' + a
                                     for c, a in zip(val_batch['context'],
                                                     val_batch['neg_sample_action'])]
        pos_probs = self.classifier(self.encode(context_pos_action))
        neg_probs = self.classifier(self.encode(context_neg_action))
        neg_sample_probs = self.classifier(self.encode(context_neg_sample_action))

        loss = self.nce_loss(pos_probs, neg_probs, neg_sample_probs)
        y = torch.cat((torch.ones_like(pos_probs),
                       torch.zeros_like(neg_probs),
                       torch.zeros_like(neg_sample_probs)), dim=0)
        all_probs = torch.cat((pos_probs, neg_probs, neg_sample_probs), dim=0)
        self.val_metrics.update(all_probs, y)
        self.log("can-val-loss", loss, on_epoch=True,
                 batch_size=len(val_batch), prog_bar=True,
                 sync_dist=True)

    def validation_epoch_end(self, output):
        val_acc, val_f1 = \
            list(self.val_metrics.compute().values())
        self.log('can-valid-acc', val_acc, sync_dist=True)
        self.log('can-valid-f1', val_f1, sync_dist=True)
        self.val_metrics.reset()

    def forward(self, context, action):
        """forward in lightning mode is used only for inference
        :return: Can score for context, action"""
        context_action = context + ' ' + action
        can_score = self.classifier(self.encode(context_action))
        return can_score

    def predict_step(self, predict_batch, batch_idx, dataloader_idx=0):
        context_batch = predict_batch['context']
        action_batch = predict_batch['action']
        # this calls forward
        can_score_batch = self(context_batch, action_batch)
        return can_score_batch


class PayModel(pl.LightningModule):
    def __init__(self, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        # self.base_model = RobertaModel.from_pretrained("roberta-base")
        # self.base_model_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # adding a special token to concat the context and the action
        self.bert_tokenizer.add_special_tokens({'additional_special_tokens': ['[NXT]']})
        self.bert_encoder.resize_token_embeddings(len(self.bert_tokenizer))
        self.value_model = \
            nn.Sequential(nn.Linear(self.bert_encoder.config.hidden_size, 128),
                          nn.Dropout(),
                          nn.ReLU(),
                          nn.Linear(128, 1),
                          nn.Sigmoid()
                          )
        # self.max_length = 64
        self.loss_fn = nn.MSELoss()
        metrics = MeanSquaredError()
        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()

    def encode(self, text):
        """:returns: roberta pooled encodings [b, 768]"""
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        pooled = self.bert_encoder(**inputs).pooler_output
        return pooled

    def configure_optimizers(self):
        value_model_params = list(self.value_model.parameters())
        base_model_params = list(self.bert_encoder.parameters())

        # Create an optimizer that has different learning rates for the value_model and the base model
        optimizer = optim.AdamW([
            {'params': value_model_params, 'lr': self.lr},  # e.g. learning rate for the value_model
            {'params': base_model_params, 'lr': 1e-5}  # e.g. learning rate for the base model
        ], weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        context_action = \
            [c + ' [NXT] ' + a for c, a in zip(train_batch['context'], train_batch['action'])]
        preds = self.value_model(self.encode(context_action))
        target = train_batch['dist']
        self.train_metrics.update(preds.view(-1), target)
        # entropy = - (preds * torch.log(preds)).mean()
        loss = self.loss_fn(preds.view(-1), target)
        # loss += 0.1 * entropy
        self.log("pay-train-loss", loss, on_epoch=True,
                 batch_size=len(train_batch), prog_bar=True,
                 sync_dist=True)
        return loss

    def training_epoch_end(self, output):
        train_mse = self.train_metrics.compute()
        self.log('pay-train-mse', train_mse, sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, val_batch, batch_idx):
        context_action = \
            [c + ' [NXT] ' + a for c, a in zip(val_batch['context'], val_batch['action'])]
        # print(context_action[0])
        preds = self.value_model(self.encode(context_action))
        target = val_batch['dist']
        self.val_metrics.update(preds.view(-1), target)
        loss = self.loss_fn(preds.view(-1), target)
        self.log("pay-val-loss", loss, on_epoch=True,
                 batch_size=len(val_batch), prog_bar=True,
                 sync_dist=True)

    def validation_epoch_end(self, output):
        val_mse = self.val_metrics.compute()
        self.log('pay-val-mse', val_mse, sync_dist=True)
        self.val_metrics.reset()

    def forward(self, context, action):
        """forward in lightning mode is used only for inference
        :return: Pay score for context, action"""
        context_action = context + ' [NXT] ' + action
        pay_score = self.value_model(self.encode(context_action))
        return pay_score

    def predict_step(self, predict_batch, batch_idx, dataloader_idx=0):
        context_batch = predict_batch['context']
        action_batch = predict_batch['action']
        # this calls forward
        pay_score_batch = self(context_batch, action_batch)
        return pay_score_batch


class CustomScorer:
    def __init__(self, decoding_score, model_name):
        super().__init__()
        # hanoi: 'epoch=27-step=140.ckpt', babyai: 'epoch=23-step=144.ckpt'
        # blocks: 'last.ckpt' (vicuna), 'last-v2.ckpt' (flan-t5)
        # virtualhome: 'episode=23-step=816.ckpt'
        can_ckpt = os.path.join(os.environ["CKPT_ROOT"], 'can', 'epoch=23-step=144.ckpt')
        # hanoi: epoch=23-step=120.ckpt (FLAN-T5, gamma=3), babyai: epoch=15-step=96-v2.ckpt
        # blocks: last.ckpt, virtualhome: epoch=15-step=544.ckpt
        pay_ckpt = os.path.join(os.environ["CKPT_ROOT"], 'pay', 'epoch=15-step=96-v2.ckpt')
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.decoding_score = decoding_score  # 'say', 'say_can', 'say_can_pay'
        self.model_name = model_name
        if self.decoding_score in ['say_can', 'say_can_pay']:
            print("\n==========Loading Can Model===========\n")
            self.can_model = CanModel.load_from_checkpoint(can_ckpt)
            self.can_model.eval().to(self.device)
            print("=============== Done =================")
        if self.decoding_score == 'say_can_pay':
            print("\n==========Loading Pay Model===========\n")
            self.pay_model = PayModel.load_from_checkpoint(pay_ckpt)
            self.pay_model.eval().to(self.device)
            print("=============== Done =================")

    def precondition_score(self, context, action):
        """returns log probability score"""
        can_score = self.can_model(context, action)
        return torch.log(can_score + 1e-3).item()

    def postcondition_score(self, context, action):
        """returns log probability score"""
        pay_score = self.pay_model(context, action)
        return torch.log(pay_score + 1e-3).item()

    def filter_beams(self, generated_sequences, generated_beam_scores,
                     action_beam_prompts, action_beam_scores,
                     prev_contexts_can, prev_contexts_pro,
                     goal, curr_action_step, task_completed_desc,
                     num_final_beams):
        done = []  # indicates whether the best plans have terminated in the 'done placing blocks in bowls' token
        updated_action_beam_prompts = []
        updated_action_beam_scores = []
        can_contexts_updated = []
        pro_contexts_updated = []
        updated_generated_sequences = []

        # TODO: considering batch_size = 1, update the code to adapt to bigger batch sizes
        # [num_final_beams, num_return_sequences]
        # print('\n')
        for ind, (gen_sequences, lm_scores) in enumerate(zip(generated_sequences, generated_beam_scores)):
            for gen_seq, lm_score in zip(gen_sequences, lm_scores):
                # new_context_pro = f'{prev_contexts_pro[ind]} [Step {curr_action_step}] {gen_seq}.'
                new_action_score = lm_score.item()

                if self.decoding_score == 'say_can':
                    with torch.no_grad():
                        pre_score = \
                            self.precondition_score(context=prev_contexts_can[ind],
                                                    action=f'{gen_seq}.')
                    # all log scores
                    # print(gen_seq)
                    # print(np.exp(new_action_score + pre_score),
                    #       np.exp(new_action_score), np.exp(pre_score))

                    new_action_score += pre_score

                elif self.decoding_score == 'say_can_pay':
                    with torch.no_grad():
                        pre_score, post_score = \
                            self.precondition_score(context=prev_contexts_can[ind],
                                                    action=f'{gen_seq}.'), \
                                self.postcondition_score(context=prev_contexts_pro[ind],
                                                         action=f'{gen_seq}.')
                    # all log scores
                    # print(gen_seq)
                    # print(np.exp(new_action_score + pre_score + post_score),
                    #       np.exp(new_action_score), np.exp(pre_score),
                    #       np.exp(post_score))

                    new_action_score += pre_score + post_score

                elif self.decoding_score == 'say':
                    pass

                else:
                    raise Exception("Not a valid decoding score; check decoding_score argument in config file")

                new_beam_score = action_beam_scores[ind] + new_action_score
                new_beam_prompt = f'{action_beam_prompts[ind]}{gen_seq}. [Step {curr_action_step + 1}] '
                new_context_can = f'{prev_contexts_can[ind]} [Step {curr_action_step}] {gen_seq}.'
                new_context_pro = f'{prev_contexts_pro[ind]} [Step {curr_action_step}] {gen_seq}.'
                updated_action_beam_scores.append(new_beam_score)
                updated_action_beam_prompts.append(new_beam_prompt)
                done.append(True) if task_completed_desc in gen_seq else done.append(False)
                can_contexts_updated.append(new_context_can)
                pro_contexts_updated.append(new_context_pro)
                updated_generated_sequences.append(gen_seq)

        # sort the corresponding beams in descending order of updated beam scores
        # print(updated_action_beam_scores)
        action_beam_scores, action_beam_prompts, done_updated, \
            can_contexts_updated, pro_contexts_updated, updated_generated_sequences = \
            zip(*sorted(zip(updated_action_beam_scores, updated_action_beam_prompts,
                            done, can_contexts_updated, pro_contexts_updated, updated_generated_sequences),
                        key=lambda x: x[0],
                        reverse=True)[:num_final_beams])  # selecting num_final_beams
        return list(action_beam_scores), list(action_beam_prompts), list(done_updated), \
            list(can_contexts_updated), list(pro_contexts_updated), list(updated_generated_sequences)


class LmPlannerBase(ABC):
    def __init__(self, env, model_name, decoding_type, decoding_score, oracle_flag, max_steps):
        self.env = env
        self.lm_model = None
        self.tokenizer = None
        self.custom_scorer = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_beams = 6  # 8, 6  # divided into num_beam_groups (2 sequences in each group)
        self.num_beam_groups = 3  # 4, 3  # using group_beam_search for diversity
        self.num_return_sequences = 6  # 8, 6  # returning one sequence from each group
        self.diversity_penalty = 2.  # higher the diversity penalization, lower the diversity
        self.max_new_tokens = 10
        self.max_steps = max_steps
        self.decoding_type = decoding_type
        self.decoding_score = decoding_score
        self.num_final_beams = None
        self.init_num_final_beams()
        self.oracle_flag = oracle_flag  # oracle agent is used to collect expert trajectories
        self.model_name = model_name
        self.task_completed_desc = None  # done action in the plan
        if not oracle_flag:
            self.__init_lm()

    def init_num_final_beams(self):
        self.num_final_beams = 3  # 4, 2  # only used for decoding_type = 'beam_action'

    def __init_lm(self):
        # vicuna (decoder-only): https://huggingface.co/eachadea/vicuna-13b-1.1
        # flan_t5 (encoder-decoder): https://huggingface.co/docs/transformers/model_doc/flan-t5
        # lm = 'flan_t5'  # ['vicuna', 'flan_t5']
        if self.model_name == 'vicuna':
            self.tokenizer = LlamaTokenizer.from_pretrained("eachadea/vicuna-13b-1.1",
                                                            padding_side='left')
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.lm_model = LlamaForCausalLM.from_pretrained("eachadea/vicuna-13b-1.1",
                                                             device_map="auto",
                                                             load_in_8bit=True)
            self.tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
        elif self.model_name == 'flan_t5':
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
            self.lm_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl",
                                                                       device_map="auto",
                                                                       load_in_8bit=True)

        self.lm_model.eval()
        # print(self.lm_model.hf_device_map)
        if self.decoding_type != 'greedy_token':
            self.custom_scorer = CustomScorer(decoding_score=self.decoding_score,
                                              model_name=self.model_name)

        # generation configuration (greedy, beam)
        self.greedy_generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id
        )
        self.beam_generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            num_beam_groups=self.num_beam_groups,
            num_return_sequences=self.num_return_sequences,
            diversity_penalty=self.diversity_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )

    @abstractmethod
    def initialize_episode(self, seed):
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    def env_step(self, action, **kwargs):
        """single environment step for online plan execution"""
        raise NotImplementedError("This is an abstract method.")

    def init_task_completed_desc(self, value):
        self.task_completed_desc = value

    def greedy_generate(self, prompt):
        """greedy generation using token probability"""
        tokenized_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

        with torch.no_grad():
            generated_ids = self.lm_model.generate(
                tokenized_inputs.input_ids.to(self.device),
                generation_config=self.greedy_generation_config
            )
        if self.model_name == 'vicuna':
            generated_ids = generated_ids[:, tokenized_inputs.input_ids.shape[-1]:]

        generated_sequence = \
            self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        return generated_sequence

    def beam_generate(self, prompt):
        """beam score-based generation using token probability"""

        tokenized_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

        with torch.no_grad():
            beam_search_output = self.lm_model.generate(
                tokenized_inputs.input_ids.to(self.device),
                generation_config=self.beam_generation_config
            )
        generated_ids, beam_scores = \
            beam_search_output.sequences, beam_search_output.sequences_scores
        # for decoder-only model, remove the starting prompt tokens, only use the generated tokens
        if self.model_name == 'vicuna':
            generated_ids = generated_ids[:, tokenized_inputs.input_ids.shape[-1]:]

        generated_sequences = \
            self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        return generated_sequences, beam_scores

    def plan_step(self, prompt, admissible_actions, prev_context_can, prev_context_pro, goal, curr_action_step=-1):
        """ single offline action step given prompt (current state and goal)"""
        # print(prompt)
        if self.decoding_type == 'greedy_token':
            generated_sequence = self.greedy_generate(prompt)
            # print(prompt)
            # print(f'generated_sequence: {generated_sequence[0]}')
            # return the generated action with the closest action available in the admissible set
            action = process_action(generated_sequence[0], admissible_actions)

        elif self.decoding_type == 'greedy_action':
            # common first step of beam search on tokens
            generated_sequences, generated_beam_scores = self.beam_generate(prompt)
            # print(f'generated_sequences: {generated_sequences}')
            # process actions to match admissible actions
            generated_sequences = \
                [process_action(seq, admissible_actions) for seq in generated_sequences]
            _, _, _, _, _, generated_sequences_updated = \
                self.custom_scorer.filter_beams(
                    [generated_sequences],
                    [generated_beam_scores],
                    action_beam_prompts=[prompt for _ in range(len(generated_sequences))],
                    action_beam_scores=[0 for _ in range(len(generated_sequences))],
                    prev_contexts_can=[prev_context_can], prev_contexts_pro=[prev_context_pro],
                    goal=goal, curr_action_step=curr_action_step,
                    task_completed_desc=self.task_completed_desc,
                    num_final_beams=self.num_final_beams)
            action = generated_sequences_updated[0]  # action with the highest score (already sorted in filter beams)
        else:
            raise Exception("Not a valid decoding type; check decoding_type argument in config file")
        return action

    def lm_planner(self, prompt, admissible_actions, context_can, context_pro, goal):
        """generates full plans via multiple calls to self.plan_step"""

        # single action at each step?
        # False if decoding type is greedy_token, beam_token, greedy_action
        beam_actions_flag = True if self.decoding_type == 'beam_action' else False
        raw_plan = []

        assert self.task_completed_desc is not None, "please initialize the task_completed_desc argument"

        if not beam_actions_flag:
            # generate the best action every time-step based on the decoding strategy
            while True:
                # print(prompt)
                curr_action_step = len(raw_plan) + 1
                curr_action = self.plan_step(prompt, admissible_actions,
                                             context_can, context_pro, goal,
                                             curr_action_step=curr_action_step)
                # context_can = [goal, initial state, seq of actions]
                # context_pro = [initial state, seq of actions]
                context_can = f'{context_can} [Step {curr_action_step}] {curr_action}.'
                context_pro = f'{context_pro} [Step {curr_action_step}] {curr_action}.'
                raw_plan.append(f'[Step {curr_action_step}] {curr_action}\n')
                if self.task_completed_desc in curr_action:
                    break
                if len(raw_plan) > self.max_steps:
                    raise ExceededMaxSteps
                prompt = f'{prompt}{curr_action}. [Step {curr_action_step + 1}] '
            plan = ''.join(raw_plan)
            print(f'\n{plan}\n')
            return plan

        else:  # generate the full plan with action-level beam search
            def postprocess_plans_from_beams(beam_prompt):
                """retrieve full plan from action beam prompts"""
                if 'Initial State' in beam_prompt:
                    beam_prompt = beam_prompt.split('[Initial State] ')[-1].strip()
                else:
                    beam_prompt = beam_prompt.split('[Goal] ')[-1].strip()  # virtualhome
                beam_prompt = beam_prompt.split('. ')
                retrieved_plan = []
                for line in beam_prompt:
                    step_match = re.search(r'\[Step \d+\] (.*)', line)
                    if step_match:
                        plan_step = step_match.group(1)
                        plan_step = re.sub(r'[^\w\s]', '', plan_step)
                        retrieved_plan.append(plan_step)
                return retrieved_plan

            all_beams = []
            self.init_num_final_beams()
            generated_sequences, generated_beam_scores = self.beam_generate(prompt)
            # postprocess
            generated_sequences = \
                [process_action(seq, admissible_actions) for seq in generated_sequences]
            # filter out repeated steps
            generated_sequences, generated_beam_scores = \
                self.select_unique_steps(generated_sequences, generated_beam_scores)
            # stores the best plans (sequences of actions) in NL
            action_beam_prompts = [prompt for _ in range(1)]
            # stores the scores corresponding to best plans
            action_beam_scores = [0 for _ in range(1)]
            contexts_can = [context_can for _ in range(1)]
            contexts_pro = [context_pro for _ in range(1)]

            action_steps = 0
            while True:  # at least one action sequence has not terminated
                # update the beam scores of actions based on pre+post conditions of [num_final_beams x num_beams]
                # also return the overall scores, sequences of 'num_final_beams' sequences
                action_steps += 1
                if action_steps == 1:
                    action_beam_scores, action_beam_prompts, done, contexts_can, contexts_pro, _ = \
                        self.custom_scorer.filter_beams(
                            np.reshape(generated_sequences, (1, len(generated_sequences))).tolist(),
                            generated_beam_scores.view(1,  len(generated_sequences)),
                            action_beam_prompts,
                            action_beam_scores,
                            contexts_can, contexts_pro,
                            goal, action_steps,
                            task_completed_desc=self.task_completed_desc,
                            num_final_beams=self.num_final_beams)
                else:
                    action_beam_scores, action_beam_prompts, done, contexts_can, contexts_pro, _ = \
                        self.custom_scorer.filter_beams(
                            generated_sequences,
                            generated_beam_scores,
                            action_beam_prompts,
                            action_beam_scores,
                            contexts_can, contexts_pro,
                            goal, action_steps,
                            task_completed_desc=self.task_completed_desc,
                            num_final_beams=self.num_final_beams)

                # store the beam which has terminated in a DONE token
                # beam search with remaining beams out of self.num_final_beams
                for ind, done_flag in enumerate(done):
                    if done_flag:
                        action_beam_prompt = action_beam_prompts[ind]
                        action_beam_score = action_beam_scores[ind]
                        # retrieve the generated plan from the prompt + generated
                        gen_plan = postprocess_plans_from_beams(action_beam_prompt)
                        gen_plan_score = action_beam_score / action_steps
                        all_beams.append((gen_plan, gen_plan_score))
                        self.num_final_beams -= 1
                # remove all the 'done' sequences and proceed with the incomplete action sequences
                try:
                    action_beam_prompts, action_beam_scores, done, contexts_can, contexts_pro = \
                        list(zip(*[x for ind, x in
                                   enumerate(zip(action_beam_prompts, action_beam_scores, done, contexts_can, contexts_pro))
                                   if not done[ind]]))
                except ValueError:  # if all are sequences result in True
                    pass

                # indicates whether the best plans have terminated in the 'done placing blocks in bowls' token
                if self.num_final_beams == 0:
                    break

                if action_steps > self.max_steps:
                    # return best terminated beam
                    if len(all_beams) > 0:
                        # selecting the plan with the highest log-score
                        raw_plan = sorted(all_beams, key=lambda x: x[1], reverse=True)[0][0]
                        plan = ''
                        step_ind = 1
                        for step in raw_plan:
                            if step != '':
                                plan += f'[Step {step_ind}] {step}\n'
                                step_ind += 1
                        print(f'\n{plan}\n')
                        return plan
                    #  raise Exception if there are no terminated beams
                    raise ExceededMaxSteps

                # beam_prompts are processed as a batch of size 'num_final_beams',
                # output of size [num_final_beams x num_return_sequences]
                new_generated_sequences, new_generated_beam_scores = self.beam_generate(action_beam_prompts)
                # postprocess
                new_generated_sequences = \
                    [process_action(seq, admissible_actions) for seq in new_generated_sequences]
                new_generated_sequences = \
                    np.reshape(new_generated_sequences, (self.num_final_beams, self.num_return_sequences)).tolist()
                new_generated_beam_scores = \
                    new_generated_beam_scores.view(self.num_final_beams, self.num_return_sequences)
                generated_sequences = []
                generated_beam_scores = []
                for gen_id, gen_seqs in enumerate(new_generated_sequences):
                    gen_beam_scores = new_generated_beam_scores[gen_id]
                    # filter out repeated steps
                    unique_gen_seqs, unique_gen_beam_scores = \
                        self.select_unique_steps(gen_seqs, gen_beam_scores)
                    generated_sequences.append(unique_gen_seqs)
                    generated_beam_scores.append(unique_gen_beam_scores)

            # selecting the plan with the highest log-score
            raw_plan = sorted(all_beams, key=lambda x: x[1], reverse=True)[0][0]
            plan = ''
            step_ind = 1
            for step in raw_plan:
                if step != '':
                    plan += f'[Step {step_ind}] {step}\n'
                    step_ind += 1
            print(f'\n{plan}\n')
            return plan

    def select_unique_steps(self, generated_sequences, beam_scores):
        # filtering out repeated steps
        # i.e. repeated action steps are present in the beam output
        gen_score_dict = {}
        for gen_seq, score in zip(generated_sequences, beam_scores):
            if gen_seq in gen_score_dict:
                gen_score_dict[gen_seq] = max(gen_score_dict[gen_seq], score)
            else:
                gen_score_dict[gen_seq] = score
        generated_sequences_processed = list(gen_score_dict.keys())
        beam_scores_processed = torch.tensor(list(gen_score_dict.values()),
                                             device=beam_scores.device).view(-1)
        # extending the lists to num_final_beams size
        # via step repetition to avoid errors in the beam_action
        while 0 < len(generated_sequences_processed) < self.num_final_beams:
            generated_sequences_processed.append(generated_sequences_processed[-1])
            beam_scores_processed = torch.cat((beam_scores_processed, beam_scores_processed[-1].view(-1)))
        return generated_sequences_processed, beam_scores_processed
