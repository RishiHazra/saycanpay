import os
import re
import pickle
import random
import numpy as np
from data_utils import load_from_disk
from torch.utils.data import Dataset, random_split, DataLoader
import lightning.pytorch as pl


class Sample:
    """Dataset Sample Class"""

    def __init__(self, graph_id, goal, actions, description):
        self.graph_id = graph_id
        self.goal = goal
        self.actions = actions
        self.description = description


class PlanningDataset(Dataset):
    def __init__(self, file_path, save_path, completed_action):
        self.completed_action = completed_action  # action denoting end of plan (different for each task)
        self.samples = self.process(file_path, save_path)  # preprocess data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]  # each_tuple: ?

    def process(self, file_path, save_path):
        """preprocess dataset to save data samples"""
        samples = []
        if os.path.isfile(save_path):
            print(f"{save_path} already exists. Loading from save_path.")
            samples = load_from_disk(save_path)
            return samples

        print('\n=============== Processing oracle trajectories. =================\n')
        with open(file_path, 'r') as infile:
            lines = infile.readlines()
            i = 0
            while i < len(lines):
                if "[Scene]" in lines[i]:
                    graph_id = int(lines[i].split("] ")[1].strip())
                    goal = lines[i + 1].split("] ")[1].strip()
                    description = lines[i + 2].split("] ")[1].strip()
                    actions = []
                    i += 3
                    while i < len(lines) and self.completed_action not in lines[i]:
                        action_step_match = re.search(r'\[Step \d+] (.*)', lines[i])
                        if action_step_match:
                            actions.append(action_step_match.group(1))
                        i += 1
                    # ensure that there are at least 2 actions in each sample
                    # including the done action
                    assert len(actions) >= 1, "at least 2 actions in each sample required."
                    actions.append(self.completed_action)
                    samples.append(Sample(graph_id, goal, actions, description))
                i += 1
        print(f"\n Processed {len(samples)} samples. Saving in {save_path} \n")
        # dump the samples
        with open(save_path, 'wb') as infile:
            pickle.dump(samples, infile)
        # return samples


class CanSample:
    """Can Model sample class"""

    def __init__(self, context, pos_action, neg_action, neg_sample_action):
        self.context = context
        self.pos_action = pos_action  # action immediately following the context
        self.neg_action = neg_action  # negative action from the same trajectory
        self.neg_sample_action = neg_sample_action  # action from a different trajectory


class CanDataset(Dataset):
    def __init__(self, file_path, save_path):
        self.all_samples = load_from_disk(file_path)  # load preprocessed data
        print(f'\nLoaded {len(self.all_samples)} samples\n')
        self.samples = self.process(save_path)  # can samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        # returns tokenized inputs
        # sample = self.samples[item]
        # return {key: tokenizer(value, return_tensors="pt", padding=True) for key, value in sample.items()}
        return self.samples[item]  # each_tuple: ?

    def process(self, can_save_path):
        """postprocess data samples for CAN model training"""
        can_samples = []
        if os.path.isfile(can_save_path):
            print(f"\n\n {can_save_path} already exists. Loading from save_path.\n\n")
            can_samples = load_from_disk(can_save_path)
            return can_samples

        for sample in self.all_samples:
            context = ['[Goal]', sample.goal]
            for i in range(len(sample.actions)):
                pos_action = sample.actions[i]

                # negative action from the same sample
                # i.e. action executed in a different context
                neg_actions = []
                for ind, action in enumerate(sample.actions):
                    # choose a negative sample that is not the same as positive
                    if action != pos_action:
                        neg_actions.append(action)
                # sample a neg_action
                neg_action = np.random.choice(neg_actions)
                assert neg_action is not None, \
                    "Negative action should not be None, double check if # actions in the sample are at least 2"

                # TODO: should we set a seed here?
                # action from a different sample
                # i.e. completely different context (initial state)
                pos_action = f'{pos_action}'
                neg_action = f'{neg_action}'

                num_neg_samples = 2
                for _ in range(num_neg_samples):
                    neg_sample = random.choice([s for s in self.all_samples if s != sample])
                    neg_sample_action = random.choice(neg_sample.actions)
                    discard_samples = [sample]

                    while neg_sample_action == pos_action or neg_sample.goal == sample.goal:
                        discard_samples.append(neg_sample)
                        neg_sample = random.choice([s for s in self.all_samples if s not in discard_samples])
                        neg_sample_action = random.choice(neg_sample.actions)

                    # only for hanoi
                    # if random.random() > 0.7 and pos_action.split()[-1] != '1.':
                    #     neg_action = ' '.join(pos_action.split()[:-1] + ['1.'])
                    neg_sample_action = f'{neg_sample_action}'
                    can_samples.append(
                        vars(CanSample(' '.join(context), pos_action, neg_action, neg_sample_action))
                    )
                # append context until the last action
                #    context.extend([sample.actions[i], sample.observations[i]])
                if i < len(sample.actions) - 1:
                    context.append(f'[Step {i + 1}] {pos_action}')

        # dump the samples
        print(f"\n\nProcessed {len(can_samples)} Can samples. Saving in {can_save_path}\n\n")
        pickle.dump(can_samples, open(can_save_path, 'wb'))


class PaySample:
    """Pay Model Sample class"""

    def __init__(self, context, action, dist):
        self.context = context
        self.action = action
        self.dist = dist


class PayDataset(Dataset):
    def __init__(self, file_path, save_path):
        self.all_samples = load_from_disk(file_path)  # load preprocessed data
        self.samples = self.process(save_path)  # pay samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        # sample = self.samples[item]
        # return {key: tokenizer(value, return_tensors="pt", padding=True) for key, value in sample.items()}
        return self.samples[item]  # each_tuple: ?

    def process(self, pay_save_path):
        """post-process data sample for PAY model training"""
        pay_samples = []
        if os.path.isfile(pay_save_path):
            print(f"\n{pay_save_path} already exists. Loading from save_path.\n")
            pay_samples = load_from_disk(pay_save_path)
            return pay_samples

        for sample in self.all_samples:
            context = ['[Goal]', sample.goal]
            for i in range(len(sample.actions)):
                # normalized distance
                # only for babyai
                # dist = np.float32(1 / (len(sample.actions) - i))
                # dist = np.float32(1. - (len(sample.actions) - i - 1) / len(sample.actions))
                gamma = 1.5  # hanoi (gamma=3), babyai (gamma=1.5)
                final_reward = 1
                dist = np.float32(1 / (gamma ** (len(sample.actions) - i)) * final_reward)
                pay_samples.append(
                    vars(PaySample(context=' '.join(context),
                                        action=f'[Step {i + 1}] {sample.actions[i]}.',
                                        dist=dist))
                )
                neg_actions = []
                for ind, action in enumerate(sample.actions):
                    if action != sample.actions[i]:
                        neg_actions.append(action)

                num_neg_actions = 1
                for _ in range(num_neg_actions):
                    neg_action = np.random.choice(neg_actions)
                    pay_samples.append(
                        vars(PaySample(context=' '.join(context),
                                            action=f'[Step {i + 1}] {neg_action}.',
                                            dist=np.float32(0)))
                    )
                    neg_actions.remove(neg_action)

                context.append(f'[Step {i + 1}] {sample.actions[i]}.')

        # dump the samples
        print(f"\nProcessed {len(pay_samples)} Pay samples. Saving in {pay_save_path}\n")
        pickle.dump(pay_samples, open(pay_save_path, 'wb'))


class DataModule(pl.LightningDataModule):
    def __init__(self, model, file_path, preprocess_path, save_path, batch_size, train_val_split, completed_action):
        super().__init__()
        self.val_set = None
        self.train_set = None
        self.model = model
        self.file_path = file_path
        self.preprocess_path = preprocess_path
        self.save_path = save_path
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.completed_action = completed_action  # action denoting end of plan (different for each task)

    def prepare_data(self) -> None:
        # preprocess (Lightning ensures this is called with single CPU)
        PlanningDataset(self.file_path,
                        self.preprocess_path,
                        self.completed_action)
        # postprocess
        if self.model == 'can':
            CanDataset(self.preprocess_path, self.save_path)
        elif self.model == 'pay':
            PayDataset(self.preprocess_path, self.save_path)

    def setup(self, stage: str) -> None:
        dataset = load_from_disk(self.save_path)
        train_size = int(self.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        self.train_set, self.val_set = \
            random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          num_workers=10)
