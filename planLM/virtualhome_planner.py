import os
from lm_planner_base import LmPlannerBase
from data_utils import ExceededMaxSteps, \
    process_saved_plans_virtualhome, latest_file_from_dir
import time
import random
import pickle
import numpy as np


class LmAgent(LmPlannerBase):
    def __init__(self, env, domain, cfg,
                 model_name, decoding_type, decoding_score,
                 oracle_flag, few_shot_prompt, max_steps, save_dir=None):
        self.domain = domain
        self.cfg = cfg
        self.num_episodes = cfg['n']
        self.few_shot_prompt = few_shot_prompt
        self.record = cfg['record']['save_video']
        self.save_data_flag = cfg['save_data']
        self.task_completed_desc = 'done task'
        self.oracle_plans = None
        self.admissible_actions = None
        self.__init_logs(save_dir)
        super().__init__(env, model_name, decoding_type, decoding_score, oracle_flag, max_steps)
        self.task_completed_desc = 'done task'

    def __init_logs(self, save_dir):
        self.plan_log_file = open(os.path.join(save_dir, time.strftime("%Y-%m-%d %H:%M")), 'w')
        # load oracle plans for that mode
        # only test-optimal is required for plan length comparison
        oracle_plan_path = \
            f'{os.environ["VIRTUALHOME"]}/data/oracle-plans/{self.domain.mode}'
        oracle_traj_file = latest_file_from_dir(oracle_plan_path)
        self.oracle_plans = \
            process_saved_plans_virtualhome(filepath=oracle_traj_file,
                                            task_completed_desc=self.task_completed_desc)
        admissible_action_path = os.path.join(os.environ['VIRTUALHOME'], 'admissible_actions')
        self.admissible_actions = pickle.load(open(admissible_action_path, 'rb'))

    def initialize_episode(self, ep):
        # set seed for each episode
        np.random.seed(ep)
        random.seed(ep)
        # try:
        saved_plan = self.oracle_plans[ep - 1]
        # except IndexError:
        #     print(f'Index Error while loading generated plan in {ep}')
        # TODO: change initial state
        initial_state = 'you are in a house.'
        info = {'lang_goal': saved_plan.goal, 'initial_state': initial_state, 'graph_id': saved_plan.graph_id,
                'task_completed_desc': self.task_completed_desc, 'admissible_actions': self.admissible_actions}
        super().init_task_completed_desc(self.task_completed_desc)
        return info

    def env_step(self, actions, **kwargs):
        """ one action step using LM """

        def parse_action(lang_action):
            """ parse action to retrieve action and arguments"""
            split_action = lang_action.split()
            action_verb = split_action[0]
            arguments = '.'
            if len(split_action) > 1:
                arguments = ' <' + '> (1) <'.join(split_action[1:]) + '> (1)'
            return action_verb, arguments

        script = []
        for action in actions:
            if action == "ExceededMaxSteps":
                return False
            env_action, env_arguments = parse_action(lang_action=action)
            act = f'<char0> [{env_action.upper()}]{env_arguments}'
            script.append(act)

        if not self.record:
            done, message_exec = self.env.render_script(script=script,
                                                        processing_time_limit=60,
                                                        find_solution=True,
                                                        recording=False,
                                                        skip_animation=True)
        else:
            # TODO: include recording
            done, message_exec = self.env.render_script(script=script,
                                                        processing_time_limit=60,
                                                        find_solution=True,
                                                        recording=False,
                                                        skip_animation=True)
        if not done:
            print(message_exec)
        return done

    def oracle_planner(self, obs, info):
        """run oracle agent to generate expert plan
        runs oracle agent from task.py"""
        pass

    def generate_plan_episode(self, ep, few_shot_prompts):
        """generate plan for each episode using oracle / LM"""
        # Initialize episode with seed = initial_seed + ep
        info = self.initialize_episode(ep=ep)

        lang_goal = info['lang_goal']
        # initial_state = info['initial_state']
        admissible_actions_list = list(info['admissible_actions'])
        admissible_actions_list.append(info['task_completed_desc'])
        plan_init = f'[Scene] {info["graph_id"]}\n' \
                    f'[Goal] {lang_goal}\n'
        # f'[Initial State] {initial_state}\n'
        # admissible_actions = ', '.join(admissible_actions_list)

        if self.oracle_flag:  # collect dataset
            pass
        else:  # use language model
            lm_plan = plan_init
            prompt = \
                f'{few_shot_prompts}' \
                f'[Goal] {lang_goal} ' \
                f'[Step 1] '
            print(f'[Goal] {lang_goal} \n')

            context_can = f'[Goal] {lang_goal}'
            context_pro = f'[Goal] {lang_goal}'
            try:
                plan = self.lm_planner(prompt, admissible_actions_list,
                                       context_can, context_pro, f'[Goal] {lang_goal}')
                lm_plan += plan
                self.plan_log_file.write(lm_plan)
                self.plan_log_file.write('\n')
            except ExceededMaxSteps:
                self.plan_log_file.write(lm_plan)
                self.plan_log_file.write('ExceededMaxSteps\n\n')
                return
        # clear the internal buffer after each episode
        self.plan_log_file.flush()

    def generate_plans(self):
        """ generate plans using oracle / LM. Run multiple episodes"""
        # run oracle for data generation
        if self.oracle_flag:
            pass
        # run LM (multi-GPU)
        else:
            print(f"\n========================= LM generation - {self.domain.mode} ===========================\n "
                  f"Task: VirtualHome | LM: {self.model_name} | Scoring: {self.decoding_type}-{self.decoding_score}\n"
                  "===========================================================================\n")
            few_shot_prompts = self.few_shot_prompt.prompt()
            for ep in range(1, self.num_episodes + 1):
                print(ep)
                self.generate_plan_episode(ep, few_shot_prompts)

        self.plan_log_file.close()

    def test_plans(self, genplan_file_path):
        print(f"\n=========================== LM test - {self.domain.mode} =============================\n "
              f"Task: VirtualHome | LM: {self.model_name} | Scoring: {self.decoding_type}-{self.decoding_score}\n"
              "===========================================================================\n")

        self.initialize_episode(ep=0)
        # rewards = num success plans / total episodes
        rewards = 0.
        optimal_rewards = 0.

        # load saved plans
        # genplan_file_path = f'{os.environ["VIRTUALHOME"]}/data/' \
        #                     f'lm-plans/{self.domain.mode}/{self.model_name}/{self.decoding_type}' \
        #                     f'-{self.decoding_score}/2023-06-28 12:36'
        # genplan_file_path = self.plan_log_file.name
        gen_plans = \
            process_saved_plans_virtualhome(filepath=genplan_file_path,
                                            task_completed_desc=self.task_completed_desc)

        # for each episode
        for ep in range(1, self.num_episodes + 1):
            try:
                plan = gen_plans[ep - 1]
            except IndexError:
                print(f'Index Error while loading generated plan in {ep}')
            info = self.initialize_episode(ep=ep)
            self.env.reset(info['graph_id'] - 1)
            self.env.add_character('chars/Male1')

            # placing additional checks for safety\
            assert plan.goal == info['lang_goal']

            # execute each action of the plan:
            done = self.env_step(plan.actions)
            if done:
                # for test-optimal, compare plan lengths wrt to the oracle plan
                # TODO: change this (something to incorporate the open-endedness of VirtualHome)
                if len(plan.actions) <= len(self.oracle_plans[ep].actions):
                    optimal_rewards += 1
                rewards += 1

        print(rewards / self.num_episodes, optimal_rewards / self.num_episodes)
        return rewards / self.num_episodes, optimal_rewards / self.num_episodes
