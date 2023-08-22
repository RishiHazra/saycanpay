import os
from lm_planner_base import LmPlannerBase
from data_utils import ExceededMaxSteps, \
    process_saved_plans, latest_file_from_dir
import re
import time
import random
import numpy as np
import multiprocessing as mp


class LmAgent(LmPlannerBase):
    def __init__(self, agent, env, domain, cfg,
                 model_name, decoding_type, decoding_score,
                 oracle_flag, few_shot_prompt,
                 initial_seed, max_steps, save_dir=None):
        self.agent = agent
        self.domain = domain
        self.cfg = cfg
        self.num_episodes = cfg['n']
        self.few_shot_prompt = few_shot_prompt
        self.initial_seed = initial_seed
        self.seed = initial_seed
        self.record = cfg['record']['save_video']
        self.save_data_flag = cfg['save_data']
        self.__init_logs(save_dir)
        super().__init__(env, model_name, decoding_type, decoding_score, oracle_flag, max_steps)

    def __init_logs(self, save_dir):
        self.plan_log_file = open(os.path.join(save_dir, time.strftime("%Y-%m-%d %H:%M")), 'w')

    def initialize_episode(self, seed):
        # set seed for each episode
        self.seed = self.initial_seed + seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.env.set_task(self.domain)
        obs = self.env.reset()
        info = self.env.info
        self.env.task.max_steps = self.max_steps
        reward = 0
        super().init_task_completed_desc(self.env.task.task_completed_desc)
        return obs, info, reward

    def env_step(self, action, **kwargs):
        """ one action step using LM """
        obs, info = kwargs.pop('obs'), kwargs.pop('info')

        def parse_action(lang_action, task):
            """ parse action to retrieve pickup object and place object"""
            lang_action = re.sub(r'[^\w\s]', '', lang_action)  # remove all strings
            if task == 'put-block-in-bowl':
                target_pattern = r'([a-zA-Z]+ block \d+)'
                recep_pattern = r'([a-zA-Z]+ bowl \d+)'
            elif task == 'towers-of-hanoi-seq':
                target_pattern = r'([a-zA-Z]+ disk)'
                recep_pattern = r"(rod 1|rod 2|rod 3)"
            else:
                raise NotImplementedError

            target_match = re.search(target_pattern, lang_action)
            recep_match = re.search(recep_pattern, lang_action)  # receptacle

            if target_match and recep_match:
                target = target_match.group(1)
                recep = recep_match.group(1)
                return target, recep
            else:
                return None, None

        # pick_obj, place_obj = parse_action(action)
        act = self.agent.act(parse_action(lang_action=action,
                                          task=self.cfg['task']),
                             obs, info)  # pick_pose, place_pose
        z = self.env.step(act)
        try:
            obs, reward, done, info = z
        except ValueError:
            obs, reward, done, info, action_ret_str = z
            print(action_ret_str)
        return obs, reward, done, info

    def oracle_planner(self, obs, info):
        """run oracle agent to generate expert plan
        runs oracle agent from task.py"""
        plan = ''
        for i in range(self.env.task.max_steps):  # fix a maximum number of steps?
            action = self.agent.act(obs, info)
            obs, reward, done, info = self.env.step(action)
            plan += f'[Step {i + 1}] {action["lang_action"]}\n' \
                    f'[Observation] {info["lang_observation"]}\n'
            if done:
                plan += f'[Step {i + 2}] {self.task_completed_desc}\n\n'
                return plan
        raise ExceededMaxSteps

    def generate_plan_episode(self, ep, few_shot_prompts, queue=None):
        """generate plan for each episode using oracle / LM"""
        try:
            # Initialize episode with seed = initial_seed + ep
            obs, info, reward = self.initialize_episode(seed=ep)

            if self.record:
                self.env.start_rec(f'{ep + 1:06d}')

            lang_goal = info['lang_goal']
            initial_state = self.env.task.lang_initial_state
            admissible_actions_list = self.env.task.admissible_actions
            admissible_actions_list.append(self.task_completed_desc)
            plan_init = f'[Seed] {self.seed}\n' \
                        f'[Goal] {lang_goal}\n' \
                        f'[Initial State] {initial_state}\n'
            # admissible_actions = ', '.join(admissible_actions_list)

            if self.oracle_flag:  # collect dataset
                oracle_plan = plan_init
                try:
                    plan = self.oracle_planner(obs, info)
                    oracle_plan += plan
                    if self.save_data_flag:
                        self.plan_log_file.write(oracle_plan)
                except ExceededMaxSteps:
                    self.plan_log_file.write('ExceededMaxSteps')
                    return
            else:  # use language model
                lm_plan = plan_init
                prompt = \
                    f'{few_shot_prompts}' \
                    f'[Goal] {lang_goal} ' \
                    f'[Initial State] {initial_state} ' \
                    f'[Step 1] '
                print(f'[Goal] {lang_goal} \n[Initial State] {initial_state} \n')

                context_can = f'[Goal] {lang_goal} [Initial State] {initial_state}'
                context_pro = f'[Initial State] {initial_state} [Goal] {lang_goal}'
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

        except Exception as e:
            error_message = f"Error in episode {ep} with seed {self.seed}: {str(e)}"
            queue.put(error_message)

    def generate_plans(self):
        """ generate plans using oracle / LM. Run multiple episodes"""
        # run oracle for data generation
        if self.oracle_flag:
            few_shot_prompts = ''
            # multiprocessing for data generation from oracle
            print(f"\n================= Oracle - {self.domain.mode} ===================\n "
                  f"Saving trajectories for Task: {self.domain.__class__.__name__}\n"
                  "====================================================\n")
            if self.cfg['parallel']:
                error_queue = mp.Queue()
                num_workers = mp.cpu_count()  # use all available cpu cores

                # process episodes parallely based on the number of cpu cores
                for i in range(1, self.num_episodes + 1, num_workers):
                    procs = [mp.Process(target=self.generate_plan_episode,
                                        args=(ep, few_shot_prompts, error_queue))
                             for ep in range(i, min(i + num_workers, self.num_episodes))]

                    try:
                        for proc_id, proc in enumerate(procs):
                            proc.start()
                            time.sleep(0.1)
                    finally:
                        for proc in procs:
                            proc.join()

                # print errors in the queue
                while not error_queue.empty():
                    print(error_queue.get())

            # sequentially generate data (very slow)
            else:
                for ep in range(1, self.num_episodes + 1):
                    self.generate_plan_episode(ep, few_shot_prompts)
        # run LM (not multiprocessing but multi-GPU)
        else:
            print(f"\n========================= LM generation - {self.domain.mode} ===========================\n "
                  f"Task: {self.domain.__class__.__name__} | LM: {self.model_name} | Scoring: {self.decoding_type}-{self.decoding_score}\n"
                  "===========================================================================\n")
            few_shot_prompts = self.few_shot_prompt.prompt()
            for ep in range(1, self.num_episodes + 1):
                print(ep)
                self.generate_plan_episode(ep, few_shot_prompts)

        self.plan_log_file.close()

    def test_plans(self, initial_seed):
        print(f"\n=========================== LM test - {self.domain.mode} =============================\n "
              f"Task: {self.domain.__class__.__name__} | LM: {self.model_name} | Scoring: {self.decoding_type}-{self.decoding_score}\n"
              "===========================================================================\n")

        self.initialize_episode(seed=0)
        # rewards = num success plans / total episodes
        rewards = 0.
        didnot_execute = 0.  # keeps track of plans that didn't execute in test sets

        # load saved plans
        # genplan_file_path = f'{os.environ["RAVENS_ROOT"]}/{self.cfg["task"]}/data/' \
        #                     f'lm-plans/{self.domain.mode}/{self.model_name}/{self.decoding_type}' \
        #                     f'-{self.decoding_score}/2023-06-28 12:36'
        genplan_file_path = self.plan_log_file.name
        gen_plans = \
            process_saved_plans(filepath=genplan_file_path,
                                task_completed_desc=self.task_completed_desc)

        # load oracle plans for that mode
        # only test-optimal is required for plan length comparison
        if self.domain.mode == 'test-optimal':
            oracle_plan_path = \
                f'{os.environ["RAVENS_ROOT"]}/{self.cfg["task"]}/data/oracle-plans/{self.domain.mode}'
            oracle_traj_file = latest_file_from_dir(oracle_plan_path)
            oracle_plans = \
                process_saved_plans(filepath=oracle_traj_file,
                                    task_completed_desc=self.task_completed_desc)

        # setting initial seed
        self.initial_seed = initial_seed
        # for each episode
        for ep in range(1, self.num_episodes + 1):
            try:
                plan = gen_plans[ep - 1]
            except IndexError:
                print(f'Index Error while loading generated plan in {ep}')
            obs, info, _ = self.initialize_episode(seed=ep)

            # placing additional checks for safety\
            assert plan.seed == self.seed
            assert plan.goal == info['lang_goal']

            # assert plan.initial_state == self.env.task.lang_initial_state

            if self.record:
                self.env.start_rec(f'{ep + 1:06d}')

            # execute each action of the plan:
            steps = 0
            for action in plan.actions:
                # if planner says completed even though the task is not complete
                if action == self.task_completed_desc:
                    break

                if action == "ExceededMaxSteps":
                    didnot_execute += 1
                    break
                # execute each step of the plan in the environment
                try:
                    obs, _, done, info = self.env_step(action, obs=obs, info=info)
                    steps += 1
                    if done:
                        # for test-optimal, compare plan lengths wrt to the oracle plan
                        if self.domain.mode == 'test-optimal':
                            try:
                                if len(plan.actions) <= len(oracle_plans[ep].actions):
                                    rewards += 1
                                else:
                                    break
                            # TODO: fix this problem later
                            except IndexError:  # in oracle plans
                                print('IndexError')
                                rewards += 1
                        else:
                            rewards += 1
                        break  # skip to the next plan
                except ExceededMaxSteps:
                    didnot_execute += 1
                    break  # skip to the next plan

        print(rewards / self.num_episodes, didnot_execute)
        return rewards / self.num_episodes, didnot_execute
