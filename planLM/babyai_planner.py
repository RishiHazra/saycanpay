import gymnasium as gym
from babyai_utils import LanguageObsWrapper
import numpy as np
from dotmap import DotMap
import os
import regex as re
from lm_planner_base import LmPlannerBase
from data_utils import ExceededMaxSteps, \
    process_saved_plans, latest_file_from_dir
import time
import random
import multiprocessing as mp


# from minigrid.envs.babyai.unlock import CustomActions


class LmAgent(LmPlannerBase):
    def __init__(self, env, domain, cfg,
                 model_name, strategy, decoding_score,
                 oracle_flag, few_shot_prompt,
                 initial_seed, max_steps, save_dir=None, disp=False):
        self.domain = domain
        self.cfg = cfg
        self.num_episodes = cfg['n']
        self.max_steps = max_steps
        self.few_shot_prompt = few_shot_prompt
        self.initial_seed = initial_seed
        self.seed = initial_seed
        self.record = cfg['record']['save_video']
        self.save_data_flag = cfg['save_data']
        self.display = disp
        self.__init_logs(save_dir)
        super().__init__(env, model_name, strategy, decoding_score, oracle_flag, max_steps)

    def __init_logs(self, save_dir):
        self.plan_log_file = open(os.path.join(save_dir, time.strftime("%Y-%m-%d %H:%M")), 'w')

    def initialize_episode(self, seed):
        # set seed for each episode
        self.seed = self.initial_seed + seed
        np.random.seed(self.seed)
        random.seed(seed)
        # select task
        selected_task = np.random.choice(self.domain.tasks)
        if self.display:
            self.env = gym.make(f"BabyAI-{selected_task}-v0", render_mode="human")
        else:
            self.env = gym.make(f"BabyAI-{selected_task}-v0")
        self.env = LanguageObsWrapper(self.env)
        state, info = self.env.reset(seed=self.seed)
        self.env.max_steps = self.max_steps
        # self.env.task.max_steps = self.max_steps
        reward = 0
        super().init_task_completed_desc("done picking up")
        return state, info, reward

    def env_step(self, action, **kwargs):
        """ one action step using LM """
        state, info = kwargs.pop('state'), kwargs.pop('info')

        def parse_action(lang_action, state):
            """ parse action to retrieve babyai env actions (CustomRoomGridLevel)
            return a tuple (action_index, target_object_pos)"""
            # lang_action = lang_action.replace("an ", "the ")
            # parsed = lang_action.replace("the ", "").split()
            lang_action = re.sub(r'[^\w\s]', '', lang_action)  # remove all strings
            parsed = lang_action.split()

            if 'pick' in parsed:
                action_step = 'pickup'
                target_obj = DotMap(color=parsed[-2], type=parsed[-1])
            elif 'toggle' in parsed:
                action_step = 'toggle'
                target_obj = DotMap(color=parsed[-2], type=parsed[-1])
            elif 'drop' in parsed:
                action_step = 'drop'
                target_obj = None

            action_index = [ind for ind, a in enumerate(self.env.actions)
                            if a.name == action_step][0]
            if action_step in ['pickup', 'toggle']:
                return action_index, target_obj
            else:
                return action_index, None

        act = parse_action(action, state)  # (action_index, target_object_pos)
        state, reward, terminated, truncated, info = self.env.step(act)
        done = terminated or truncated
        return state, reward, done, info

    def oracle_planner(self, state):
        """run oracle to generate expert plan
        runs oracle() from CustomRoomGrid in minigrid/envs/babyai/unlock.py"""
        plan = ''
        oracle_actions = self.env.oracle(grid=state["image"])
        if len(oracle_actions) > self.env.max_steps:
            raise ExceededMaxSteps
        # fix a maximum number of steps?
        for i, (action, lang_action) in \
                enumerate(zip(oracle_actions["actions"], oracle_actions["lang_actions"])):
            curr_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            plan += f'[Step {i + 1}] {lang_action}\n' \
                    f'[Observation] {info["state_change"]}\n'
            # prev_state = curr_state.copy()
            if done:
                plan += f'[Step {i + 2}] {self.task_completed_desc}\n\n'
                return plan
        raise ExceededMaxSteps

    def generate_plan_episode(self, ep, few_shot_prompts, queue=None):
        """generate plan for each episode using oracle / LM"""
        try:
            # Initialize episode with seed = initial_seed + ep
            state, _, _ = self.initialize_episode(seed=ep)

            # TODO: add recording
            # if self.record:
            #     self.env.start_rec(f'{ep + 1:06d}')

            lang_goal = state['mission'] + '.'
            initial_state = state['language']
            admissible_actions_list = state['admissible_actions']
            plan_init = f'[Seed] {self.seed}\n' \
                        f'[Goal] {lang_goal}\n' \
                        f'[Initial State] {initial_state}\n'
            # admissible_actions = ', '.join(admissible_actions_list)

            if self.oracle_flag:  # collect dataset
                oracle_plan = plan_init
                try:
                    plan = self.oracle_planner(state)
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
                # context_pro = f'[Goal] {lang_goal} [Initial State] {initial_state}'
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
                  f"Saving trajectories for Task: BabyAI-Pickup\n"
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
                  f"Task: BabyAI-Pickup | LM: {self.model_name} | Scoring: {self.decoding_type}-{self.decoding_score}\n"
                  "===========================================================================\n")
            few_shot_prompts = self.few_shot_prompt.prompt()
            for ep in range(1, self.num_episodes + 1):
                print(ep)
                self.generate_plan_episode(ep, few_shot_prompts)

        self.plan_log_file.close()

    def test_plans(self, initial_seed):
        print(f"\n=========================== LM test - {self.domain.mode} =============================\n "
              f"Task: BabyAI-Pickup | LM: {self.model_name} | Scoring: {self.decoding_type}-{self.decoding_score}\n"
              "===========================================================================\n")
        # rewards = num success plans / total episodes
        rewards = 0.
        didnot_execute = 0.  # keeps track of plans that didn't execute in test sets
        self.initialize_episode(seed=0)

        # load saved plans
        genplan_file_path = self.plan_log_file.name
        # genplan_file_path = f'{os.environ["BabyAI"]}/pickup/data/' \
        #                     f'lm-plans/test-optimal/vicuna/beam_action-say/2023-07-03 10:05'
        gen_plans = \
            process_saved_plans(genplan_file_path,
                                task_completed_desc=self.task_completed_desc)

        # load oracle plans for that mode
        # only test-optimal is required for plan length comparison
        if self.domain.mode == 'test-optimal':
            oracle_plan_path = \
                f'{os.environ["BabyAI"]}/pickup/data/oracle-plans/{self.domain.mode}'
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
            state, info, _ = self.initialize_episode(seed=ep)

            # placing additional checks for safety
            assert plan.seed == self.seed
            assert plan.goal == state['mission'] + '.'
            # assert plan.initial_state == state['language']

            # if self.record:
            #     self.env.start_rec(f'{ep + 1:06d}')

            # execute each action of the plan:
            steps = 0
            for action in plan.actions:
                # if planner says Done even though the task is not complete
                if action == self.task_completed_desc:
                    break

                if action == "ExceededMaxSteps":
                    didnot_execute += 1
                    break
                # execute each step of the plan in the environment
                try:
                    state, _, done, info = \
                        self.env_step(action, state=state, info=info)
                    steps += 1
                    if done:
                        # for test-optimal, compare plan lengths wrt to the oracle plan
                        if self.domain.mode == 'test-optimal':
                            try:
                                if len(plan.actions) <= len(oracle_plans[ep].actions):
                                    rewards += 1
                                else:
                                    break  # skip to the next plan
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
                except AttributeError:
                    print(f"Seed {self.seed}: Target position unreachable")
                    continue
                except Exception as e:
                    print('===============================')
                    print(f'Error: {e} with seed: {self.seed}')
                    print('===============================')
        print(rewards / self.num_episodes, didnot_execute)
        return rewards / self.num_episodes, didnot_execute