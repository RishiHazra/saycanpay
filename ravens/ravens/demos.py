"""Data collection script."""

import os
import sys

sys.path.append(os.environ['RAVENS_ROOT'])
import hydra
import numpy as np
import random
from ravens import tasks
from ravens.dataset import RavensDataset
from ravens.environments.environment import Environment


@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
    # Initialize environment and task.
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']
    record = cfg['record']['save_video']
    save_data = cfg['save_data']

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
    # agent = task.langAgent(env)
    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")

    # splits: train, test (executability, optimality, generalizability)

    # test sets should not be overlapping with train (ensure through seeds?)
    # generalizability need not have seeds? since we will increase the number of objects

    # 1. executability: recreate episodes with the same seed, no need to save data?
    # evaluate: plan, then run plan in simulation, check reward

    # 2. optimality: save data and expert plans and cost of expert plans (use reproducible episodes)
    # evaluate: check if success, then compare cost with expert plan

    # 3. generalizability: save data, expert plans might not be needed?
    # evaluate: execute the plan, reward = binary (1 for every executable plan, 0 otherwise)

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
    if seed < 0:
        if task.mode == 'train':
            seed = -2
        elif task.mode == 'val':  # NOTE: beware of increasing val set to >100
            seed = -1
        elif task.mode == 'test':
            seed = -1 + 10000
        else:
            raise Exception("Invalid mode. Valid options: train, val, test")

    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < cfg['n']:
        episode, total_reward = [], 0

        seed += 2
        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        print('Oracle demo: {}/{}'.format(dataset.n_episodes + 1, cfg['n']))

        env.set_task(task)
        obs = env.reset()
        info = env.info
        reward = 0

        # Unlikely, but a safety check to prevent leaks.
        if task.mode == 'val' and seed > (-1 + 10000):
            raise Exception("!!! Seeds for val set will overlap with the test set !!!")

        # Start video recording (NOTE: super slow)
        if record:
            env.start_rec(f'{dataset.n_episodes + 1:06d}')

        pick_obj_indices = []
        place_obj_indices = []
        for k, v in env.task.ind2obj.items():
            if 'block' in v:
                pick_obj_indices.append(k)
            elif 'bowl' in v:
                place_obj_indices.append(k)

        lang_goal = info['lang_goal']
        print(f'Initial Scene: {env.task.lang_initial_state}')
        print(f'Goal: {lang_goal}')

        # Rollout expert policy
        updated_scene = 'you have completed: '
        for _ in range(task.max_steps):
            action = (np.random.choice(pick_obj_indices),
                      np.random.choice(place_obj_indices))
            print('action: pick {} and place in {}'.format(env.task.ind2obj[action[0]],
                                                           env.task.ind2obj[action[1]]))
            act = agent.act(action, obs, info)  # pick_pose, place_pose
            episode.append((obs, act, reward, info))
            # lang_goal = info['lang_goal']
            obs, reward, done, info, obs_lang = env.step(act)
            print('next scene: {}'.format(obs_lang))
            if obs_lang != '':
                updated_scene += obs_lang + '. '

            print('updated scene: {}'.format(updated_scene))
            total_reward += reward
            print(f'Total Reward: {total_reward:.3f} | Done: {done}')
            if done:
                break
        episode.append((obs, None, reward, info))

        # End video recording
        if record:
            env.end_rec()
        print('\n ------------------------- \n')

        # Only save completed demonstrations.
        # if save_data and total_reward > 0.99:
        #     dataset.add(seed, episode)


if __name__ == '__main__':
    main()
