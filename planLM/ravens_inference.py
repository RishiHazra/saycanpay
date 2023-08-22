import os
import sys

sys.path.append(os.environ['RAVENS_ROOT'])
sys.path.append(os.environ['PLANNER'])
from ravens import tasks
from ravens.environments.environment import Environment
import hydra
from ravens_planner import LmAgent
import prompts
import logging


def get_initial_seed(mode, train_seeds, test_seeds):
    """get initial seed for each data split"""
    if mode == 'train':
        seed = 0  # NOTE: beware of increasing set to > 300
    elif mode == 'test-success':
        seed = train_seeds
    elif mode == 'test-optimal':
        seed = train_seeds + test_seeds
    elif mode == 'test-generalize':  # more objects, unseen color, more distractors
        seed = train_seeds + 2 * test_seeds
    else:
        raise Exception("Invalid mode. Valid options are: "
                        "train, test-success, test-optimal, test-generalize")
    return seed


@hydra.main(config_path=os.path.join(os.environ['RAVENS_ROOT'], 'ravens/cfg'),
            config_name='inference')
def main(cfg):
    # Initialize environment and task.
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    domain = tasks.names[cfg['task']]()
    domain.mode = cfg['mode']
    if not domain.mode == 'train':  # set test episodes to 100
        cfg['n'] = 100
    print(f"Mode: {domain.mode}")

    # set initial seed, change seed based on this for different splits
    initial_seed = get_initial_seed(domain.mode,
                                    train_seeds=800,
                                    test_seeds=100)

    # initialize planner type
    planner = cfg['planner']
    oracle_flag = planner.agent_type == 'oracle'
    if oracle_flag:
        agent = domain.oracle(env)
        few_shot_prompt = ''
    else:
        agent = domain.langAgent(env)
        # prompting for llms
        few_shot_prompt = prompts.names[cfg['task']]()
    # used when agent_type = 'lm'
    model_name = planner.model_name
    decoding_type = planner.decoding_type
    decoding_score = planner.decoding_score

    # save_paths
    save_data_flag = cfg['save_data']
    save_dir = cfg['save_dir']  # to save the generated plans
    if not oracle_flag:
        save_dir = f"{save_dir}/{model_name}/{decoding_type}-{decoding_score}"
    os.makedirs(save_dir, exist_ok=True)  # makedir if not already exists
    if save_data_flag:
        print(f"Saving plans to: {save_dir}")

    os.environ['CKPT_ROOT'] = \
        os.path.join(os.environ['RAVENS_ROOT'], cfg['task'], 'checkpoints')
    # sys.path.append(os.environ['CKPT_ROOT'])

    # Logger
    log_filename = os.path.join(os.environ['PLANNER'], 'all_test_logs')
    logging.basicConfig(filename=log_filename,  # the file where the logs will be stored
                        level=logging.INFO,  # the level of logging
                        format='%('
                               ')s\n%(message)s',  # including timestamp in logs
                        datefmt='%Y-%m-%d %H:%M')  # format of the timestamp
    logger = logging.getLogger()

    max_steps = cfg['max_steps']
    llm_planner = LmAgent(agent, env, domain, cfg,
                          model_name, decoding_type, decoding_score,
                          oracle_flag, few_shot_prompt,
                          initial_seed, max_steps, save_dir)
    # generate and save plans
    llm_planner.generate_plans()
    # execute the saved plans to test feasibility
    if not oracle_flag:
        rewards, didnot_execute = llm_planner.test_plans(initial_seed)
        logger.info(f"Task: {cfg['task']} | Mode: {domain.mode} | Model: {model_name} | "
                    f"Strategy: {decoding_type}-{decoding_score} | Rewards: {rewards} | "
                    f"Not-Executed: {didnot_execute}")
        with open(os.path.join(os.environ['PLANNER'], 'all_test_logs_3'), 'a') as infile:
            infile.write(f"\nTask: {cfg['task']} | Mode: {domain.mode} | Model: {model_name} | "
                         f"Strategy: {decoding_type}-{decoding_score} | Rewards: {rewards} | "
                         f"Not-Executed: {didnot_execute}\n")


if __name__ == '__main__':
    # TODO: should the expert data be deterministic or stochastic?
    main()
