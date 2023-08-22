import os
import sys

sys.path.append(os.environ['VIRTUALHOME'])
sys.path.append(os.environ['PLANNER'])
import glob
# import virtualhome
# from simulation.unity_simulator.comm_unity import UnityCommunication
# from simulation.unity_simulator import utils_viz
import hydra
from virtualhome_planner import LmAgent
import prompts
import logging
from dotmap import DotMap


@hydra.main(config_path=os.path.join(os.environ['VIRTUALHOME'], 'cfg'),
            config_name='inference')
def main(cfg):
    # Initialize communication (virtualhome simulator).
    env = None

    domain = DotMap(mode=None, tasks=None)
    domain.mode = cfg['domain']['mode']
    if domain.mode == 'test-generalize':  # set test-generalize episodes to 10
        cfg['n'] = 10
        cfg['max_steps'] = 20  # some leeway for test-generalize
    print(f"Mode: {domain.mode}")

    # initialize planner type
    planner = cfg['planner']
    oracle_flag = planner.agent_type == 'oracle'
    few_shot_prompt = ''
    if oracle_flag:
        pass
    else:
        # prompting for llms
        few_shot_prompt = prompts.names['virtualhome']()
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
        os.path.join(os.environ['VIRTUALHOME'], 'checkpoints')
    # sys.path.append(os.environ['CKPT_ROOT'])

    # Logger
    log_filename = os.path.join(os.environ['PLANNER'], 'all_test_logs_4')
    logging.basicConfig(filename=log_filename,  # the file where the logs will be stored
                        level=logging.INFO,  # the level of logging
                        format='%('
                               ')s\n%(message)s',  # including timestamp in logs
                        datefmt='%Y-%m-%d %H:%M')  # format of the timestamp
    logger = logging.getLogger()

    max_steps = cfg['max_steps']
    llm_planner = LmAgent(env, domain, cfg, model_name, decoding_type, decoding_score,
                          oracle_flag, few_shot_prompt, max_steps, save_dir)
    # generate and save plans
    llm_planner.generate_plans()

    # execute the saved plans to test feasibility
    # if not oracle_flag:
    #     osname = "linux"
    #     exec_file = '../linux_exec*.x86_64'
    #     exec_file_names = glob.glob(exec_file)
    #     if len(exec_file_names) > 0:
    #         exec_file_name = exec_file_names[0]
    #         env = UnityCommunication(file_name=exec_file_name, port="8082", x_display="0")
    #     else:
    #         print("Error: executable path not found.")
    #     llm_planner.env = env
    #     rewards, optimal_rewards = llm_planner.test_plans(genplan_file_path)
    #     logger.info(f"Task: VirtualHome | Mode: {domain.mode} | Model: {model_name} | "
    #                 f"Strategy: {decoding_type}-{decoding_score} | Rewards: {rewards} | "
    #                 f"Optimal Rewards: {optimal_rewards}")
    #     with open(os.path.join(os.environ['PLANNER'], 'all_test_logs_4'), 'a') as infile:
    #         infile.write(f"\nTask: VirtualHome | Mode: {domain.mode} | Model: {model_name} | "
    #                      f"Strategy: {decoding_type}-{decoding_score} | Rewards: {rewards} | "
    #                      f"Optimal Rewards: {optimal_rewards}\n")


if __name__ == '__main__':
    main()
