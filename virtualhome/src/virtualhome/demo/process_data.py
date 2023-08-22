import glob
import pickle

from utils_demo import *
import os
import re
import time
import random
# from tqdm import tqdm
# import virtualhome
from unity_simulator.comm_unity import UnityCommunication
from unity_simulator import utils_viz


def initialize_env():
    osname = "linux"
    exec_file = '../linux_exec*.x86_64'
    exec_file_names = glob.glob(exec_file)
    if len(exec_file_names) > 0:
        exec_file_name = exec_file_names[0]
        comm = UnityCommunication(file_name=exec_file_name, port="8080", x_display="0")
    else:
        print("Error: executable path not found.")


def run_script(graph_id, script):
    comm.reset(graph_id - 1)
    comm.add_character('chars/Male1')
    #     _, graph = comm.environment_graph()
    #     print(f'Title: {title} \nDescription: {description} \nScript: {script}')
    success, message_exec = comm.render_script(script=script[1:],
                                               processing_time_limit=60,
                                               find_solution=True,
                                               recording=False,
                                               skip_animation=True)
    print(message_exec)
    return success


def retrieve_script(raw_script):
    split_script = raw_script.split('[')
    graph_id = int(split_script[1].split(']')[1].strip())
    actions = [action.split(']')[1].replace(".", "").strip() for action in split_script[3:]][:-1]
    script = []
    for action in actions:
        split_action = action.split()
        action_verb = split_action[0]
        arguments = '.'
        if len(split_action) > 1:
            arguments = ' <' + '> (1) <'.join(split_action[1:]) + '> (1)'
        script.append(f'<char0> [{action_verb.upper()}]{arguments}')
    return graph_id, script


def test_gen_scripts(raw_script_gen, raw_script_opt):
    graph_id_gen, script_gen = retrieve_script(raw_script_gen)
    if raw_script_opt is not None:
        graph_id_opt, script_opt = retrieve_script(raw_script_opt)
        assert graph_id_gen == graph_id_opt
    try:
        success = run_script(graph_id_gen, script_gen)
    except:
        if raw_script_opt is None:   # for test-generalize
            return 0
        return 0, 0
    if success:
        if raw_script_opt is None:  # for test-generalize
            return 1
        if len(script_gen) <= len(script_opt):
            return 1, 1
        else:
            return 1, 0
    if raw_script_opt is None:  # for test-generalize
        return 0
    return 0, 0


def test_success_optimal(gen_file, test_file):
    succ_total = 0
    opt_total = 0
    initialize_env()

    def read_file_and_process(file_path):
        with open(file_path, 'r') as f:
            raw_scripts = f.readlines()
        processed_scripts = []
        processed_script = ''
        for line in raw_scripts:
            if line == '\n':
                processed_scripts.append(processed_script)
                processed_script = ''
            else:
                processed_script += line
        return processed_scripts

    raw_scripts_gen = read_file_and_process(gen_file)
    if test_file is not None:
        raw_scripts_opt = read_file_and_process(test_file)

    if test_file is not None:
        for raw_script_gen, raw_script_opt in zip(raw_scripts_gen, raw_scripts_opt):
            goal = raw_script_gen.split('[')[2].split(']')[1].strip()
            if 'ExceededMaxSteps' in raw_script_gen:
                succ_total += 0
                opt_total += 0
            else:
                succ_score, opt_score = test_gen_scripts(raw_script_gen, raw_script_opt)
                succ_total += succ_score
                opt_total += opt_score
            print(f'Goal: {goal} | Success Total: {succ_total} | Optimal Total: {opt_total}')
    else:  # for test-generalize
        for raw_script_gen in raw_scripts_gen:
            goal = raw_script_gen.split('[')[2].split(']')[1].strip()
            if 'ExceededMaxSteps' in raw_script_gen:
                succ_total += 0
            else:
                succ_score = test_gen_scripts(raw_script_gen, raw_script_opt=None)
                succ_total += succ_score
            print(f'Goal: {goal} | Success Total: {succ_total}')


gen_file = 'saycanpay/virtualhome/src/virtualhome/data/lm-plans/test-generalize/vicuna/greedy_action-say/2023-07' \
           '-23 13:31'
test_file = 'saycanpay/virtualhome/src/virtualhome/data/oracle-plans/test-generalize/2023-07-23 09:09'
# gen_file = 'saycanpay/virtualhome/src/virtualhome/data/oracle-plans/test-generalize/2023-07-23 09:09'
# test_file = None
test_success_optimal(gen_file, test_file)


def transform_string(s):
    # Regular expression to capture a decimal number
    pattern = r"\(([0-9]*\.[0-9]*)\)"
    # Use the re.sub function to replace the matched patterns
    return re.sub(pattern, f"(1)", s)


def parse_string_for_lm(s):
    pattern = r'<(.*?)> \[(.*?)\](.*?)(\(1\)|$)'
    match = re.search(pattern, s)
    additional_words = ' '.join(re.findall(r'<(.*?)>', match.group(3)))
    return f"{match.group(2).lower()} {additional_words.lower()}"


def filter_data():
    log_file_path = '../processed_data'
    log_file = open(log_file_path, 'a')
    count = 0
    net = 0
    admissible_actions = set()
    data_root = '../dataset'
    initialize_env()
    for graph_id in range(1, 8, 1):
        script_path = f'executable_programs/TrimmedTestScene{graph_id}_graph/results_intentions_march-13-18'
        script_root = os.path.join(data_root, script_path)
        completed = 0
        script_file_names = os.listdir(script_root)[completed:]
        for file_index, file_name in enumerate(script_file_names):
            script_file = os.path.join(script_root, file_name)
            # print every 20 to resume from the last processed file
            net_file_index = completed + file_index
            if net_file_index % 20 == 0:
                print(f'{graph_id}.{net_file_index}')
            with open(script_file, 'r') as f:
                content = f.readlines()
                content = [x.strip() for x in content]
                title = content[0]
                description = content[1]
                script_raw = content[4:]
            script = []
            for elem in script_raw:
                if 'dining_room' not in elem and 'home_office' not in elem:
                    script.append(f'<char0> {transform_string(elem)}')
            success = run_script(graph_id, script)
            if success is True:
                count += 1
                print(f'{count}/{net}')
                plan_init = f'[Scene] {graph_id}\n' \
                            f'[Goal] {title.lower()}\n' \
                            f'[Description] {description.lower()}\n'
                lm_steps = ''
                for i, step in enumerate(script):
                    step = parse_string_for_lm(step)
                    lm_steps += f'[Step {i + 1}] {step}\n'
                    admissible_actions.add(step)
                lm_steps += f'[Step {i + 2}] done task'
                plan = f'{plan_init}{lm_steps}\n'
                log_file.write(plan)
                log_file.write('\n')
            net += 1
    log_file.close()
    with open('../admissible_actions', 'wb') as f:
        pickle.dump(admissible_actions, f)
        print(admissible_actions)
    print(f'Executed: {count} | Total: {net}')


def process_saved_plans_virtualhome(filepath, task_completed_desc):
    class Sample:
        """Dataset Sample Class"""

        def __init__(self, graph_id, goal, description, actions):
            self.graph_id = graph_id
            self.goal = goal
            self.description = description
            self.actions = actions

    print(f'=============== Processing {filepath} =================')
    samples = []
    with open(filepath, 'r') as infile:
        lines = infile.readlines()
        i = 0
        while i < len(lines):
            if "[Scene]" in lines[i]:
                graph_id = int(lines[i].split("] ")[1].strip())
                goal_first = 'Goal' in lines[i + 1] and 'Description' in lines[i + 2]
                if goal_first:
                    goal = lines[i + 1].split("] ")[1].strip()
                    description = lines[i + 2].split("] ")[1].strip()
                else:
                    description = lines[i + 1].split("] ")[1].strip()
                    goal = lines[i + 2].split("] ")[1].strip()
                actions = []
                i += 3
                if "ExceededMaxSteps" in lines[i]:
                    actions.append("ExceededMaxSteps")
                    i += 1
                else:
                    while i < len(lines) and task_completed_desc not in lines[i]:
                        # removing (consecutive) repeated actions
                        # assuming no recursive / repetitive actions
                        action_step_match = re.search(r'\[Step \d+\] (.*)', lines[i])
                        if action_step_match:
                            act = action_step_match.group(1)
                            if len(actions) > 0 and act != actions[-1]:
                                actions.append(act)
                            elif len(actions) == 0:
                                actions.append(act)
                        i += 1
                # actions.append(task_completed_desc)
                samples.append(Sample(graph_id, goal, description, actions))
            i += 1
    # sort by increasing order of seed value
    # samples = sorted(samples, key=lambda sample: sample.seed)
    return samples


def split_train_test():
    task_completed_desc = 'done task'
    log_file_path = '../processed_data'
    samples = process_saved_plans_virtualhome(log_file_path,
                                              task_completed_desc=task_completed_desc)
    print(f'---------- Total Samples: {len(samples)} ----------')  # 845
    num_test = 50
    num_train = len(samples) - num_test
    train_ids = random.sample(range(len(samples)), num_train)
    test_ids = [i for i in range(len(samples)) if i not in train_ids]

    admissible_actions = set()
    for sample in samples:
        admissible_actions = admissible_actions.union(set(sample.actions))

    def save_train_test_samples(dir_path, sample_ids):
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, time.strftime("%Y-%m-%d %H:%M")), 'w') as file:
            for sample_id in sample_ids:
                train_sample = samples[sample_id]
                scene = train_sample.graph_id
                goal = train_sample.goal
                description = train_sample.description
                actions = train_sample.actions
                actions = '\n'.join([f'[Step {i + 1}] {action}.' for i, action in enumerate(actions)]) \
                          + f'\n[Step {len(actions) + 1}] {task_completed_desc}.'
                file.write(f'[Scene] {scene}\n'
                           f'[Goal] {goal}.\n'
                           f'[Description] {description}\n'
                           f'{actions}\n\n')

    root_dir = '../data/oracle-plans'
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    save_train_test_samples(train_dir, train_ids)
    save_train_test_samples(test_dir, test_ids)
    with open('../admissible_actions', 'wb') as f:
        pickle.dump(admissible_actions, f)
        print(admissible_actions)
        print(f'Total Admissible Actions: {len(admissible_actions)}')  # 113

# filter_data()
# split_train_test()
