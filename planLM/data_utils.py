import os
import re
import pickle
import numpy as np


class ExceededMaxSteps(Exception):
    """Raised when maximum allowable steps/actions are exceeded"""
    pass


def load_checkpoint():
    pass


def save_checkpoint():
    pass


def load_from_disk(save_path):
    print(save_path)
    if os.path.isfile(save_path):
        with open(save_path, 'rb') as infile:
            return pickle.load(infile)
    print(f"{save_path} does not exist.")


def process_action(generated_action, admissible_actions):
    """matches LM generated action to all admissible actions
        and outputs the best matching action"""

    def editDistance(str1, str2, m, n):
        # Create a table to store results of sub-problems
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        # Fill d[][] in bottom up manner
        for i in range(m + 1):
            for j in range(n + 1):

                # If first string is empty, only option is to
                # insert all characters of second string
                if i == 0:
                    dp[i][j] = j  # Min. operations = j

                # If second string is empty, only option is to
                # remove all characters of second string
                elif j == 0:
                    dp[i][j] = i  # Min. operations = i

                # If last characters are same, ignore last char
                # and recur for remaining string
                elif str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]

                # If last character are different, consider all
                # possibilities and find minimum
                else:
                    dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                       dp[i - 1][j],  # Remove
                                       dp[i - 1][j - 1])  # Replace

        return dp[m][n]

    output_action = ''
    min_edit_dist_action = 100
    for action in admissible_actions:
        dist = editDistance(str1=generated_action, str2=action,
                            m=len(generated_action), n=len(action))
        if dist < min_edit_dist_action:
            output_action = action
            min_edit_dist_action = dist

    return output_action


def process_saved_plans(filepath, task_completed_desc):
    class Sample:
        """Dataset Sample Class"""

        def __init__(self, seed, goal, initial_state, actions):
            self.seed = seed
            self.goal = goal
            self.initial_state = initial_state
            self.actions = actions

    print(f'=============== Processing {filepath} =================')
    samples = []
    with open(filepath, 'r') as infile:
        lines = infile.readlines()
        i = 0
        while i < len(lines):
            if "[Seed]" in lines[i]:
                seed = int(lines[i].split("] ")[1].strip())
                goal_first = 'Goal' in lines[i+1] and 'Initial State' in lines[i+2]
                if goal_first:
                    goal = lines[i + 1].split("] ")[1].strip()
                    initial_state = lines[i + 2].split("] ")[1].strip()
                else:
                    initial_state = lines[i + 1].split("] ")[1].strip()
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
                samples.append(Sample(seed, goal, initial_state, actions))
            i += 1
    # sort by increasing order of seed value
    samples = sorted(samples, key=lambda sample: sample.seed)
    return samples


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
                goal_first = 'Goal' in lines[i+1] and 'Description' in lines[i+2]
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


def get_empty_pos(grid, agent_pos, fwd_pos):
    def bfs(grid, start, visited):
        queue = [start]
        room = set()
        while queue:
            (x, y, o) = queue.pop(0)
            if o.type == "empty" or o.type not in ['wall', 'door']:
                visited.add((x, y))
                room.add((x, y))
                if x > 0 and (x - 1, y) not in visited:
                    queue.append((x - 1, y, grid[x - 1][y][2]))
                if x < len(grid) - 1 and (x + 1, y) not in visited:
                    queue.append((x + 1, y, grid[x + 1][y][2]))
                if y > 0 and (x, y - 1) not in visited:
                    queue.append((x, y - 1, grid[x][y - 1][2]))
                if y < len(grid[x]) - 1 and (x, y + 1) not in visited:
                    queue.append((x, y + 1, grid[x][y + 1][2]))
        return room

    def get_rooms(grid):
        rooms = []  # list of tuples, each tuple is (x, y)
        visited = set()
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                cell = grid[i][j][2]
                if cell.type == "empty" or cell.type not in ['wall', 'door']:
                    if (i, j) not in visited:
                        room = bfs(grid, (i, j, cell), visited)
                        rooms.append(room)
        return rooms

    rooms = get_rooms(grid)  # list of tuples (x, y)
    # find agent_room
    for room in rooms:
        if agent_pos in room:
            agent_room = room
            break

    # retrieve an empty pos in agent_room
    for i, j in agent_room:
        obj = grid[i][j][2]
        # empty cell which is not adjacent to the cell the object was picked from
        if obj.type == "empty" and np.all([i, j] != fwd_pos):
            # if cell is not right next to a door:
            door_flag = False
            for i_del, j_del in zip([0, 0, 1, -1], [1, -1, 0, 0]):
                i_next, j_next = i + i_del, j + j_del
                next_cell = grid[i_next][j_next][2]
                if next_cell.type != "empty" and next_cell.type == 'door':
                    door_flag = True
            if not door_flag:
                empty_pos = (i, j)

    return empty_pos


def latest_file_from_dir(path):
    list_of_files = os.listdir(path)
    latest_file = \
        max([os.path.join(path, file) for file in list_of_files],
            key=os.path.getctime)
    return latest_file

