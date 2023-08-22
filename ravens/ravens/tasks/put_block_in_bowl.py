"""Put Blocks in Bowl Task."""

import random

import numpy as np
import pybullet as p
from ravens.tasks.task import Task
from ravens.utils import utils
from itertools import product


class PutBlockInBowl(Task):
    """Put Blocks in Bowl base class and task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put the {pick} blocks in {place} bowls."
        self.lang_initial_state = ""
        self.lang_observation = ""
        self.admissible_actions = ""
        self.task_completed_desc = "done placing blocks in bowls"
        # self.ind2obj = dict()
        self.obj_in_tar_rel = []
        self.obj_in_tar_rel_old = []

    def reset(self, env):
        super().reset()
        min_bowls, max_bowls = 2, 5
        min_blocks = 2
        max_distractors = 4
        all_color_names = self.get_colors()
        if self.mode == 'test-generalize':  # generalization episode has longer episodes
            min_bowls, max_bowls = 5, 7
            min_blocks = min_bowls
            # max_distractors = 6

        n_bowls = np.random.randint(min_bowls, max_bowls)
        n_blocks = np.random.randint(min_blocks, n_bowls + 1)

        selected_color_names = random.sample(all_color_names, 2)
        colors = [utils.COLORS[cn] for cn in selected_color_names]

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = {}
        all_bowl_poses = {}
        for _ in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=colors[1] + [1])
            bowl_poses[bowl_id] = bowl_pose
            all_bowl_poses[bowl_id] = bowl_pose
            self.ind2obj[bowl_id] = selected_color_names[1] + ' bowl'

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        all_blocks = []
        for _ in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)  # position, rotation
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[0] + [1])
            blocks.append((block_id, (0, None)))
            all_blocks.append((block_id, (0, None)))
            self.ind2obj[block_id] = selected_color_names[0] + ' block'

        # Goal: put each block in a different bowl.
        self.goals.append((blocks, np.ones((len(blocks), len(bowl_poses))),
                           bowl_poses, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template.format(pick=selected_color_names[0],
                                                         place=selected_color_names[1]))

        # Only one mistake allowed.
        # TODO: executability and generalizability splits should get longer attempts
        self.max_steps = len(blocks) + 1

        # Colors of distractor objects.
        distractor_bowl_colors, distractor_block_color_names = [], []
        distractor_block_colors, distractor_bowl_color_names = [], []
        for c in utils.COLORS:
            if c not in selected_color_names:
                distractor_bowl_colors.append(utils.COLORS[c])
                distractor_bowl_color_names.append(c)
                distractor_block_colors.append(utils.COLORS[c])
                distractor_block_color_names.append(c)

        # distractor_bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        # distractor_block_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]

        # Add distractors.
        n_distractors = 0
        while n_distractors < max_distractors:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = distractor_block_colors if is_block else distractor_bowl_colors
            color_names = distractor_block_color_names if is_block else distractor_bowl_color_names
            pose = self.get_random_pose(env, size)
            if not pose:
                continue
            obj_id = env.add_object(urdf, pose)
            color, color_name = colors[n_distractors % len(colors)], color_names[n_distractors % len(colors)]
            if not obj_id:
                continue
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            if is_block:
                all_blocks.append((obj_id, (0, None)))
                self.ind2obj[obj_id] = color_name + ' block'
            else:
                all_bowl_poses[obj_id] = pose
                self.ind2obj[obj_id] = color_name + ' bowl'

            n_distractors += 1

        self.all_goals.append((all_blocks, np.ones((len(all_blocks), len(all_bowl_poses))),
                               all_bowl_poses, False, True, 'pose', None, 1))

        def add_unique_identity(input_dict):
            """adds unique identifier to each object: 2 red blocks: red block 1, red block 2"""
            counter_dict = {}
            output_dict = {}

            for key, value in input_dict.items():
                if value in counter_dict:
                    counter_dict[value] += 1
                else:
                    counter_dict[value] = 1

                output_dict[key] = f"{value} {counter_dict[value]}"
            return output_dict

        # get a one-to-one mapping of index to object mapping
        self.ind2obj = add_unique_identity(self.ind2obj)
        # get natual language initial scene description (obtained through prompt engineering)
        # self.lang_initial_state = '["' + '", "'.join(list(self.ind2obj.values())) + '"]. '
        self.lang_initial_state = 'There is a ' + ', '.join(list(self.ind2obj.values())) + '.'
        unique_id_blocks = [obj for obj in self.ind2obj.values() if 'block' in obj]
        unique_id_bowls = set(list(self.ind2obj.values())) - set(unique_id_blocks)
        self.admissible_actions = \
            [f"place {disk} in {rod}" for disk, rod in product(unique_id_blocks, unique_id_bowls)]

    def observation(self):
        """records the next observation (language):
        change in state after executing an action"""

        # update the relational representation of the scene
        self.relations()
        self.lang_observation = ''

        # undone updates from previous states
        remove = set(self.obj_in_tar_rel_old) - set(self.obj_in_tar_rel)
        if len(remove) > 0:
            for (pick_obj_id, place_obj_id) in remove:
                self.lang_observation += \
                    '{} not in {}. '.format(self.ind2obj[pick_obj_id],
                                            self.ind2obj[place_obj_id])

        # new updates
        add = set(self.obj_in_tar_rel) - set(self.obj_in_tar_rel_old)
        if len(add) > 0:
            for (pick_obj_id, place_obj_id) in add:
                self.lang_observation += \
                    '{} in {}. '.format(self.ind2obj[pick_obj_id],
                                        self.ind2obj[place_obj_id])

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode != 'test-generalize' else utils.EVAL_COLORS

    def relations(self):
        self.obj_in_tar_rel_old = self.obj_in_tar_rel.copy()
        objs, _, targs, _, _, _, _, _ = self.all_goals[0]
        self.obj_in_tar_rel = []
        for obj in objs:
            obj_id, (_, _) = obj
            obj_pose = p.getBasePositionAndOrientation(obj_id)
            for tar_id, tar_pose in targs.items():
                if self.is_match(obj_pose, tar_pose, 0):
                    self.obj_in_tar_rel.append((obj_id, tar_id))
