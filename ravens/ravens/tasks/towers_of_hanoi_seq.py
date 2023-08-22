"""Towers of Hanoi task."""

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

import pybullet as p
import random
from ordered_set import OrderedSet
from itertools import product


class TowersOfHanoiSeq(Task):
    """Towers of Hanoi Sequence base class and task."""

    def __init__(self):
        super().__init__()
        # TODO: mistakes can be allowed for LM (feasible, generalize splits)
        self.max_steps = 10
        self.lang_template = "move the {obj} in {loc}."
        self.task_completed_desc = "done putting disks in rods"
        self.lang_initial_state = ""
        self.lang_observation = ""
        self.admissible_actions = ""
        self.obj_in_tar_rel = []
        self.obj_in_tar_rel_old = []

    def reset(self, env):
        super().reset()
        n_disks = np.random.randint(2, 4)
        if self.mode == 'test-generalize':
            n_disks = 4  # for more disks, we need to generate obj file using blender
            # self.max_steps = 10

        # Add stand.
        base_size = (0.12, 0.36, 0.01)
        base_urdf = 'hanoi/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        base_id = env.add_object(base_urdf, base_pose, 'fixed')
        self.ind2obj[base_id] = 'base'

        # All colors.
        full_color_names = self.get_colors()

        # Choose three colors for three disks.
        color_names = random.sample(full_color_names, n_disks)
        colors = [utils.COLORS[cn] for cn in color_names]

        # Rod positions in base coordinates.
        rod_pos = ((0, -0.12, 0.03), (0, 0, 0.03), (0, 0.12, 0.03))
        # rod_names = ('lighter brown side', 'middle of the stand', 'darker brown side')
        rod_names = ('rod 1', 'rod 2', 'rod 3')
        # storing rod indices as 20 + ind to be safe
        self.ind2obj.update({20 + ind: name for ind, name in enumerate(rod_names)})

        # Add disks.
        disks = []
        disks_names = {}
        # n_disks = 3
        for i in range(n_disks):
            disk_urdf = 'hanoi/disk%d.urdf' % i
            pos = utils.apply(base_pose, rod_pos[0])
            z = 0.015 * (n_disks - i - 2)
            pos = (pos[0], pos[1], pos[2] + z)
            disk_id = env.add_object(disk_urdf, (pos, base_pose[1]))
            p.changeVisualShape(disk_id, -1, rgbaColor=colors[i] + [1])
            disks.append(disk_id)
            disks_names[disk_id] = color_names[i] + ' disk'
            self.ind2obj[disk_id] = color_names[i] + ' disk'

        # sample disk (hardest one is lowest disk)
        all_rods_ids = [21, 22]
        pick_disk_id = np.random.choice(list(disks_names.keys()))
        place_rod_id = np.random.choice(all_rods_ids)  # any of the other two rods

        # Solve Hanoi sequence with dynamic programming.
        hanoi_steps = []  # [[object index, from rod, to rod], ...]

        def solve_hanoi_seq(n, t0, t1, t2):
            if n == 0:
                hanoi_steps.append([list(disks_names.keys())[n], t0, t1])
                return
            solve_hanoi_seq(n - 1, t0, t2, t1)
            hanoi_steps.append([list(disks_names.keys())[n], t0, t1])
            solve_hanoi_seq(n - 1, t2, t1, t0)

        # find #disks on top of pick_disk_id
        disks_on_top = list(disks_names.keys()).index(pick_disk_id)
        support_rod = [rod_id for rod_id in all_rods_ids if rod_id != place_rod_id][0]
        if disks_on_top > 0:
            # move all top disks to the support rod
            solve_hanoi_seq(disks_on_top - 1, 20, support_rod, place_rod_id)
        hanoi_steps.append([pick_disk_id, 20, place_rod_id])
        # solve_hanoi_seq(n_disks - 1, 0, 2, 1)

        # Goal: pick and place disks using Hanoi sequence.
        for step in hanoi_steps:
            disk_id, _, rod_id = step
            place_pos = (utils.apply(base_pose, rod_pos[rod_id - 20]),
                         (0, 0, 0, 1))
            self.goals.append(([(disk_id, (0, None))], np.int32([[1]]),
                               {rod_id: place_pos},
                               False, True, 'pose', None, 1 / len(hanoi_steps)))
        # place_pos = (utils.apply(base_pose, rod_pos[place_rod_id - 20]),
        #              (0, 0, 0, 1))
        # self.goals.append(([(pick_disk_id, (0, None))], np.int32([[1]]),
        #                    {place_rod_id: place_pos},
        #                    False, True, 'pose', None, 1 / len(hanoi_steps)))
        self.lang_goals.append(self.lang_template.format(obj=self.ind2obj[pick_disk_id],
                                                         loc=self.ind2obj[place_rod_id]))
        all_rods = {}
        all_disks = []
        for obj_id, obj_name in self.ind2obj.items():
            if 'disk' not in obj_name and 'base' not in obj_name:
                recep_pos = (utils.apply(base_pose, rod_pos[obj_id - 20]),
                             (0, 0, 0, 1))
                all_rods[obj_id] = recep_pos
            elif 'disk' in obj_name:  # disk
                all_disks.append((obj_id, (0, None)))

        self.all_goals = [(all_disks, np.int32([[1]]),
                          all_rods,
                          False, True, 'pose', None, 1 / len(hanoi_steps))]
        # get natural language initial scene description
        self.observation()
        self.lang_initial_state = f'{self.lang_observation}The disks can be moved in ' \
                                  f'rod 1, rod 2, rod 3. '
        self.admissible_actions = \
            [f"put {disk} in {rod}" for disk, rod in product(disks_names.values(), rod_names)]

    def observation(self):
        """records the next observation (in language):
        change in state after executing an action"""

        # update the relational representation of the scene
        self.relations()
        self.lang_observation = ''

        # undone updates from previous states
        remove = self.obj_in_tar_rel_old - self.obj_in_tar_rel
        if len(remove) > 0:
            for (disk_obj_id_1, disk_obj_id_2) in remove.items:
                obj_1, obj_2 = \
                    self.ind2obj[disk_obj_id_1], self.ind2obj[disk_obj_id_2]
                disk_rel = 'disk' in obj_2
                if disk_rel:
                    self.lang_observation += \
                        '{} not on top of {}. '.format(obj_1, obj_2)
                else:
                    self.lang_observation += \
                        '{} not in {}. '.format(obj_1, obj_2)

        # new updates
        add = self.obj_in_tar_rel - self.obj_in_tar_rel_old
        if len(add) > 0:
            for (disk_obj_id_1, disk_obj_id_2) in add.items:
                obj_1, obj_2 = \
                    self.ind2obj[disk_obj_id_1], self.ind2obj[disk_obj_id_2]
                disk_rel = 'disk' in obj_2
                if disk_rel:
                    self.lang_observation += \
                        '{} on top of {}. '.format(obj_1, obj_2)
                else:
                    self.lang_observation += \
                        '{} in {}. '.format(obj_1, obj_2)

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode != 'test-generalize' else utils.EVAL_COLORS

    def relations(self):
        """ (a, b) where 'a' is on top of 'b'"""
        self.obj_in_tar_rel_old = self.obj_in_tar_rel.copy()
        self.obj_in_tar_rel = OrderedSet()

        # rod and base positions
        rod_pos = ((0, -0.12, 0.03), (0, 0, 0.03), (0, 0.12, 0.03))
        rod_ids = []
        for k, v in self.ind2obj.items():
            if 'base' not in v and 'disk' not in v:
                rod_ids.append(k)
        base_pos = \
            p.getBasePositionAndOrientation(
                list(filter(lambda k: self.ind2obj[k] == 'base', self.ind2obj))[0])
        rod_to_disk_pos = {rod_id: (utils.apply(base_pos, pos), (0, 0, 0, 1))
                           for rod_id, pos in zip(rod_ids, rod_pos)}

        # find disk positions
        poses = {}
        for obj_id, obj in self.ind2obj.items():
            if 'disk' in obj:
                obj_pose = p.getBasePositionAndOrientation(obj_id)
                poses[obj_id] = obj_pose

        # obj_in_region_rel = set()
        # use disk positions to get a relational representation (small_disk, big_disk)
        for rod_id, rod_pos in rod_to_disk_pos.items():
            # get the bottommost disk for each rod
            bottommost = []
            for disk_id_1, disk_pos_1 in poses.items():
                pose, _ = disk_pos_1
                # keep track of the disk immediately below disk_1
                # to adhere to transitive relations
                immediate_below = []
                for disk_id_2, disk_pos_2 in poses.items():
                    if self.is_match(disk_pos_1, disk_pos_2, 0) \
                            and disk_id_2 > disk_id_1:
                        immediate_below.append(disk_id_2)
                if len(immediate_below) > 0:
                    immediate_below = min(immediate_below)
                    self.obj_in_tar_rel.add((min(disk_id_1, immediate_below),
                                             max(disk_id_1, immediate_below)))

                # bottommost
                if self.is_match(rod_pos, disk_pos_1, 0):
                    bottommost.append(disk_id_1)
            if len(bottommost) > 0:
                # append to self.obj_in_tar_rel at the end
                self.obj_in_tar_rel.add((max(bottommost), rod_id))

        # self.obj_in_tar_rel = self.obj_in_tar_rel.items
