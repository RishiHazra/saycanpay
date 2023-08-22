"""Stacking task."""

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

import pybullet as p


class StackBlockPyramid(Task):
    """Stacking task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 12
        self.lang_template = "make the {row} row with {blocks}"
        self.task_completed_desc = "done stacking block pyramid."

    def reset(self, env):
        super().reset(env)

        # Add base.
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # Block colors.
        colors = [
            utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
            utils.COLORS['yellow'], utils.COLORS['orange'], utils.COLORS['red']
        ]

        # Add blocks.
        objs = []
        # sym = np.pi / 2
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in a pyramid (bottom row: green, blue, purple).
        self.goals.append((objs[:3], np.ones((3, 3)), targs[:3],
                           False, True, 'pose', None, 1 / 2))
        self.lang_goals.append(self.lang_template.format(blocks="the green, blue and purple blocks",
                                                         row="bottom"))

        # Goal: blocks are stacked in a pyramid (middle row: yellow, orange).
        self.goals.append((objs[3:5], np.ones((2, 2)), targs[3:5],
                           False, True, 'pose', None, 1 / 3))
        self.lang_goals.append(self.lang_template.format(blocks="the yellow and orange blocks",
                                                         row="middle"))

        # Goal: blocks are stacked in a pyramid (top row: red).
        self.goals.append((objs[5:], np.ones((1, 1)), targs[5:],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(blocks="the red block",
                                                         row="top"))
