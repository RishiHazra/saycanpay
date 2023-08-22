"""Packing Box Pairs task."""

import os

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

import pybullet as p


class PackingBoxesPairsUnseenColors(Task):
    """Packing Box Pairs task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "pack all the {colors} blocks into the brown box" # should have called it boxes :(
        self.task_completed_desc = "done packing blocks."

        # Tight z-bound (0.0525) to discourage stuffing everything into the brown box
        self.zone_bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.0525]])

    def reset(self, env):
        super().reset(env)

        # Add container box.
        zone_size = self.get_random_size(0.05, 0.3, 0.05, 0.3, 0.05, 0.05)
        zone_pose = self.get_random_pose(env, zone_size)
        container_template = 'container/container-template.urdf'
        half = np.float32(zone_size) / 2
        replace = {'DIM': zone_size, 'HALF': half}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')
        if os.path.exists(container_urdf):
            os.remove(container_urdf)

        margin = 0.01
        min_object_dim = 0.05
        bboxes = []

        class TreeNode:

            def __init__(self, parent, children, bbox):
                self.parent = parent
                self.children = children
                self.bbox = bbox  # min x, min y, min z, max x, max y, max z

        def KDTree(node):
            size = node.bbox[3:] - node.bbox[:3]

            # Choose which axis to split.
            split = size > 2 * min_object_dim
            if np.sum(split) == 0:
                bboxes.append(node.bbox)
                return
            split = np.float32(split) / np.sum(split)
            split_axis = np.random.choice(range(len(split)), 1, p=split)[0]

            # Split along chosen axis and create 2 children
            cut_ind = np.random.rand() * \
                      (size[split_axis] - 2 * min_object_dim) + \
                      node.bbox[split_axis] + min_object_dim
            child1_bbox = node.bbox.copy()
            child1_bbox[3 + split_axis] = cut_ind - margin / 2.
            child2_bbox = node.bbox.copy()
            child2_bbox[split_axis] = cut_ind + margin / 2.
            node.children = [
                TreeNode(node, [], bbox=child1_bbox),
                TreeNode(node, [], bbox=child2_bbox)
            ]
            KDTree(node.children[0])
            KDTree(node.children[1])

        # Split container space with KD trees.
        stack_size = np.array(zone_size)
        stack_size[0] -= 0.01
        stack_size[1] -= 0.01
        root_size = (0.01, 0.01, 0) + tuple(stack_size)
        root = TreeNode(None, [], bbox=np.array(root_size))
        KDTree(root)

        all_color_names = [c for c in self.get_colors()]
        relevant_color_names = np.random.choice(all_color_names, min(2, len(bboxes)), replace=False)
        distractor_color_names = [c for c in all_color_names if c not in relevant_color_names]

        pack_colors = [utils.COLORS[c] for c in relevant_color_names]
        distractor_colors = [utils.COLORS[c] for c in distractor_color_names]

        # Add objects in container.
        object_points = {}
        object_ids = []
        bboxes = np.array(bboxes)
        object_template = 'box/box-template.urdf'
        for bbox in bboxes:
            size = bbox[3:] - bbox[:3]
            position = size / 2. + bbox[:3]
            position[0] += -zone_size[0] / 2
            position[1] += -zone_size[1] / 2
            pose = (position, (0, 0, 0, 1))
            pose = utils.multiply(zone_pose, pose)
            urdf = self.fill_template(object_template, {'DIM': size})
            box_id = env.add_object(urdf, pose)
            if os.path.exists(urdf):
                os.remove(urdf)
            object_ids.append((box_id, (0, None)))
            icolor = np.random.choice(range(len(pack_colors)), 1).squeeze()
            p.changeVisualShape(box_id, -1, rgbaColor=pack_colors[icolor] + [1])
            object_points[box_id] = self.get_box_object_points(box_id)

        # Randomly select object in box and save ground truth pose.
        object_volumes = []
        true_poses = []
        for object_id, _ in object_ids:
            true_pose = p.getBasePositionAndOrientation(object_id)
            object_size = p.getVisualShapeData(object_id)[0][3]
            object_volumes.append(np.prod(np.array(object_size) * 100))
            pose = self.get_random_pose(env, object_size)
            p.resetBasePositionAndOrientation(object_id, pose[0], pose[1])
            true_poses.append(true_pose)

        # Add distractor objects
        num_distractor_objects = 4
        distractor_bbox_idxs = np.random.choice(len(bboxes), num_distractor_objects)
        for bbox_idx in distractor_bbox_idxs:
            bbox = bboxes[bbox_idx]
            size = bbox[3:] - bbox[:3]
            position = size / 2. + bbox[:3]
            position[0] += -zone_size[0] / 2
            position[1] += -zone_size[1] / 2

            pose = self.get_random_pose(env, size)
            urdf = self.fill_template(object_template, {'DIM': size})
            box_id = env.add_object(urdf, pose)
            if os.path.exists(urdf):
                os.remove(urdf)
            icolor = np.random.choice(range(len(distractor_colors)), 1).squeeze()
            if box_id:
                p.changeVisualShape(box_id, -1, rgbaColor=distractor_colors[icolor] + [1])

        # Some scenes might contain just one relevant block that fits in the box.
        if len(relevant_color_names) > 1:
            relevant_desc = f'{relevant_color_names[0]} and {relevant_color_names[1]}'
        else:
            relevant_desc = f'{relevant_color_names[0]}'

        self.goals.append((
            object_ids, np.eye(len(object_ids)), true_poses,
            False, True, 'zone',
            (object_points, [(zone_pose, zone_size)]), 1))
        self.lang_goals.append(self.lang_template.format(
            colors=relevant_desc,
        ))

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PackingBoxesPairsSeenColors(PackingBoxesPairsUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS


class PackingBoxesPairsFull(PackingBoxesPairsUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
        return all_colors