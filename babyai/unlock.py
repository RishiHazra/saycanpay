"""
Copied and adapted from https://github.com/mila-iqia/babyai.
Levels described in the Baby AI ICLR 2019 submission, with the `Unlock` instruction.
"""
from __future__ import annotations

import random
import sys
from enum import IntEnum
import numpy as np
from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Ball, Box, Key
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, OpenInstr, PickupInstr


class CustomActions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    # Pick up an object
    pickup = 3
    # Drop an object
    drop = 4
    # Toggle/activate an object
    toggle = 5
    # Done completing task
    done = 6
    # Goto
    goto = 7


class CustomRoomGridLevel(RoomGridLevel):
    """
       custom babyAI pickup calling multiple classes
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.actions = CustomActions
        self.action_space = spaces.Discrete(len(self.actions))

    @staticmethod
    def in_vicinity(curr_pos, goal_pos):
        del_x_list = [0, 0, 1, -1]
        del_y_list = [1, -1, 0, 0]
        for del_x, del_y in zip(del_x_list, del_y_list):
            new_x = curr_pos[0] + del_x
            new_y = curr_pos[1] + del_y
            if (new_x, new_y) == goal_pos:
                return True
        return False

    @staticmethod
    def next_cells(curr_pos, grid, visited):
        width, height = grid.shape
        del_x_list = [0, 0, 1, -1]
        del_y_list = [1, -1, 0, 0]
        for del_x, del_y in zip(del_x_list, del_y_list):
            new_x = curr_pos[0] + del_x
            new_y = curr_pos[1] + del_y
            if 0 <= new_x < width and 0 <= new_y < height and not visited[new_x][new_y]:
                yield new_x, new_y

    @staticmethod
    def BFS(curr_pos, goal_pos, grid):
        """
        shortest path
        """
        width, height = grid.shape
        queue = [curr_pos]
        parent = {curr_pos: None}
        visited = [[False for _ in range(height)] for _ in range(width)]
        visited[curr_pos[0]][curr_pos[1]] = True
        while len(queue) > 0:
            pos = queue.pop(0)
            for next_x, next_y in CustomRoomGridLevel.next_cells(pos, grid, visited):
                if grid[next_x][next_y] == 1:  # free space
                    queue.append((next_x, next_y))
                    parent[(next_x, next_y)] = pos
                    visited[next_x][next_y] = True
                elif grid[next_x][next_y] != 1 and (next_x, next_y) == goal_pos:  # goal is not empty
                    parent[(next_x, next_y)] = pos
                    visited[next_x][next_y] = True

        ret_path = []
        if not visited[goal_pos[0]][goal_pos[1]]:
            return None
        # reconstruct the path
        pos = parent[goal_pos]
        while pos is not None:
            ret_path.append(pos)
            pos = parent[pos]
        return ret_path[::-1][1:]

    @staticmethod
    def navigate(curr_pos, curr_dir, path):
        """
        plan agent actions to navigate from curr_pos following path
        dir >: 0, v: 1, <: 2, ^: 3
        """
        actions = []
        rel_dir_map = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
        cur_pos = curr_pos
        cur_dir = curr_dir
        for path_cell in path:
            if cur_pos == path_cell:
                break
            del_x, del_y = path_cell[0] - cur_pos[0], path_cell[1] - cur_pos[1]
            rel_dir = rel_dir_map[(del_x, del_y)]
            # if cur_dir == rel_dir:
            if cur_dir - rel_dir in [-1, 3]:  # rotate clockwise/right
                actions.append(1)
            elif cur_dir - rel_dir in [1, -3]:  # rotate anticlockwise/left
                actions.append(0)
            elif abs(cur_dir - rel_dir) == 2:
                [actions.append(1) for _ in range(2)]
            cur_dir = rel_dir
            actions.append(2)
            cur_pos = path_cell
        return actions[:-1]

    @staticmethod
    def get_object_pos(grid, target_object, agent_pos, fwd_pos):
        """
        for goto action
        """
        if target_object is None:  # empty cell
            # find an empty position in the same room as agent
            def bfs(grid, start, visited):
                queue = [start]
                room = set()
                while queue:
                    (x, y, o) = queue.pop(0)
                    if o is None or o.type not in ['wall', 'door']:
                        visited.add((x, y))
                        room.add((x, y))
                        if x > 0 and (x - 1, y) not in visited:
                            queue.append((x - 1, y, grid.get(x - 1, y)))
                        if x < grid.width - 1 and (x + 1, y) not in visited:
                            queue.append((x + 1, y, grid.get(x + 1, y)))
                        if y > 0 and (x, y - 1) not in visited:
                            queue.append((x, y - 1, grid.get(x, y - 1)))
                        if y < grid.height - 1 and (x, y + 1) not in visited:
                            queue.append((x, y + 1, grid.get(x, y + 1)))
                return room
            def get_rooms(grid):
                rooms = []  # list of tuples, each tuple is (x, y)
                visited = set()
                for i in range(grid.width):
                    for j in range(grid.height):
                        cell = grid.get(i, j)
                        if cell is None or cell.type not in ['wall', 'door']:
                            if (i, j) not in visited:
                                room = bfs(grid, (i, j, cell), visited)
                                rooms.append(room)
                return rooms
            # find all rooms
            rooms = get_rooms(grid)  # list of tuples (x, y)
            # find agent_room
            for room in rooms:
                if agent_pos in room:
                    agent_room = room
                    break
            # retrieve an empty pos in agent_room
            for i, j in agent_room:
                cell = grid.get(i, j)
                # empty cell which is not adjacent to the cell the object was picked from
                if cell is None and np.all([i, j] != fwd_pos):
                    # if cell is not right next to a door:
                    door_flag = False
                    for i_del, j_del in zip([0, 0, 1, -1], [1, -1, 0, 0]):
                        i_next, j_next = i + i_del, j + j_del
                        next_cell = grid.get(i_next, j_next)
                        if next_cell is not None and next_cell.type == 'door':
                            door_flag = True
                    if not door_flag:
                        return (i, j)

        # if target_object is not None (not an empty cell)
        target_color, target_type = target_object.color, target_object.type
        # Iterate over the grid
        for i in range(grid.width):
            for j in range(grid.height):
                # Get the object at the current position
                cell = grid.get(i, j)
                # If the cell is not None and the object matches the target color and type
                if cell and cell.color == target_color and cell.type == target_type:
                    return (i, j)
        return None  # Return None if no matching object is found

    def step(self, action):
        action, target_object = action
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # change in state after taking actions: goto, pickup, drop, toggle
        state_change = ''

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Goto an object
        elif action == self.actions.goto:
            width, height = self.grid.width, self.grid.height
            obj_pos = \
                    CustomRoomGridLevel.get_object_pos(self.grid,
                                                       target_object,
                                                       self.agent_pos,
                                                       self.front_pos)
            if obj_pos is not None:
                grid = \
                    np.array([1 if o is None or (o.type == 'door' and o.is_open) else 0 for o in self.grid.grid]).reshape(self.grid.height, self.grid.width).T
                visited = [[False for _ in range(height)] for _ in range(width)]
                ret_path = \
                    CustomRoomGridLevel.BFS(curr_pos=self.agent_pos,
                                            goal_pos=obj_pos, grid=grid)
                ret_path.append(obj_pos)
                try:
                    actions = CustomRoomGridLevel.navigate(self.agent_pos, self.agent_dir, ret_path)
                    while len(actions) != 0:
                        step_action = actions.pop(0)
                        _, reward, terminated, _, _ = self.step((step_action, None))
                    if target_object is None:  # go to empty cell
                        state_change = f'at empty cell'
                    else:
                        state_change = f'at {target_object.color} {target_object.type}'
                except KeyError:  # cannot navigate (agent and obj in different rooms)
                    pass

        # Pick up an object
        elif action == self.actions.pickup:
            self.step((7, target_object))
            # Get the position in front of the agent
            fwd_pos = self.front_pos
            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)
                    state_change = f'picked {fwd_cell.color} {fwd_cell.type}'

        # Drop an object
        elif action == self.actions.drop:
            self.step((7, None))
            # Get the position in front of the agent
            fwd_pos = self.front_pos
            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                state_change = f'dropped {self.carrying.color} {self.carrying.type}'
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            self.step((7, target_object))
            # Get the position in front of the agent
            fwd_pos = self.front_pos
            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)
                state_change = f'{fwd_cell.color} {fwd_cell.type} unlocked'

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        if self.carrying is not None:
            if self.carrying == self.goal_obj:
                terminated = True
                reward = 1.

        return obs, reward, terminated, truncated, {'state_change': state_change}

    def oracle(self, grid):
        """
        UnlockPickup Unlock a door, then pick up a box in another room
        	- goto the key
            - pickup key
            - goto the door
            - toggle door
            - drop the key
            - goto the box
            - pickup box

        BlockedUnlockPickup: Unlock a door blocked by a ball, then pick up a box in another room
	        - goto the ball
            - pickup ball
            - goto an empty space
            - drop ball
            - goto the key
            - pickup key
            - goto the door
            - toggle door
            - drop the key
            - goto the box
            - pickup box

        UnlockToUnlock: Unlock a door A that requires to unlock a door B before
            - goto the key in the room
            - pickup key in the room
            - goto the door with the matching key
            - toggle door
            - drop the key
            - goto the key in the next room
            - pickup key
            - goto door with matching key
            - toggle door
            - drop the key
            - goto the ball
            - pickup ball
        """

        def bfs(grid, start, visited):
            queue = [start]
            room = set()
            while queue:
                (x, y, o) = queue.pop(0)
                if o.type not in ['wall', 'door']:
                    visited.add((x, y))
                    room.add((x, y))
                    if x > 0 and (x - 1, y) not in visited:
                        queue.append((x - 1, y, grid[x-1, y, 2]))
                    if x < len(grid) - 1 and (x + 1, y) not in visited:
                        queue.append((x + 1, y, grid[x+1, y, 2]))
                    if y > 0 and (x, y - 1) not in visited:
                        queue.append((x, y - 1, grid[x, y-1, 2]))
                    if y < len(grid[0]) - 1 and (x, y + 1) not in visited:
                        queue.append((x, y + 1, grid[x, y+1, 2]))
            return room

        def get_rooms(grid):
            rooms = []  # list of tuples, each tuple is (x, y)
            visited = set()
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    if grid[i, j, 2].type not in ['wall', 'door'] and (i, j) not in visited:
                        room = bfs(grid, (i, j, grid[i, j, 2]), visited)
                        rooms.append(room)
                        # visited.update(room)
                    # elif grid[i][j] == 'door':
                    #     doors.append((i, j))
            return rooms

        def get_object_of_type(obj_type, grid):
            objects = []  # list of tuples, each tuple is (x, y, object of type obj_type)
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    obj = grid[i, j, 2]
                    if obj.type == obj_type:
                        objects.append((i, j, obj))
            return objects

        def get_adjoining_rooms(x, y, rooms):
            """
            get adjoining room ids given door pos (x, y)
            """
            adjoining_room_ids = []
            for del_x, del_y in zip([0, 1, -1, 0], [1, 0, 0, -1]):
                new_x, new_y = x + del_x, y + del_y
                for ind, room in enumerate(rooms):
                    room = [(x, y) for (x, y) in room]
                    if (new_x, new_y) in room:
                        adjoining_room_ids.append(ind)
                        if len(adjoining_room_ids) == 2:
                            return adjoining_room_ids
                        break

        actions = {"actions": [], "lang_actions": []}
        task = self.__class__.__name__
        stochasticity = 0.8

        # Unlock Pickup
        if task == 'UnlockPickup':
            rooms = get_rooms(grid)  # get rooms
            # key_pos = locate key in the room where the agent is
            for room in rooms:
                if self.agent_pos in room:
                    agent_room = room
                    break
            keys = get_object_of_type('key', grid)
            keys = [(x, y, o) for (x, y, o) in keys if (x, y) in agent_room]
            if len(keys) == 1:
                key_pos = keys[0][:2]
                key_obj = keys[0][-1]
            else:
                raise Exception("How are there > 1 keys for Unlock Pickup?")
            # actions["actions"].append((7, key_obj))  # goto key
            # actions["lang_actions"].append(f"go to the {key_obj.color} key")
            actions["actions"].append((3, key_obj))  # pickup key
            actions["lang_actions"].append(f"pick up {key_obj.color} key")

            # if random.random() <= stochasticity:
            #     empty_obj = None  # dummy empty obj
            #     # actions["actions"].append((7, empty_obj))  # goto the empty pos
            #     # actions["lang_actions"].append("go to an empty cell")
            #     actions["actions"].append((4, None))  # drop the ball
            #     actions["lang_actions"].append(f"drop key in void")
            #     rooms = get_rooms(grid)  # get rooms
            #     # key_pos = locate key in the room where the agent is
            #     for room in rooms:
            #         if self.agent_pos in room:
            #             agent_room = room
            #             break
            #     keys = get_object_of_type('key', grid)
            #     keys = [(x, y, o) for (x, y, o) in keys if (x, y) in agent_room]
            #     if len(keys) == 1:
            #         key_pos = keys[0][:2]
            #         key_obj = keys[0][-1]
            #     else:
            #         raise Exception("How are there > 1 keys for Unlock Pickup?")
            #     # actions["actions"].append((7, key_obj))  # goto key
            #     # actions["lang_actions"].append(f"go to the {key_obj.color} key")
            #     actions["actions"].append((3, key_obj))  # pickup key
            #     actions["lang_actions"].append(f"pick up {key_obj.color} key")

            # door_pos = locate door pos
            doors = get_object_of_type('door', grid)
            if len(doors) == 1:
                door_pos = doors[0][:2]
                door_obj = doors[0][-1]
            else:
                raise Exception("How are there > 1 doors for Unlock Pickup?")
            # actions["actions"].append((7, door_obj))  # goto door
            # actions["lang_actions"].append(f"go to the {door_obj.color} door")
            actions["actions"].append((5, door_obj))  # toggle door
            actions["lang_actions"].append(f"toggle {door_obj.color} door")
            # drop key in empty pos before picking box
            empty_obj = None  # dummy empty obj
            # actions["actions"].append((7, empty_obj))  # goto the empty pos
            # actions["lang_actions"].append("go to an empty cell")
            actions["actions"].append((4, None))  # drop the ball
            actions["lang_actions"].append(f"drop key in void")
            # box_pos = locate box in the other room
            boxes = get_object_of_type('box', grid)
            if len(boxes) == 1:
                box_pos = boxes[0][:2]
                box_obj = boxes[0][-1]
            else:
                raise Exception("How are there > 1 boxes for Unlock Pickup?")
            # actions["actions"].append((7, box_obj))  # goto box
            # actions["lang_actions"].append(f"go to the {box_obj.color} box")
            actions["actions"].append((3, box_obj))  # pickup box
            actions["lang_actions"].append(f"pick up {box_obj.color} box")

        # Blocked Unlock Pickup
        elif task == 'BlockedUnlockPickup':
            rooms = get_rooms(grid)  # get_rooms()
            # ball_pos = locate ball in the room which is blocking the door
            balls = get_object_of_type('ball', grid)
            if len(balls) == 1:
                ball_pos = balls[0][:2]
                ball_obj = balls[0][-1]
            else:
                raise Exception("How are there > 1 balls for Blocked Unlock Pickup?")
            doors = get_object_of_type('door', grid)
            if len(doors) == 1:
                door_pos = doors[0][:2]
            else:
                raise Exception("How are there > 1 doors for Blocked Unlock Pickup?")
            ball_blocks_door = False  # is ball_pos the correct one (lies next to the door?)
            for x, y in zip([1, -1, 0, 0], [0, 0, 1, -1]):
                new_pos = (door_pos[0] + x, door_pos[1] + y)
                if ball_pos == new_pos:
                    ball_blocks_door = True
                    break
            if not ball_blocks_door:
                raise Exception("Ball must be blocking the door.")
            # actions["actions"].append((7, ball_obj))  # goto ball
            # actions["lang_actions"].append(f"go to the {ball_obj.color} ball")
            if random.random() > stochasticity:  # adding some stochasticity (30 %)
                actions["actions"].append((3, ball_obj))  # pickup ball
                actions["lang_actions"].append(f"pick up {ball_obj.color} ball")
                empty_obj = None  # dummy empty obj
                # actions["actions"].append((7, empty_obj))  # goto the empty pos
                # actions["lang_actions"].append("go to an empty cell")
                actions["actions"].append((4, None))  # drop the ball
                actions["lang_actions"].append(f"drop ball in void")
                # key_pos = locate key in the room where the agent is
                for room_id, room in enumerate(rooms):
                    if self.agent_pos in room:
                        agent_room = room
                        agent_room_id = room_id
                        break
                keys = [(x, y, grid[x, y, 2]) for (x, y) in agent_room if grid[x, y, 2].type == 'key']
                if len(keys) == 1:
                    key_pos = keys[0][:2]
                    key_obj = keys[0][-1]
                else:
                    raise Exception("How are there > 1 keys for Blocked Unlock Pickup?")
                # actions["actions"].append((7, key_obj))
                # actions["lang_actions"].append(f"go to the {key_obj.color} key")
                actions["actions"].append((3, key_obj))  # pickup key
                actions["lang_actions"].append(f"pick up {key_obj.color} key")
            else:
                # key_pos = locate key in the room where the agent is
                for room_id, room in enumerate(rooms):
                    if self.agent_pos in room:
                        agent_room = room
                        agent_room_id = room_id
                        break
                keys = [(x, y, grid[x, y, 2]) for (x, y) in agent_room if grid[x, y, 2].type == 'key']
                if len(keys) == 1:
                    key_pos = keys[0][:2]
                    key_obj = keys[0][-1]
                else:
                    raise Exception("How are there > 1 keys for Blocked Unlock Pickup?")
                actions["actions"].append((3, key_obj))  # pickup key
                actions["lang_actions"].append(f"pick up {key_obj.color} key")
                empty_obj = None  # dummy empty obj
                # actions["actions"].append((7, empty_obj))  # goto the empty pos
                # actions["lang_actions"].append("go to an empty cell")
                actions["actions"].append((4, None))  # drop the ball
                actions["lang_actions"].append(f"drop key in void")
                actions["actions"].append((3, ball_obj))  # pickup ball
                actions["lang_actions"].append(f"pick up {ball_obj.color} ball")
                empty_obj = None  # dummy empty obj
                # actions["actions"].append((7, empty_obj))  # goto the empty pos
                # actions["lang_actions"].append("go to an empty cell")
                actions["actions"].append((4, None))  # drop the ball
                actions["lang_actions"].append(f"drop ball in void")
                # key_pos = locate key in the room where the agent is
                for room_id, room in enumerate(rooms):
                    if self.agent_pos in room:
                        agent_room = room
                        agent_room_id = room_id
                        break
                keys = [(x, y, grid[x, y, 2]) for (x, y) in agent_room if grid[x, y, 2].type == 'key']
                if len(keys) == 1:
                    key_pos = keys[0][:2]
                    key_obj = keys[0][-1]
                else:
                    raise Exception("How are there > 1 keys for Blocked Unlock Pickup?")
                actions["actions"].append((3, key_obj))  # pickup key
                actions["lang_actions"].append(f"pick up {key_obj.color} key")

            # door_pos = door_pos of the room
            doors = get_object_of_type('door', grid)
            if len(doors) == 1:
                door_pos = doors[0][:2]
                door_obj = doors[0][-1]
            else:
                raise Exception("How are there > 1 doors for Unlock Pickup?")
            # actions["actions"].append((7, door_obj))  # goto the door
            # actions["lang_actions"].append(f"go to the {door_obj.color} door")
            actions["actions"].append((5, door_obj))  # toggle door
            actions["lang_actions"].append(f"toggle {door_obj.color} door")
            # drop key in empty pos before picking box
            empty_obj = None  # dummy empty obj
            # actions["actions"].append((7, empty_obj))  # goto the empty pos
            # actions["lang_actions"].append("go to an empty cell")
            actions["actions"].append((4, None))  # drop the ball
            actions["lang_actions"].append(f"drop key in void")
            # box_pos = locate box in the other room
            boxes = get_object_of_type('box', grid)
            if len(boxes) == 1:
                box_pos = boxes[0][:2]
                box_obj = boxes[0][-1]
            else:
                raise Exception("How are there > 1 boxes for Unlock Pickup?")
            # actions["actions"].append((7, box_obj))  # goto the box
            # actions["lang_actions"].append(f"go to the {box_obj.color} box")
            actions["actions"].append((3, box_obj))  # pickup box
            actions["lang_actions"].append(f"pick up {box_obj.color} box")

        # Unlock To Unlock
        elif task == 'UnlockToUnlock':
            rooms = get_rooms(grid)
            # key_pos = locate the key in the same room as the agent
            for room_id, room in enumerate(rooms):
                if self.agent_pos in room:
                    agent_room = room
                    agent_room_id = room_id
                    break
            keys = get_object_of_type('key', grid)
            keys = [(x, y, o) for (x, y, o) in keys if (x, y) in agent_room]
            if len(keys) == 1:
                key_pos = keys[0][:2]
                key_obj = keys[0][-1]
            else:
                raise Exception("How are there > 1 keys for Unlock To Unlock Pickup?")
            # actions["actions"].append((7, key_obj))  # goto the key
            # actions["lang_actions"].append(f"go to the {key_obj.color} key")
            actions["actions"].append((3, key_obj))  # pickup key
            actions["lang_actions"].append(f"pick up {key_obj.color} key")
            # door_pos = locate door with the same color as the key
            doors = get_object_of_type('door', grid)
            for x, y, door_obj in doors:
                if door_obj.color == key_obj.color:
                    door_pos = (x, y)
                    break
            # actions["actions"].append((7, door_obj))  # goto door
            # actions["lang_actions"].append(f"go to the {door_obj.color} door")
            actions["actions"].append((5, door_obj))  # toogle door
            actions["lang_actions"].append(f"toggle {door_obj.color} door")
            # drop key in empty pos before picking box
            empty_obj = None  # dummy empty obj
            # actions["actions"].append((7, empty_obj))  # goto the empty pos
            # actions["lang_actions"].append("go to an empty cell")
            actions["actions"].append((4, None))  # drop the ball
            actions["lang_actions"].append(f"drop key in void")
            # key_pos = locate the key in the adjoining room
            adjoining_room_ids = get_adjoining_rooms(door_pos[0], door_pos[1], rooms)
            adjoining_room_id = [id for id in adjoining_room_ids if id != agent_room_id][0]
            adj_room = rooms[adjoining_room_id]
            keys = get_object_of_type('key', grid)
            keys = [(x, y, o) for (x, y, o) in keys if (x, y) in adj_room]
            if len(keys) == 1:
                key_pos = keys[0][:2]
                key_obj = keys[0][-1]
            else:
                raise Exception("How are there > 1 keys for Unlock To Unlock Pickup?")
            # actions["actions"].append((7, key_obj))  # goto the key
            # actions["lang_actions"].append(f"go to the {key_obj.color} key")
            actions["actions"].append((3, key_obj))  # pickup key
            actions["lang_actions"].append(f"pick up {key_obj.color} key")
            # door_pos = locate the door to the room which is the same color as the key
            for x, y, door_obj in doors:
                if door_obj.color == key_obj.color:
                    door_pos = (x, y)
                    assert not door_obj.is_open, "Not the correct door"
                    break
            # actions["actions"].append((7, door_obj))  # goto the door
            # actions["lang_actions"].append(f"go to the {door_obj.color} door")
            actions["actions"].append((5, door_obj))  # toggle the door
            actions["lang_actions"].append(f"toggle {door_obj.color} door")
            # drop key in empty pos before picking box
            empty_obj = None  # dummy empty obj
            # actions["actions"].append((7, empty_obj))  # goto the empty pos
            # actions["lang_actions"].append("go to an empty cell")
            actions["actions"].append((4, None))  # drop the ball
            actions["lang_actions"].append(f"drop key in void")
            # ball_pos = locate the ball
            balls = get_object_of_type('ball', grid)
            if len(balls) == 1:
                ball_pos = balls[0][:2]
                ball_obj = balls[0][-1]
            else:
                raise Exception("How are there > 1 balls for Unlock To Unlock Pickup?")
            # actions["actions"].append((7, ball_obj))  # goto the ball
            # actions["lang_actions"].append(f"go to the {ball_obj.color} ball")
            actions["actions"].append((3, ball_obj))  # pickup ball
            actions["lang_actions"].append(f"pick up {ball_obj.color} ball")

        else:
            raise NotImplementedError

        return actions


class Unlock(RoomGridLevel):
    """

    ## Description

    Unlock a door.

    Competencies: Maze, Open, Unlock. No unblocking.

    ## Mission Space

    "open the {color} door"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent opens the correct door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-Unlock-v0`

    """

    def gen_mission(self):
        # Add a locked door to a random room
        id = self._rand_int(0, self.num_cols)
        jd = self._rand_int(0, self.num_rows)
        door, pos = self.add_door(id, jd, locked=True)
        locked_room = self.get_room(id, jd)

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_cols)
            jk = self._rand_int(0, self.num_rows)
            if ik is id and jk is jd:
                continue
            self.add_object(ik, jk, "key", door.color)
            break

        # With 50% probability, ensure that the locked door is the only
        # door of that color
        if self._rand_bool():
            colors = list(filter(lambda c: c is not door.color, COLOR_NAMES))
            self.connect_all(door_colors=colors)
        else:
            self.connect_all()

        # Add distractors to all but the locked room.
        # We do this to speed up the reachability test,
        # which otherwise will reject all levels with
        # objects in the locked room.
        for i in range(self.num_cols):
            for j in range(self.num_rows):
                if i is not id or j is not jd:
                    self.add_distractors(i, j, num_distractors=3, all_unique=False)

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is locked_room:
                continue
            break

        self.check_objs_reachable()

        self.instrs = OpenInstr(ObjDesc(door.type, door.color))


class UnlockLocal(RoomGridLevel):
    """

    ## Description

    Fetch a key and unlock a door
    (in the current room)

    ## Mission Space

    "open the door"

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent opens the door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-UnlockLocal-v0`
    - `BabyAI-UnlockLocalDist-v0`

    """

    def __init__(self, distractors=False, **kwargs):
        self.distractors = distractors
        super().__init__(**kwargs)

    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)
        self.add_object(1, 1, "key", door.color)
        if self.distractors:
            self.add_distractors(1, 1, num_distractors=3)
        self.place_agent(1, 1)

        self.instrs = OpenInstr(ObjDesc(door.type))


class KeyInBox(RoomGridLevel):
    """

    ## Description

    Unlock a door. Key is in a box (in the current room).

    ## Mission Space

    "open the door"

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent opens the door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-KeyInBox-v0`

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)

        # Put the key in the box, then place the box in the room
        key = Key(door.color)
        box = Box(self._rand_color(), key)
        self.place_in_room(1, 1, box)

        self.place_agent(1, 1)

        self.instrs = OpenInstr(ObjDesc(door.type))


class UnlockPickup(CustomRoomGridLevel):
    """

    ## Description

    Unlock a door, then pick up a box in another room

    ## Mission Space

    "pick up the {color} box"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent picks up the correct box.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-UnlockPickup-v0`
    - `BabyAI-UnlockPickupDist-v0`

    """

    def __init__(self, distractors=False, max_steps: int | None = None, **kwargs):
        self.distractors = distractors
        room_size = 6
        if max is None:
            max_steps = 8 * room_size**2

        super().__init__(
            num_rows=1, num_cols=2, room_size=6, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):
        # Add a random object to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)
        if self.distractors:
            self.add_distractors(num_distractors=4)

        self.place_agent(0, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class BlockedUnlockPickup(CustomRoomGridLevel):
    """

    ## Description

    Unlock a door blocked by a ball, then pick up a box
    in another room

    ## Mission Space

    "pick up the box"

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent picks up the box.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-BlockedUnlockPickup-v0`

    """

    def __init__(self, max_steps: int | None = None, **kwargs):
        room_size = 6
        if max_steps is None:
            max_steps = 16 * room_size**2

        super().__init__(
            num_rows=1, num_cols=2, room_size=room_size, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):
        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, pos = self.add_door(0, 0, 0, locked=True)
        # Block the door with a ball
        color = self._rand_color()
        self.grid.set(pos[0] - 1, pos[1], Ball(color))
        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type))


class UnlockToUnlock(CustomRoomGridLevel):
    """

    ## Description

    Unlock a door A that requires to unlock a door B before

    ## Mission Space

    "pick up the ball"

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent picks up the ball.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-UnlockToUnlock-v0`

    """

    def __init__(self, max_steps: int | None = None, **kwargs):
        room_size = 6
        if max_steps is None:
            max_steps = 30 * room_size**2

        super().__init__(
            num_rows=1, num_cols=3, room_size=room_size, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, 2)

        # Add a door of color A connecting left and middle room
        self.add_door(0, 0, door_idx=0, color=colors[0], locked=True)

        # Add a key of color A in the room on the right
        self.add_object(2, 0, kind="key", color=colors[0])

        # Add a door of color B connecting middle and right room
        self.add_door(1, 0, door_idx=0, color=colors[1], locked=True)

        # Add a key of color B in the middle room
        self.add_object(1, 0, kind="key", color=colors[1])

        obj, _ = self.add_object(0, 0, kind="ball")

        self.place_agent(1, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type))