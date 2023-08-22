from gymnasium.core import ObservationWrapper
# from minigrid.wrappers import FullyObsWrapper
# from minigrid.envs.babyai.pickup import Pickup
from minigrid.core.constants import OBJECT_TO_IDX
from gymnasium import spaces
import numpy as np
# from minigrid.envs.babyai.unlock import CustomRoomGridLevel


class EmptyObject:
    def __init__(self, cur_pos):
        self.type = 'empty'
        self.cur_pos = cur_pos
        self.color = None


class AgentObject:
    def __init__(self, cur_pos):
        self.type = 'agent'
        self.cur_pos = cur_pos


class LanguageObsWrapper(ObservationWrapper):
    """
    Fully observable grid with a language state representation.
    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import SymbolicObsWrapper
        >>> env = gym.make("BabyAI-GoToRedBlueBall-v0")
        >>> env_obs = LanguageObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (11, 11, 3)
    """

    def __init__(self, env):
        # env = FullyObsWrapper(env)
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=max(OBJECT_TO_IDX.values()),
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    @staticmethod
    def bfs(grid, start, visited):
        queue = [start]
        room = set()
        while queue:
            (x, y) = queue.pop(0)
            if grid[x][y].type not in ['wall', 'door']:
                visited.add((x, y))
                room.add((x, y))
                if x > 0 and (x - 1, y) not in visited:
                    queue.append((x - 1, y))
                if x < len(grid) - 1 and (x + 1, y) not in visited:
                    queue.append((x + 1, y))
                if y > 0 and (x, y - 1) not in visited:
                    queue.append((x, y - 1))
                if y < len(grid[0]) - 1 and (x, y + 1) not in visited:
                    queue.append((x, y + 1))
        return room

    @staticmethod
    def get_rooms(grid):
        rooms = []  # list of tuples, each tuple is (x, y, object)
        visited = set()
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j].type not in ['wall', 'door'] and (i, j) not in visited:
                    room = LanguageObsWrapper.bfs(grid, (i, j), visited)
                    rooms.append(room)
                    # visited.update(room)
                # elif grid[i][j] == 'door':
                #     doors.append((i, j))
        return rooms

    @staticmethod
    def get_object_of_type(obj_type, grid):
        objects = []  # list of tuples, each tuple is (x, y, object of type obj_type)
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                obj = grid[i][j]
                if obj.type == obj_type:
                    objects.append((i, j, obj))
        return objects

    @staticmethod
    def get_adjoining_rooms(x, y, rooms):
        """
        get adjoining rooms given door pos (x, y)
        """
        adjoining_room_ids = []
        for del_x, del_y in zip([0, 1, -1, 0], [1, 0, 0, -1]):
            new_x, new_y = x + del_x, y + del_y
            for ind, room in enumerate(rooms):
                if (new_x, new_y) in room:
                    adjoining_room_ids.append(ind + 1)
                    if len(adjoining_room_ids) == 2:
                        return adjoining_room_ids
                    break

    def observation(self, obs):
        ncol, nrow = self.width, self.height
        objects = np.array(
            [o if o is not None else -1 for o in self.grid.grid]
        )

        _objects = np.transpose(objects.reshape(1, nrow, ncol), (0, 2, 1))
        agent_pos = self.env.agent_pos
        _objects[0, agent_pos[0], agent_pos[1]] = AgentObject(cur_pos=agent_pos)
        # ncol, nrow = self.width, self.height
        grid = _objects

        # adding the empty object class
        for x in range(len(grid[0])):
            for y in range(len(grid[0][x])):
                if grid[0, x, y] == -1:
                    grid[0, x, y] = EmptyObject(cur_pos=(x, y))
                elif grid[0, x, y].type == obs['mission'].split()[-1]:
                    self.env.env.env.goal_obj = grid[0, x, y]

                    # retrieve the rooms positions using BFS
        # TODO: only once at the start
        rooms = LanguageObsWrapper.get_rooms(grid[0])
        # get language description of the grid
        descriptions = []
        for i, room in enumerate(rooms):
            objects = []
            for (x, y) in room:
                obj_in_cell = grid[0][x][y]
                if obj_in_cell.type not in ['empty', 'wall', 'door', 'agent']:
                    objects.append(f'{obj_in_cell.color} {obj_in_cell.type}')
                elif obj_in_cell.type in ['door', 'agent']:
                    objects.append(f'{obj_in_cell.type}')
                # TODO: key in box
            objects = ', '.join((objects))
            descriptions.append(f"Room {i + 1} has {objects}")

        doors = LanguageObsWrapper.get_object_of_type('door', grid[0])
        if len(doors) > 0:
            doors_status = []
            for door in doors:
                x, y, door_obj = door
                door_status = "open" if door_obj.is_open else "locked"
                # adjoining rooms for the door object
                adj_room_ids = LanguageObsWrapper.get_adjoining_rooms(x, y, rooms)
                doors_status.append(
                    f"The door connecting Room {min(adj_room_ids)} "
                    f"and Room {max(adj_room_ids)} is {door_status}")
            descriptions.extend(doors_status)

            if self.env.spec.name == 'BabyAI-BlockedUnlockPickup':
                try:
                    balls = LanguageObsWrapper.get_object_of_type('ball', grid[0])
                    descriptions.append(f'The {balls[0][2].color} ball is blocking the door')
                except IndexError:
                    pass  # ball has been picked up in the subsequent steps
        obs["language"] = '. '.join(descriptions) + '. '

        objects = np.array(
            [o if o is not None else -1 for o in self.grid.grid]
        )
        agent_pos = self.env.agent_pos
        ncol, nrow = self.width, self.height
        grid = np.mgrid[:ncol, :nrow]
        _objects = np.transpose(objects.reshape(1, nrow, ncol), (0, 2, 1))

        grid = np.concatenate([grid, _objects])
        grid = np.transpose(grid, (1, 2, 0))
        grid[agent_pos[0], agent_pos[1], 2] = AgentObject(cur_pos=agent_pos)
        # adding the empty object class
        for x in range(len(grid)):
            for y in range(len(grid[x])):
                if grid[x, y, 2] == -1:
                    grid[x, y, 2] = EmptyObject(cur_pos=(x, y))
        obs["image"] = grid

        # admissible actions
        admissible_actions = []
        for o in objects:
            if o != -1 and o.type != 'wall':
                if o.type == 'door':
                    name = f"{o.color} door"
                    # admissible_actions.append(f"go to the {name}")
                    # admissible_actions.append(f"toggle the door")
                    admissible_actions.append(f"toggle {name}")
                elif o.type == 'box':
                    name = f"{o.color} box"
                    # admissible_actions.append(f"go to the {name}")
                    admissible_actions.append(f"pick up {name}")
                    # admissible_actions.append(f"drop the box")
                    admissible_actions.append(f"drop box in void")
                elif o.type == 'ball':
                    name = f"{o.color} ball"
                    # admissible_actions.append(f"go to the {name}")
                    admissible_actions.append(f"pick up {name}")
                    # admissible_actions.append(f"drop the ball")
                    admissible_actions.append(f"drop ball in void")
                elif o.type == 'key':
                    name = f"{o.color} key"
                    # admissible_actions.append(f"go to the {name}")
                    admissible_actions.append(f"pick up {name}")
                    # admissible_actions.append(f"drop the key")
                    admissible_actions.append(f"drop key in void")
        # admissible_actions.append("go to an empty cell")
        admissible_actions.append("done picking up")
        obs["admissible_actions"] = list(set(admissible_actions))
        return obs
