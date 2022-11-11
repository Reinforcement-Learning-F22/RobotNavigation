import gym
from gym import spaces
import numpy as np

import cv2

from PIL import Image


########################################################################
class World(gym.Env):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, map_file: str):
        """"""
        self.coloured_map = cv2.imread(map_file, cv2.IMREAD_COLOR)
        self.map = self.generate_gray_map(map_file)
        e = self.map.shape
        x, y = e

        self.center = tuple(np.ceil([(e[1] + 2) / 2, (e[0] / 2) + 64]).astype('int'))

        self.agent_radius = 10
        self.agent_thickness = -1
        self.agent_color = (0, 0, 255)

        self.movable_radius = self.center[0] - 90
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(0, x, shape=(1,), dtype=float),
                "y": spaces.Box(0, y, shape=(1,), dtype=float),
                "theta": spaces.Box(-np.pi, np.pi, shape=(1,), dtype=float),
                "target": spaces.Box(low=np.array([0, 0]), high=np.array([x, y]), dtype=int),
            }
        )
        self.action_space = spaces.Dict(
            {
                "v": spaces.Box(0, 1, shape=(1,), dtype=float),
                "w": spaces.Box(-1, 1, shape=(1,), dtype=float)
            }
        )

        self.render_mode = None

        self.window = None
        self.clock = None

        self._agent_location = np.array([0.0, 0.0, 0.0])
        self._target_location = np.array([0, 0])
        self.ts = 0.033
        self.tolerance = 4

        self.goal_not_set = True
        self.goals = []

    # ----------------------------------------------------------------------
    def _get_obs(self) -> dict:
        """
        Return robot's x, y, theta and target location according to the format of `self.observation_space`

        :return:
        """
        x, y, theta = self._agent_location
        return {"x": x, "y": y, "theta": theta, "target": self._target_location}

    # ----------------------------------------------------------------------
    def _get_info(self):
        """
        Return euclidean distance between robot current pose and target pose.

        :return:
        :rtype: np.ndarray
        """
        x, y, _ = self._agent_location
        x_, y_ = self._target_location
        distance = np.sqrt(((x - x_) ** 2) + ((y - y_) ** 2))
        return {"distance": distance}

    # ----------------------------------------------------------------------
    def _render_frame(self):
        """"""

    # ----------------------------------------------------------------------
    def reset(self, seed=None, options=None, **kwargs):
        """
        Randomly initialize the robot's pose and the target location until it does not concide with the
        current robot pose also set both positions within the movable region and not in an obstacle.
        In order to make the initial robot pose, and the target to be constant, pass reset=False, in options dict.
        i.e. options=dict(reset=False). Default is True.

        :param seed:
        :param options:
        """
        if options is None:
            options = {}
        super(World, self).reset(seed=seed)
        return_info = kwargs.get("return_info", False)
        reset = options.get("reset", True)

        if self.goal_not_set or reset:
            # if goals haven't been initially set
            # OR
            # if the flag to reset was passed as True i.e. `reset=true`
            while True:
                x, y = self.get_coordinates()
                if self.valid_pose(x, y):
                    break
            theta = self.np_random.uniform(-np.pi, np.pi, size=1)
            self._agent_location = np.array([x, y, theta])

            self._target_location = self._agent_location
            x, y, _ = self._agent_location
            d = 0
            while d < (2 * self.agent_radius) + 3:
                # while target location is still within robot radius. Number 3 is some padding
                x_, y_ = self.get_coordinates()
                if self.valid_pose(x_, y_):
                    d = np.sqrt(((x - x_) ** 2) + ((y - y_) ** 2))
                    self._target_location = np.array([x_[0], y_[0]])

            self.goal_not_set = False
            self.goals = [self._agent_location.copy(), self._target_location.copy()]
        else:
            self._agent_location = self.goals[0].copy()
            self._target_location = self.goals[1].copy()
        observation = self._get_obs()
        info = self._get_info()

        self.render(reset=True)
        if return_info:
            return observation, info
        return observation

    def get_coordinates(self):
        """
        Get random point (x, y) within movable region.
        Randomly take `x` along the diameter of the circle of the movable region,
        Take `y` based on how far `x` is from the centre of the circle.
        Particularly we use the distance of `x` from either ends of the circle,
        to calculate the radius within which `y` can be chosen.

        :return: x, y
        """
        x = self.np_random.integers(self.center[0] - self.movable_radius,
                                    self.center[0] + self.movable_radius, size=1, dtype=int)
        # is x on the left, middle or right of centre point?
        if x < self.center[0]:
            # left
            radius = (self.center[0] - self.movable_radius) + x
        elif x > self.center[0]:
            # right
            radius = (self.center[0] + self.movable_radius) - x
        else:
            # middle
            radius = self.movable_radius
        y = self.np_random.integers(self.center[1] - radius, self.center[1] + radius, size=1, dtype=int)
        return x, y

    # ----------------------------------------------------------------------
    def step(self, action: dict):
        """

        :param action:
        :return: observation, reward, done, info
        """
        v = action["v"]
        w = action["w"]
        x0, y0, t0 = self._agent_location
        x = x0 + (v * self.ts * np.cos(t0 + (0.5 * w * self.ts)))
        y = y0 + (v * self.ts * np.sin(t0 + (0.5 * w * self.ts)))
        theta = self.wrap_to_pi(t0 + (w * self.ts))

        self._agent_location = np.array([x, y, theta])

        done, info, observation, reward = self.get_data()
        return observation, reward, done, info

    def get_data(self):
        info = self._get_info()
        distance = info["distance"][0]
        reward = self.get_reward(distance)
        observation = self._get_obs()
        done = bool((distance <= self.tolerance) or (reward < 0))
        if self.render_mode == "human":
            self._render_frame()
        return done, info, observation, reward

    def get_reward(self, distance):
        x, y, theta = self._agent_location
        return (1 / (1 + distance)) if self.valid_pose(int(x), int(y)) else -10

    def render(self, mode="human", **kwargs):
        if self.render_mode == "rgb_array" or mode == "rgb_array":
            return self._render_frame()

    def _circle(self, x, y):
        """
        Circle robot. To be implemented later.
        Consider masking. https://colab.research.google.com/drive/1g0pGp1hBLzhpISA0fQcp5Od448a8itFj#scrollTo=nnwWkQEZVCrG

        :param x:
        :param y:
        :return:
        """

    # ----------------------------------------------------------------------
    def _square(self, x, y):
        """
        A square robot.

        :param x:
        :param y:
        :return:
        """
        x, y = int(x), int(y)
        x0 = np.ceil(x - self.agent_radius).astype("int")
        x1 = np.ceil(x + self.agent_radius).astype("int")
        y0 = np.ceil(y - self.agent_radius).astype("int")
        y1 = np.ceil(y + self.agent_radius).astype("int")
        return self.map[y0:y1, x0:x1]

    # ----------------------------------------------------------------------
    def valid_pose(self, x: int, y: int):
        """
        If all the pixels in region are white valid, else invalid.

        :param x:
        :param y:
        :return:
        """
        region = self._square(x, y)
        # normalize
        region = region / 255
        return np.sum(region) == len(region.flatten())

    # ----------------------------------------------------------------------
    @staticmethod
    def wrap_to_pi(theta: float):
        """"""
        x, max_ = theta + np.pi, 2 * np.pi
        return -np.pi + ((max_ + (x % max_)) % max_)

    def generate_gray_map(self, filename) -> np.ndarray:
        img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        thresh = self.binary_thresh(img_gray)
        _, thresh2 = cv2.threshold(thresh, 250, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
        return closing

    @staticmethod
    def binary_thresh(image, threshold=150, max_value=255):
        """
        param: image: image being processed
        param: threshold: threshold value
        param: max_value: value to set to pixels that are greater than threshold
        return binary image
        """
        img = image.copy()
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if threshold - 10 <= image[i][j] <= threshold + 10:
                    img[i][j] = max_value
                else:
                    img[i][j] = 0
        return img


class World1(World):
    # ----------------------------------------------------------------------
    def __init__(self, map_file: str):
        """"""
        super().__init__(map_file)
        x, y = self.map.shape
        self.observation_space = spaces.Box(low=np.array([0, 0, -np.pi]), high=np.array([x, y, np.pi]), dtype=float)
        self.action_space = spaces.Box(low=np.array([-10, -0.2]), high=np.array([10, 0.2]), dtype=float)

    # ----------------------------------------------------------------------
    def _get_obs(self):
        """
        Return robot's x, y, theta coordinates according to the format of `self.observation_space`

        :return:
        """
        return self._agent_location.flatten()

    # ----------------------------------------------------------------------
    def step(self, action):
        """

        :param action:
        :return: observation, reward, done, info
        """
        v = action[0]
        w = action[1]
        x0, y0, t0 = self._agent_location
        x = x0 + (v * self.ts * np.cos(t0 + (0.5 * w * self.ts)))
        y = y0 + (v * self.ts * np.sin(t0 + (0.5 * w * self.ts)))
        theta = self.wrap_to_pi(t0 + (w * self.ts))

        self._agent_location = np.array([x, y, theta])

        done, info, observation, reward = self.get_data()

        return observation, reward, done, info

    def get_coordinates(self):
        """
        Get random point (x, y) within movable region.
        Randomly take `x` along the horizontal diameter of the circle of the movable region,
        Take `y` such that (y^2) < (r^2) - (x^2)
        Given that x^2 + y^2 < r^2

        :return: x, y
        """
        while True:
            try:
                x = self.np_random.integers(self.center[0] - self.movable_radius,
                                            self.center[0] + self.movable_radius, size=1, dtype=int)
                # distance from x to centre of circle
                x_ = np.abs(self.center[0] - x)
                # constraint d < sqrt(r**2 - x_**2)
                d = int(np.ceil(np.sqrt((self.movable_radius ** 2) - (x_[0] ** 2))))
                radius = d
                y = self.np_random.integers(self.center[1] - radius, self.center[1] + radius, size=1, dtype=int)
                return x, y
            except ValueError:
                pass


class DiscreteWorld(World1):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, map_file: str):
        super().__init__(map_file)
        self.strActions = {
            0: 'Forward',
            1: 'Left Turn',
            2: 'Right Turn'
        }

        self.actionVel = {
            'Forward': [0.8, 0.0, 0.0],
            'Left Turn': [0.8, 0.0, 0.5],
            'Right Turn': [0.8, 0.0, -0.5]
        }

        self.action_space = spaces.Discrete(len(self.strActions))

    def step(self, action: int):
        """

        :param action: integer between 0 and 2
        :return: observation, reward, done, info
        """
        vel = self.actionVel[self.strActions[action]]
        vx = vel[0]
        vy = vel[1]
        w = self.wrap_to_pi(vel[2])
        x0, y0, t0 = self._agent_location
        x = x0 + (vx * self.ts * np.cos(t0 + (0.5 * w * self.ts)))
        y = y0 + (vy * self.ts * np.sin(t0 + (0.5 * w * self.ts)))
        theta = self.wrap_to_pi(t0 + (w * self.ts))

        self._agent_location = np.array([x, y, theta])

        done, info, observation, reward = self.get_data()

        return observation, reward, done, info


########################################################################
class World2(World1):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, map_file: str, robot_file: str, render_mode=None):
        """"""
        super().__init__(map_file)
        self.agent = self.create_agent(robot_file)
        rgba = cv2.cvtColor(self.coloured_map, cv2.COLOR_RGB2RGBA)
        self.frame = Image.fromarray(rgba)
        self.render_mode = render_mode
        self.target_radius = 5

    # ----------------------------------------------------------------------
    def create_agent(self, file):
        """"""
        im = Image.open(file)

        size = max(map(lambda i: int(i * 0.1), im.size))
        self.agent_radius = int(np.ceil(size / 2))

        return im

    # ----------------------------------------------------------------------
    def _render_frame(self):
        """"""
        x, y, theta = self._get_obs()
        deg = np.degrees(theta)

        # rotate agent by degrees
        # noinspection PyTypeChecker
        # allign robot image to match the motion
        im_ = self.agent.rotate(-deg - 90)
        # noinspection PyTypeChecker
        im_ = im_.resize(tuple(map(lambda i: int(i * 0.1), im_.size)))

        # superimpose agent on map
        image = self.merge(im_, (x, y))

        # noinspection PyTypeChecker
        return np.asarray(image)

    # ----------------------------------------------------------------------
    def merge(self, robot, pose):
        """"""
        im = self.frame.copy()
        # noinspection PyTypeChecker
        im.alpha_composite(robot, dest=tuple(map(int, pose)))

        return im

    def render(self, mode="human", **kwargs):
        """
        Set's the target position on the frame when called from a .reset method.

        :param mode:
        :param kwargs:
        :return:
        """
        reset = kwargs.get("reset", False)
        if reset:
            # draw target location on coloured map
            frame = cv2.cvtColor(self.coloured_map, cv2.COLOR_RGB2RGBA)
            img_ = cv2.circle(frame, self._target_location, radius=self.target_radius, color=(0, 0, 255), thickness=-1)
            self.frame = Image.fromarray(img_)
        return super().render(mode, **kwargs)


class DiscreteWorld1(World2):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, map_file: str, robot_file: str, render_mode=None):
        super().__init__(map_file, robot_file, render_mode)
        self.strActions = {
            0: 'Forward',
            1: 'Left Turn',
            2: 'Right Turn'
        }

        self.actionVel = {
            'Forward': [0.8, 0.0, 0.0],
            'Left Turn': [0.8, 0.0, 0.5],
            'Right Turn': [0.8, 0.0, -0.5]
        }

        self.action_space = spaces.Discrete(len(self.strActions))

    def step(self, action: int):
        """

        :param action: integer between 0 and 2
        :return: observation, reward, done, info
        """
        vel = self.actionVel[self.strActions[action]]
        vx = vel[0]
        vy = vel[1]
        w = self.wrap_to_pi(vel[2])
        x0, y0, t0 = self._agent_location
        x = x0 + (vx * self.ts * np.cos(t0 + (0.5 * w * self.ts)))
        y = y0 + (vx * self.ts * np.sin(t0 + (0.5 * w * self.ts)))
        theta = self.wrap_to_pi(t0 + (w * self.ts))

        self._agent_location = np.array([x, y, theta])

        done, info, observation, reward = self.get_data()

        return observation, reward, done, info


class World3(World2):
    # ----------------------------------------------------------------------
    def __init__(self, map_file: str, robot_file: str):
        """"""
        super().__init__(map_file, robot_file)

    def get_reward(self, distance):
        x, y, theta = self._agent_location
        xg, yg = self._target_location

        reward = super().get_reward(distance)
        bearing = np.arctan2(yg - y, xg - x)
        alpha = bearing - theta
        alpha /= np.pi
        reward -= alpha
        return reward[0]


########################################################################
class DiscreteWorld2(World3):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, map_file: str, robot_file: str):
        """
        Uses a dicrete action scheme.
        0, 1 ... 9, 10 for v; and
        -10, -9, ... 0, ..., 9, 10 for w.
        These values are clased down in the step method.

        :param map_file:
        :param robot_file:
        """
        super().__init__(map_file, robot_file)
        self.actions = {}
        count = 0
        for v in range(5):
            for w in range(-5, 5):
                self.actions[count] = [v, w]
                count += 1
        self.action_space = spaces.Discrete(len(self.actions))
        self.ts = 0.03

    # ----------------------------------------------------------------------
    def step(self, action: int):
        """
        Performs a forward step given control action. Control actions are scaled down by 0.1

        :param action: dict containing linear and angular velocity `v` & `w` respectively.
        :return:
        """
        # noinspection PyTypeChecker
        _action = self.actions[action]
        v = _action[0] * 2
        # noinspection PyTypeChecker
        w = _action[1] * 2

        x0, y0, t0 = self._agent_location
        x = x0 + (v * self.ts * np.cos(t0 + (0.5 * w * self.ts)))
        y = y0 + (v * self.ts * np.sin(t0 + (0.5 * w * self.ts)))
        theta = self.wrap_to_pi(t0 + (w * self.ts))

        self._agent_location = np.array([x, y, theta])

        done, info, observation, reward = self.get_data()
        return observation, reward, done, info
