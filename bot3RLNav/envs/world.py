import gym
from gym import spaces
import numpy as np

import cv2


########################################################################
class World(gym.Env):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, map_file: str):
        """"""
        self.map = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
        e = self.map.shape
        x, y = e

        self.center = tuple(np.ceil([(e[1] + 2) / 2, (e[0] / 2) + 64]).astype('int'))

        self.agent_radius = 10
        self.agent_thickness = -1
        self.agent_color = (0, 0, 255)

        self.movable_radius = self.center[0] - 75
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
        self.ts = 1
        self.tolerance = 3

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

        :param seed:
        :param options:
        """
        super(World, self).reset(seed=seed)
        return_info = kwargs.get("return_info", False)

        while True:
            x, y = self.get_coordinates()
            if self.valid_pose(x, y):
                break
        theta = self.np_random.uniform(-np.pi, np.pi, size=1)
        self._agent_location = np.array([x, y, theta])

        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            x, y = self.get_coordinates()
            if self.valid_pose(x, y):
                self._target_location = np.array([x[0], y[0]])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
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

        info = self._get_info()

        distance = info["distance"][0]

        reward = (1 / (1 + distance)) if self.valid_pose(int(x), int(y)) else -1.0

        observation = self._get_obs()

        done = bool((distance <= self.tolerance) or (reward < 0))

        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, done, info

    def render(self, mode="human"):
        pass

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
        x, max_ = theta + np.pi, 2*np.pi
        return -np.pi + ((max_ + (x % max_)) % max_)


class World1(World):
    # ----------------------------------------------------------------------
    def __init__(self, map_file: str):
        """"""
        super().__init__(map_file)
        x, y = self.map.shape
        self.observation_space = spaces.Box(low=np.array([0, 0, -np.pi]), high=np.array([x, y, np.pi]), dtype=float)

    # ----------------------------------------------------------------------
    def _get_obs(self):
        """
        Return robot's x, y, theta coordinates according to the format of `self.observation_space`

        :return:
        """
        return self._agent_location.flatten()
