## Introduction
<img src="data/map.jpg" width="50%" alt="Image of World">

Given the world shown above, navigate robot within the gray areas, within the bounds of the green pillars and black 
walls. The white circles indicates obstacles. 

<img src="data/gray.jpg" width="50%" alt="Gray Image of World">

We converted the original map to a grayscale version using computer vision techniques. Now the white space indicates 
movable areas while the black circles within the wall indicates the obstacles. See the file `rl_map.ipynb` for details.

<img src="data/world_with_robot.png" width="50%" alt="Image of World with Robot">

We design our agent (_robot_) to have a certain diameter with a pose (x, y, theta). The agent is indicated by the red 
spot in the image above.

With this background information do we build our world having the following additional properties.
1. Our observation space given by _x_, _y_ and _theta_ indicating the robot's current pose and the target position specified by its _x_ and _y_ coordinates. 
2. _x_ and _y_ have a range of (0 to the maximum _x_ & _y_ coordinates of the world), while theta has a range of (`-pi` to `pi`)
3. The action space is given by the linear (`v`) and angular velocity (`w`) of the robot.
4. `v` and `w` are designed to be continous between _0_ to _1_ for `v` and _-1_ to _1_ for `w`.
5. The initial robot pose and the target pose are initialized randomly within the movable area, with only two constraints.
   1. The robot pose and the target pose should not be equivalent
   2. The robot pose and the target pose should not be on the walls or obstacles.
6. We calculate the reward signal as a function of the euclidean distance to the target. 
7. We set a threshold of 3units to acceptable target and robot location equivalence.
8. Given an action (_v, w_), new position and orientation are calculated using the integration approximation formula 
   ```
   x0, y0, t0 = robot_pose 
   x = x0 + v*cos(t0 + 0.5*w)
   y = y0 + v*sin(t0 + 0.5*w)
   t = t0 + w   //wrapped to values between -pi to pi
   ```

## Helpers
### Workspace Setup
Set up a `conda` environment. 

To activate terminal.
```
conda activate bot3RLNavigation
```

Install `gym` using the code below. See [here](https://anaconda.org/conda-forge/gym) for details.
```
conda install -c conda-forge gym
```

To create requirements file. See [More](https://linuxhint.com/conda-install-requirements-txt/)
```
conda list -e > requirements.txt
```

For `pip`, use
```
pip freeze > pip_reqs.txt
```

You can create an environment for work using this package's requirements via
```
conda create --name <env> --file requirements.txt
```

Install `opencv`. See [here](https://anaconda.org/conda-forge/opencv) for details.
```
conda install -c conda-forge opencv
```

Install `PILLOW` See [here](https://anaconda.org/conda-forge/pillow) for details
```
conda install -c conda-forge pillow
```

### Gym Environment Registration
In the `__init__.py` file in `envs` add an import statement to your created world.  
In the package's `__init__.py` supply the `id`, `entry_point` and other necessary options for your environment.  
After registration, our custom World environment can be created with 
```
env = gym.make('id')
```
See the mentioned `__init__` files for details.

### Simulation
To render simulation using `World-v2`, use the format below
```
env.reset()
name = "bot3"
cv2.namedWindow(name)
rate = 500  # frame rate in ms
while True:
    frame = env.render(mode="rgb_array")
    cv2.imshow("bot3", frame)
    cv2.waitKey(rate)
    action = env.action_space.sample() # random (or use policy)
    obs, reward, done, info step = env.step(action)
    if done:
      frame = env.render(mode="rgb_array")
      cv2.imshow("bot3", frame)
      cv2.waitKey(rate)
      break
```

### Map Version
Two map versions exist. The map shown in the preceeding sections (with obstalces), and another shown below, 
(without obstacles). The `env` can be created using the code fragment below.  
Has been tested on `World-v1` and should work with this version and above, with all it's subclasses.
```
env = gym.make('bot3RLNav/World-v2', map_file="data/map01.jpg",
             robot_file="data/robot.png")
```
<img src="data/map01.jpg" width="50%" alt="Image of World">

### Resetting
To set fixed targets use the code block below. This feature means, the robot initial pose and the target location are 
always reset to the same first poses that the preceeding `.reset()` call made.
```
env = gym.make(...)
env.reset(options={'reset': False})
# or env.reset(options=dict(reset=False))
```
Works on `v0` and all the subclasses.

## References
1. [Custom Environment](https://www.gymlibrary.dev/content/environment_creation/)
2. [Spaces](https://www.gymlibrary.dev/api/spaces/)
3. [Gym Github](https://github.com/openai/gym)
4. [Markdown](https://daringfireball.net/projects/markdown/)
5. [Create CV Window](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html#ga5afdf8410934fd099df85c75b2e0888b)
6. [CV imshow](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563)
