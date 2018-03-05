import gym
from gym import spaces
from gym.envs.registration import EnvSpec
from gym.envs.registration import register

import numpy as np

class ParticleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, goal, reward_structure):
        self.mass = 1.0
        self.max_force = 3.0
        self.dt = 0.01 # seconds between state updates
        self.tol = 1e-1

        self.goal = goal # np array of size 2, in xy coordinates
        self.reward_structure = reward_structure # 'sparse' or 'dense'

        self.max_pos = 5 # assume square
        self.max_vel = 3
        self.max_states = np.array([self.max_pos, self.max_pos, self.max_vel, self.max_vel]) # x, y, vx, vz
        self.max_actions = np.array([self.max_force, self.max_force])

        self.action_space = spaces.Box(low=-self.max_actions, high=self.max_actions)
        self.observation_space = spaces.Box(low=-self.max_states, high=self.max_states)

        self.viewer = None

        self.seed()

    def seed(self, seed=None):
        return 0

    def reset(self):
        self.state = np.array([
            np.zeros(2),
            # np.random.uniform(low=-self.max_pos, high=self.max_pos, size=(2,)), 
            np.zeros(2) # assume starting with zero velocity
            ]).flatten()
        return np.array(self.state)

    def step(self, action):
        """
        action: np.array of size 2
        """

        pos = np.squeeze(self.state[:2])
        vel = np.squeeze(self.state[2:])

        clamp = lambda x, minx, maxx: np.minimum(np.maximum(minx, x), maxx)
        force = clamp(action, -self.max_actions, self.max_actions)

        vel += force * self.dt
        max_vels = np.array([self.max_vel, self.max_vel])
        vel = clamp(vel, -max_vels, max_vels)

        pos += vel * self.dt

        # If it hits a wall, set velocity to zero 
        if (np.max(pos) > self.max_pos) or (np.min(pos) < -self.max_pos):
            max_pos_arr = np.array([self.max_pos, self.max_pos])
            pos = clamp(pos, -max_pos_arr, max_pos_arr)
            vel = np.zeros(2)

        done = bool(np.linalg.norm(pos - self.goal) < self.tol)
        if self.reward_structure == 'sparse':
            reward = 100.0 if done else -1.0
        elif self.reward_structure == 'dense':
            reward = 100.0 if done else 1/(np.linalg.norm(pos - self.goal) + 1e-5) 

        self.state = np.array([pos, vel]).flatten()

        return self.state, reward, done, {}

    def render(self, mode='human'):
        screen_dim = 600
        screen_dim = 600

        world_width = self.max_pos*2
        particle_radius = 20
        goal_radius = 10

        scale = (screen_dim-particle_radius*2)/world_width

        state2render = lambda x: x*scale + screen_dim/2.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_dim, screen_dim)

            self.particle_render = rendering.make_circle(particle_radius)
            self.particle_trans = rendering.Transform()
            self.particle_render.add_attr(self.particle_trans)
            self.particle_render.set_color(.5,.5,.8)
            self.viewer.add_geom(self.particle_render)

            self.goal_render = rendering.make_circle(goal_radius)
            self.goal_trans = rendering.Transform()
            self.goal_render.add_attr(self.goal_trans)
            self.goal_render.set_color(0,1.0,0)
            self.viewer.add_geom(self.goal_render)
            self.goal_trans.set_translation(state2render(self.goal[0]), state2render(self.goal[1]))

            # TODO add boundaries? 
            # self.track = rendering.Line((50,50), (600,50))
            # self.track.set_color(0,0,0)
            # self.viewer.add_geom(self.track)

        if self.state is None: return None

        render_x = state2render(self.state[0])
        render_y = state2render(self.state[1])
        self.particle_trans.set_translation(render_x, render_y)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

class ParticleEnv_3_4(ParticleEnv):
    def __init__(self):
        goal = np.array([3, 4])
        reward_structure = 'sparse'
        super().__init__(goal, reward_structure)





