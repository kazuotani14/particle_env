import numpy as np
from math import sqrt
import gym
import particle_env 

if __name__ == "__main__":
    env = gym.make('Particle-3-4-Sparse-v0')
    goal = np.array([3, 4]) # TODO find a way to build this into env? 

    obs = env.reset()

    done = False
    while not done: 
        env.render()

        # PD controller 
        kp = 3
        kd = 1.5*sqrt(kp)
        force = (goal - obs[:2]) * kp + (-obs[2:]*kd)

        obs, reward, done, _ = env.step(force)

        if done:
            if reward==100:
                print('Goal achieved!')
            else:
                print('Episode ended')

