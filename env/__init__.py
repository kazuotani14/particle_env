from gym.envs.registration import registry, register, make, spec

from particle_env.env.particle_env import ParticleEnv, ParticleEnv_3_4

register(
    id='Particle-v0',
    entry_point='particle_env.env:ParticleEnv',
    max_episode_steps=500, 
    )

register(
    id='Particle-3-4-Sparse-v0',
    entry_point='particle_env.env:ParticleEnv_3_4',
    max_episode_steps=500, 
    )

