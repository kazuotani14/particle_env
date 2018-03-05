2D particle with double-integrator dynamics. Positions, velocities, forces are clamped. 

### Adding an environment with a different goal position

1. Make a new class in  `particle_env.py` 

```
class ParticleEnv_1_1(ParticleEnv):
    def __init__(self):
        goal = np.array([1, 1]) # replace this with your goal
        reward_structure = 'sparse' # 'sparse' (-1 unless at goal) or 'dense' (reward = 1/dist2goal)
        super().__init__(goal, reward_structure)
```

2. Import and register environment in `__init__.py`

```
from .particle_env import ParticleEnv_1_1 
register(
    id='Particle-1-1-Sparse-v0', # follow this naming convention, or at least have "-v0" at end (required by gym)
    entry_point='particle_env:ParticleEnv_1_1', 
    max_episode_steps=200, 
    )
```


