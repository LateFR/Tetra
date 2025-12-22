import jax
from brax import envs
from brax.training.agents.ppo import train as ppo_train
from model import Level1Network
import optax
import jax.numpy as jnp
from flax import linen as nn

# --- initialisation ---
env = envs.get_environment('humanoid', backend='mjx')
learning_rate = 0.001
optimizer = optax.adam(learning_rate)

class BraxAdapter(nn.Module):
    base_network: Level1Network
    
    def __call__(self, *args, **kwargs):
        instruction_N2 = jnp.zeros(4)
        return self.base_network(instruction_N2=instruction_N2, *args, **kwargs)
# --- entraînement PPO prêt à l’emploi ---
state = ppo_train(
    environment=env,
    policy_network=BraxAdapter,
    seed=0,
    num_timesteps=1_000_000,  # ou le nombre de steps souhaité
    learning_rate=learning_rate
)
