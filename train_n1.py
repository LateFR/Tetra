from brax import envs

env = envs.get_environment("ant")

print(f"Action size: {env.action_size}")
print(f"Observation size: {env.observation_size}")