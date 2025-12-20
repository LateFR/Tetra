# notebook_visualize.py  (fonctionne dans Jupyter / Colab)
import jax, jax.numpy as jnp
from brax.envs import create
from brax.io import html
from model import Level1Network
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from IPython.display import clear_output
import os

network = Level1Network()

env = create('humanoid')               # ou 'ant','humanoid',...

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)



# Initialiser
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
dummy_instruction = jnp.zeros(4)
dummy_proprio = jnp.zeros(env.observation_size)
params = network.init(init_rng, dummy_instruction, dummy_proprio)
stats = []

pop_size = 64
k = 8

def compute_reward(state):
    ps = state.pipeline_state

    # Torse = body 0
    torso_pos = ps.x.pos[0]     # (x, y, z)
    torso_rot = ps.x.rot[0]     # (w, x, y, z)

    torso_height = torso_pos[2]

    # pénalité d'inclinaison
    tilt_penalty = jnp.sum(torso_rot[1:3] ** 2)

    lin_vel = jnp.linalg.norm(ps.xd.vel[0])
    ang_vel = jnp.linalg.norm(ps.xd.ang[0])

    height_reward = jnp.clip(torso_height - 1.0, 0.0, 1.0)
    balance_reward = jnp.exp(-5.0 * tilt_penalty)

    stability_penalty = 0.05 * lin_vel + 0.05 * ang_vel

    reward = height_reward + balance_reward - stability_penalty
    reward = jnp.where(torso_height < 0.8, -5.0, reward)

    return reward




def rollout_fitness(params, rng, env, network, steps=200):
    state = env.reset(rng)
    total_reward = 0.0
    
    for step in range(steps):
        instruction = jnp.ones(4)
        proprio = state.obs
        
        action = network.apply(params, instruction, proprio)
        state = env.step(state, action)
        total_reward += compute_reward(state)
    return total_reward

def init_pop(base_params, pop_size, rng, sigma=0.1):
    rngs = jax.random.split(rng, pop_size)
    
    def mutate(rng):
        noise = jax.tree_util.tree_map(
            lambda p: sigma * jax.random.normal(rng, p.shape),
            base_params
        )
        return jax.tree_util.tree_map(lambda p, n: p + n, base_params, noise)
    return jax.vmap(mutate)(rngs)

def select(pop, fitness, k):
    top_k = jnp.argsort(fitness)[-k:][::-1]
    elites = jax.tree_util.tree_map(lambda x: x[top_k], pop)
    return elites
    
def generate_stats(fitness, k):
    return {
        "fitness": fitness,
        "average": jnp.mean(fitness),
        "k_best": jnp.max(fitness),
        "k_worst": jnp.min(fitness),
        "k_average": jnp.mean(fitness),
        "k_bests": jnp.sort(fitness)[-k:],
        "k_best_average": jnp.mean(jnp.sort(fitness)[-k:])
    }

def reproduce(elites, pop_size, rng, k, sigma=0.05):
    rngs = jax.random.split(rng, pop_size)
    
    def mutate(rng):
        idx = jax.random.randint(rng, (), 0, k)
        parent = jax.tree_util.tree_map(lambda x: x[idx], elites)
        
        noise = jax.tree_util.tree_map(lambda p: sigma * jax.random.normal(rng, p.shape), parent)
        return jax.tree_util.tree_map(lambda p, n: p + n, parent, noise)
    return jax.vmap(mutate)(rngs)

def plot_stats(stats):
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Graphique 1: Fitness moyen et best
    ax1.plot([s["average"] for s in stats], label="Moyenne pop", alpha=0.7)
    ax1.plot([s["k_best"] for s in stats], label="Meilleur", alpha=0.7)
    ax1.plot([s["k_best_average"] for s in stats], label=f"Moyenne top-{k}", alpha=0.7)
    ax1.set_xlabel("Génération")
    ax1.set_ylabel("Fitness")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Graphique 2: Distribution dernière génération
    ax2.hist(stats[-1]["fitness"], bins=20, alpha=0.7)
    ax2.axvline(stats[-1]["average"], color='r', linestyle='--', label="Moyenne")
    ax2.set_xlabel("Fitness")
    ax2.set_ylabel("Nombre d'individus")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("./stats", exist_ok=True)
    plt.savefig(f"./stats/stats-{step}.png")
    plt.close()

def run_best_model(best_params, env, network, rng, steps=300, name="best_model", path="./models"):
    
    os.makedirs(path, exist_ok=True)
    
    state = env.reset(rng)

    torso_heights = []
    tilt_values = []
    rewards = []
    states = []

    for step in range(steps):
        instruction = jnp.ones(4)
        proprio = state.obs

        action = network.apply(best_params, instruction, proprio)
        state = env.step(state, action)

        # --- métriques physiques ---
        torso_pos = state.pipeline_state.x.pos[0]     # (x, y, z)
        torso_rot = state.pipeline_state.x.rot[0]     # quaternion (w, x, y, z)

        height = float(torso_pos[2])
        tilt = float(jnp.sum(torso_rot[1:3] ** 2))
        reward = float(compute_reward(state))

        torso_heights.append(height)
        tilt_values.append(tilt)
        rewards.append(reward)

        states.append(state.pipeline_state)

    # --- plots ---
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.plot(torso_heights)
    plt.axhline(1.0, linestyle="--", alpha=0.5)
    plt.title("Torso height")
    plt.xlabel("Step")

    plt.subplot(1,3,2)
    plt.plot(tilt_values)
    plt.title("Tilt penalty")
    plt.xlabel("Step")

    plt.subplot(1,3,3)
    plt.plot(rewards)
    plt.title("Reward")
    plt.xlabel("Step")

    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{name}_metrics.png"))
    plt.close()

    # --- HTML animation ---
    html_str = html.render(
        sys=env.sys,
        states=states,
        height=480
    )

    with open(os.path.join(path, f"{name}.html"), "w") as f:
        f.write(html_str)
    print("✓ Visualisation générée :")

vmap_fitness = jax.vmap(rollout_fitness, in_axes=(0, 0, None, None))
pop = init_pop(params, pop_size, rng)

num_generations = 100
for step in range(num_generations):
    rng, step_rng = jax.random.split(rng)
    rngs = jax.random.split(step_rng, pop_size)
    fitness = vmap_fitness(pop, rngs, env, network)
    
    # normaliser
    mean_f = jnp.mean(fitness)
    std_f = jnp.std(fitness) + 1e-8  # éviter division par zéro
    fitness_norm = (fitness - mean_f) / std_f

    stats.append(generate_stats(fitness_norm, k))
    
    # Logging simple
    if step % 10 == 0 or step == num_generations - 1:
        print(f"Gen {step:3d}/{num_generations} | "
              f"Best: {stats[-1]['k_best']:.2f} | "
              f"Avg: {stats[-1]['average']:.2f} | "
              f"Top-{k}: {stats[-1]['k_best_average']:.2f}")
    
    best_idx = jnp.argmax(stats[-1]["fitness"])
    best_params = jax.tree_util.tree_map(lambda x: x[best_idx], pop)
    run_best_model(best_params, env, network, rng, steps=300, name=f"model_{step}", path=f"./models/v1/{step}")
    # Affichage graphique
    if step % 20 == 0 or step == num_generations - 1:
        plot_stats(stats)
    
    elites = select(pop, fitness_norm, k)
    pop = reproduce(elites, pop_size, rng, k)

print("\n✓ Entraînement terminé!")
print("Best model:")

best_idx = jnp.argmax(stats[-1]["fitness"])
best_params = jax.tree_util.tree_map(lambda x: x[best_idx], pop)

run_best_model(best_params, env)