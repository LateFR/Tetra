import os
from env_1d import SimpleCart1D
import jax
from model import Level1Network
from jax import numpy as jnp
from brax.io import html
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from IPython.display import clear_output

env = SimpleCart1D()
network = Level1Network(instruction_dim=4, hidden_dim=8)

# Initialiser
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
dummy_instruction = jnp.zeros(4)
dummy_proprio = jnp.array([1.0, 1.0])
params = network.init(init_rng, dummy_instruction, dummy_proprio)
stats = []

pop_size = 64
k = 8

def compute_reward(state):
    return 1 - jnp.abs(state.position - 1) - 0.1 * jnp.abs(state.velocity)

def rollout_fitness(params, rng, env, network, steps=200):
    state = env.reset(rng)
    total_reward = 0.0
    
    for step in range(steps):
        instruction = jnp.ones(4)
        proprio = jnp.array([state.velocity, state.position])
        
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
    os.makedirs("./learn/stats", exist_ok=True)
    plt.savefig(f"./learn/stats/stats-{step}.png")
    plt.close()

vmap_fitness = jax.vmap(rollout_fitness, in_axes=(0, 0, None, None))
pop = init_pop(params, pop_size, rng)

num_generations = 200
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
    
    # Affichage graphique
    if step % 20 == 0 or step == num_generations - 1:
        plot_stats(stats)
    
    elites = select(pop, fitness_norm, k)
    pop = reproduce(elites, pop_size, rng, k)

print("\n✓ Entraînement terminé!")
print("Best model:")

best_idx = jnp.argmax(stats[-1]["fitness"])
best_params = jax.tree_util.tree_map(lambda x: x[best_idx], pop)

def run_best_model(best_params, env):
    state = env.reset(rng)
    positions = []
    velocities = []

    for step in range(200):
        instruction = jnp.ones(4)
        proprio = jnp.array([state.velocity, state.position])

        action = network.apply(best_params, instruction, proprio)
        state = env.step(state, action)

        positions.append(float(state.position))
        velocities.append(float(state.velocity))
        
    plt.figure(figsize=(10,4))
    plt.plot(positions, label="Position")
    plt.plot(velocities, label="Vitesse")
    plt.xlabel("Step")
    plt.ylabel("Valeur")
    plt.title("Trajectoire du meilleur modèle")
    plt.legend()
    plt.savefig("./learn/best_model.png")
    plt.close()


run_best_model(best_params, env)