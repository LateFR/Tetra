import jax
import jax.numpy as jnp
from model import Level1Network
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from IPython.display import clear_output
import os
import tqdm
import brax
from brax import envs
from brax.io import html
import optax
from typing import NamedTuple

# ============================================================================
# FONCTIONS PPO
# ============================================================================

def compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
    """
    Generalized Advantage Estimation (GAE)
    
    GAE permet d'estimer l'avantage de prendre une action dans un état
    en équilibrant biais et variance.
    """
    advantages = []
    gae = 0.0
    
    # On parcourt à l'envers (du futur vers le présent)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = jnp.array(advantages)
    returns = advantages + jnp.array(values)
    
    return advantages, returns

def ppo_loss(params, network, batch, clip_epsilon=0.2, vf_coef=0.5, ent_coef=0.01):
    """
    Loss function de PPO
    
    PPO optimise une fonction objectif "clippée" qui empêche des changements
    trop importants de la politique.
    """
    obs, actions, old_log_probs, returns, advantages = batch
    
    # Forward pass - ton réseau retourne (mu, log_std, value)
    instruction = jnp.ones(4)
    
    def forward(o):
        mu, log_std, value = network.apply(params, instruction, o)
        return mu, log_std, value
    
    mus, log_stds, values = jax.vmap(forward)(obs)
    
    # Calcul des log probabilités avec la distribution gaussienne
    # log P(action | state) = -0.5 * ((action - mu) / exp(log_std))^2 - log_std - 0.5*log(2π)
    std = jnp.exp(log_stds)
    log_probs = -0.5 * jnp.sum(((actions - mus) / std) ** 2, axis=-1) \
                - jnp.sum(log_stds, axis=-1) \
                - 0.5 * actions.shape[-1] * jnp.log(2 * jnp.pi)
    
    # Ratio de probabilité (nouveau / ancien)
    ratio = jnp.exp(log_probs - old_log_probs)
    
    # Loss policy (clippée) - empêche des changements trop brutaux
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
    
    # Loss value (Mean Squared Error entre valeur prédite et retour réel)
    value_loss = jnp.mean((returns - values) ** 2)
    
    # Entropy bonus (encourage l'exploration en gardant la politique stochastique)
    # Pour une gaussienne : entropie = 0.5 * log(2πe * σ²)
    entropy = jnp.mean(jnp.sum(log_stds + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1))
    
    # Loss totale
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    
    return total_loss, {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy,
        'total_loss': total_loss
    }

# ============================================================================
# COLLECTE DE DONNÉES
# ============================================================================

def collect_trajectories(env, network, params, rng, num_envs=32, num_steps=200):
    """
    Collecte des trajectories en parallèle dans plusieurs environnements
    
    Utilise jax.lax.scan pour paralléliser AUSSI la boucle temporelle !
    """
    rngs = jax.random.split(rng, num_envs)
    states = jax.vmap(env.reset)(rngs)
    
    instruction = jnp.ones(4)
    
    # JIT uniquement la fonction scan_step (pas collect_trajectories entière)
    @jax.jit
    def rollout_scan(params, states, rngs_steps):
        """Partie JIT-able du rollout"""
        
        def scan_step(carry, rng_step):
            """Une step de collecte - sera déroulée par scan"""
            states = carry
            
            # Forward pass - ton réseau retourne (mu, log_std, value)
            def forward(s):
                mu, log_std, value = network.apply(params, instruction, s.obs)
                return mu, log_std, value
            
            mus, log_stds, values = jax.vmap(forward)(states)
            
            # Sample actions depuis la distribution gaussienne
            action_noise = jax.random.normal(rng_step, mus.shape)
            actions = mus + jnp.exp(log_stds) * action_noise
            
            # Clip actions (Brax aime généralement [-1, 1])
            actions = jnp.clip(actions, -1.0, 1.0)
            
            # Log probs des actions choisies
            std = jnp.exp(log_stds)
            log_probs = -0.5 * jnp.sum(((actions - mus) / std) ** 2, axis=-1) \
                        - jnp.sum(log_stds, axis=-1) \
                        - 0.5 * actions.shape[-1] * jnp.log(2 * jnp.pi)
            
            # Step dans l'environnement
            next_states = jax.vmap(lambda s, a: env.step(s, a))(states, actions)
            
            # Rewards personnalisés
            rewards = jax.vmap(compute_reward)(next_states)
            
            # Ce qu'on garde pour cette step
            transition = {
                'obs': states.obs,
                'action': actions,
                'reward': rewards,
                'next_obs': next_states.obs,
                'done': jnp.zeros(num_envs),
                'log_prob': log_probs,
                'value': values
            }
            
            return next_states, transition
        
        # SCAN : déroule la boucle et compile tout !
        final_states, transitions = jax.lax.scan(
            scan_step,
            states,
            rngs_steps
        )
        
        return transitions
    
    # Génère les RNG keys pour toutes les steps d'un coup
    rng, scan_rng = jax.random.split(rng)
    rngs_steps = jax.random.split(scan_rng, num_steps)
    
    # Appelle la version JIT
    transitions = rollout_scan(params, states, rngs_steps)
    
    # transitions est maintenant un dictionnaire de arrays de shape (num_steps, num_envs, ...)
    # On les reformate en liste de dicts pour compatibilité avec le reste du code
    trajectories = []
    for i in range(num_steps):
        trajectories.append({
            'obs': transitions['obs'][i],
            'action': transitions['action'][i],
            'reward': transitions['reward'][i],
            'next_obs': transitions['next_obs'][i],
            'done': transitions['done'][i],
            'log_prob': transitions['log_prob'][i],
            'value': transitions['value'][i]
        })
    
    return trajectories

# ============================================================================
# FONCTION DE REWARD (identique à ton code)
# ============================================================================

def compute_reward(state):
    ps = state.pipeline_state
    torso_pos = ps.x.pos[0]
    torso_rot = ps.x.rot[0]
    torso_height = torso_pos[2]
    
    tilt_penalty = jnp.sum(torso_rot[1:3] ** 2)
    lin_vel = jnp.linalg.norm(ps.xd.vel[0])
    ang_vel = jnp.linalg.norm(ps.xd.ang[0])
    
    height_reward = jnp.clip(torso_height, 0.0, 1.5)
    balance_reward = 2.0 * jnp.exp(-5.0 * tilt_penalty)
    stability_penalty = 0.01 * (lin_vel + ang_vel)
    survival_bonus = 0.1
    
    reward = height_reward + balance_reward + survival_bonus - stability_penalty
    reward = jnp.where(torso_height < 0.5, -1.0, reward)
    
    return reward

# ============================================================================
# ENTRAÎNEMENT PPO
# ============================================================================

def train_ppo(
    env,
    network,
    params,
    num_iterations=1000,
    num_envs=32,
    num_steps=200,
    num_epochs=4,
    batch_size=512,
    learning_rate=3e-4
):
    """Boucle d'entraînement PPO"""
    
    # Optimiseur Adam avec learning rate scheduling
    schedule = optax.linear_schedule(
        init_value=learning_rate,
        end_value=learning_rate * 0.1,
        transition_steps=num_iterations
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)
    
    rng = jax.random.PRNGKey(0)
    stats = []
    
    print(f"Entraînement PPO: {num_iterations} iterations")
    print(f"Collecte: {num_envs} envs × {num_steps} steps = {num_envs * num_steps} transitions/iter")
    print(f"Optimisation: {num_epochs} epochs × {(num_envs * num_steps) // batch_size} batches")
    print()
    
    for iteration in tqdm.tqdm(range(num_iterations), desc="PPO Training"):
        # 1. COLLECTE DE DONNÉES
        rng, collect_rng = jax.random.split(rng)
        trajectories = collect_trajectories(
            env, network, params, collect_rng, num_envs, num_steps
        )
        
        # 2. CALCUL DES AVANTAGES (GAE)
        obs = jnp.concatenate([t['obs'] for t in trajectories])
        actions = jnp.concatenate([t['action'] for t in trajectories])
        rewards = jnp.concatenate([t['reward'] for t in trajectories])
        values = jnp.concatenate([t['value'] for t in trajectories])
        log_probs = jnp.concatenate([t['log_prob'] for t in trajectories])
        dones = jnp.concatenate([t['done'] for t in trajectories])
        next_obs = jnp.concatenate([t['next_obs'] for t in trajectories])
        
        # Valeurs des next states
        instruction = jnp.ones(4)
        def get_value(o):
            _, _, value = network.apply(params, instruction, o)
            return value
        next_values = jax.vmap(get_value)(next_obs)
        
        # GAE
        advantages, returns = compute_gae(rewards, values, next_values, dones)
        
        # Normalisation des avantages (stabilise l'entraînement)
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        
        # 3. OPTIMISATION (plusieurs epochs sur le même batch de données)
        dataset_size = obs.shape[0]
        
        # JIT la fonction d'update pour accélérer !
        @jax.jit
        def update_step(params, opt_state, batch):
            """Une step d'optimisation - JIT-able !"""
            loss_fn = lambda p: ppo_loss(p, network, batch)
            (loss_val, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, info
        
        epoch_losses = []
        for epoch in range(num_epochs):
            # Shuffle des données
            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, dataset_size)
            
            obs_shuffled = obs[perm]
            actions_shuffled = actions[perm]
            log_probs_shuffled = log_probs[perm]
            returns_shuffled = returns[perm]
            advantages_shuffled = advantages[perm]
            
            # Mini-batches
            num_batches = dataset_size // batch_size
            
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                
                batch = (
                    obs_shuffled[start:end],
                    actions_shuffled[start:end],
                    log_probs_shuffled[start:end],
                    returns_shuffled[start:end],
                    advantages_shuffled[start:end]
                )
                
                # Appelle la version JIT
                params, opt_state, info = update_step(params, opt_state, batch)
                epoch_losses.append(info)
        
        # 4. LOGGING
        mean_reward = jnp.mean(rewards)
        max_reward = jnp.max(rewards)
        min_reward = jnp.min(rewards)
        
        # Moyenne des losses sur toutes les batches
        avg_info = {k: jnp.mean(jnp.array([d[k] for d in epoch_losses])) 
                    for k in epoch_losses[0].keys()}
        
        stats.append({
            'iteration': iteration,
            'mean_reward': float(mean_reward),
            'max_reward': float(max_reward),
            'min_reward': float(min_reward),
            'policy_loss': float(avg_info['policy_loss']),
            'value_loss': float(avg_info['value_loss']),
            'entropy': float(avg_info['entropy']),
            'total_loss': float(avg_info['total_loss'])
        })
        
        if iteration % 10 == 0:
            tqdm.tqdm.write(
                f"Iter {iteration:4d} | "
                f"Reward: {mean_reward:6.2f} (min: {min_reward:6.2f}, max: {max_reward:6.2f}) | "
                f"PL: {avg_info['policy_loss']:.4f} | "
                f"VL: {avg_info['value_loss']:.4f} | "
                f"Ent: {avg_info['entropy']:.4f}"
            )
        
        # Sauvegarde périodique
        if iteration % 100 == 0 and iteration > 0:
            os.makedirs("./models/ppo", exist_ok=True)
            # Tu peux sauvegarder params ici si besoin
            plot_training_stats(stats, iteration)
    
    return params, stats

def plot_training_stats(stats, iteration):
    """Génère des graphiques de l'entraînement"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Rewards
    ax = axes[0, 0]
    ax.plot([s['mean_reward'] for s in stats], label='Mean', alpha=0.8)
    ax.plot([s['max_reward'] for s in stats], label='Max', alpha=0.5)
    ax.fill_between(
        range(len(stats)),
        [s['min_reward'] for s in stats],
        [s['max_reward'] for s in stats],
        alpha=0.2
    )
    ax.set_title('Reward')
    ax.set_xlabel('Iteration')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Policy Loss
    ax = axes[0, 1]
    ax.plot([s['policy_loss'] for s in stats])
    ax.set_title('Policy Loss')
    ax.set_xlabel('Iteration')
    ax.grid(alpha=0.3)
    
    # Value Loss
    ax = axes[1, 0]
    ax.plot([s['value_loss'] for s in stats])
    ax.set_title('Value Loss')
    ax.set_xlabel('Iteration')
    ax.grid(alpha=0.3)
    
    # Entropy
    ax = axes[1, 1]
    ax.plot([s['entropy'] for s in stats])
    ax.set_title('Entropy (exploration)')
    ax.set_xlabel('Iteration')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("./stats", exist_ok=True)
    plt.savefig(f'./stats/ppo_training_{iteration}.png')
    plt.close()

# ============================================================================
# ÉVALUATION DU MODÈLE
# ============================================================================

def evaluate_model(env, network, params, rng, num_episodes=10, num_steps=500):
    """Évalue le modèle entraîné sur plusieurs épisodes"""
    
    instruction = jnp.ones(4)
    all_rewards = []
    
    for ep in range(num_episodes):
        rng, reset_rng = jax.random.split(rng)
        state = env.reset(reset_rng)
        
        episode_reward = 0.0
        states_list = []
        
        for step in range(num_steps):
            # Action déterministe (mu sans bruit)
            mu, log_std, value = network.apply(params, instruction, state.obs)
            action = jnp.clip(mu, -1.0, 1.0)
            
            state = env.step(state, action)
            reward = compute_reward(state)
            episode_reward += reward
            
            states_list.append(state.pipeline_state)
        
        all_rewards.append(episode_reward)
        print(f"Episode {ep+1}/{num_episodes}: reward = {episode_reward:.2f}")
    
    mean_reward = jnp.mean(jnp.array(all_rewards))
    std_reward = jnp.std(jnp.array(all_rewards))
    
    print(f"\nÉvaluation finale: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return states_list, mean_reward

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("✓ Initialisation...")
    
    # Environnement
    env = envs.get_environment(env_name='humanoid', backend='mjx')
    
    # Ton réseau (il retourne déjà mu, log_std, value!)
    network = Level1Network()
    
    # Initialisation
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    dummy_instruction = jnp.ones(4)
    dummy_proprio = jnp.zeros(env.observation_size)
    params = network.init(init_rng, dummy_instruction, dummy_proprio)
    
    print(f"Observation size: {env.observation_size}")
    print(f"Action size: {env.action_size}")
    print()
    
    print("✓ Entraînement PPO...")
    
    # Entraînement
    final_params, stats = train_ppo(
        env=env,
        network=network,
        params=params,
        num_iterations=500,      # Nombre d'itérations PPO
        num_envs=64,             # Nombre d'environnements parallèles
        num_steps=200,           # Steps par environnement
        num_epochs=4,            # Epochs d'optimisation par iteration
        batch_size=512,          # Taille des mini-batches
        learning_rate=3e-4       # Learning rate initial
    )
    
    print("\n✓ Entraînement terminé!")
    
    # Graphiques finaux
    plot_training_stats(stats, len(stats))
    
    # Évaluation
    print("\n✓ Évaluation du modèle...")
    rng, eval_rng = jax.random.split(rng)
    states, mean_reward = evaluate_model(env, network, final_params, eval_rng)
    
    # Génération HTML
    print("\n✓ Génération de la visualisation HTML...")
    os.makedirs("./models/ppo", exist_ok=True)
    html_str = html.render(sys=env.sys, states=states[-500:], height=480)
    with open("./models/ppo/best_model.html", "w") as f:
        f.write(html_str)
    
    print("\n✓ Terminé! Visualisation disponible dans ./models/ppo/best_model.html")