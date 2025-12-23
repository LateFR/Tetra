import time
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
from functools import partial
# ============================================================================
# CONFIGURATION GLOBALE
# ============================================================================
STATS_DIR = "./stats"
MODELS_DIR = "./models/ppo"
os.makedirs(STATS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# FONCTIONS PPO
# ============================================================================

@jax.jit
def compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
    # 1. On calcule le TD error (delta) pour TOUT le tableau d'un coup (Vectorisation JAX)
    # C'est beaucoup plus efficace que de le faire dans la boucle scan
    delta = rewards + gamma * next_values * (1 - dones) - values
    
    # 2. La fonction de step pour le scan
    # Elle ne reçoit que delta et le masque (1-dones) pour l'instant t
    def step(gae, inputs):
        d_t, mask_t = inputs  # On ne reçoit que 2 valeurs ici !
        
        # Formule GAE : gae = delta + gamma * lambda * mask * gae_precedent
        gae = d_t + gamma * gae_lambda * mask_t * gae
        
        return gae, gae # (carry, output)

    # 3. Le scan
    # On passe (delta, 1-dones) comme inputs à scanner
    _, advantages = jax.lax.scan(
        step,
        jnp.zeros(rewards.shape[1]), # Carry initial (GAE=0 pour le futur lointain)
        (delta, (1 - dones)),        # Les inputs qu'on itère (axis 0 par défaut)
        reverse=True                 # On remonte le temps (de la fin vers le début)
    )
    
    # 4. Calcul final des returns
    returns = advantages + values
    
    return advantages, returns

def ppo_loss(params, network, batch, clip_epsilon=0.2, vf_coef=0.5, ent_coef=0.005):
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

    # Normalisation des returns pour stabiliser le critic
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
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
    start_time = time.time()
    rngs = jax.random.split(rng, num_envs)
    states = jax.vmap(env.reset)(rngs)

    instruction = jnp.ones(4)
    @jax.jit
    def batch_compute_reward(states):
        """Vectorise le calcul de reward"""
        return jax.vmap(compute_reward)(states)
    
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
            rewards = batch_compute_reward(next_states)

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
    trajectories = rollout_scan(params, states, rngs_steps)


    tqdm.tqdm.write(f"Trajectories collected in {time.time() - start_time:.2f}s")

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
# PLOT ET SAUVEGARDE DES STATISTIQUES
# ============================================================================

def plot_and_save_stats(stats, iteration, model_params, run_states, run_html):
    """
    Génère les plots des statistiques, sauvegarde les résultats et le modèle.
    Crée également une version textuelle condensée des statistiques.
    """
    # 1. Sauvegarde des statistiques textuelles
    stats_text = f"--- Statistics for iteration {iteration} ---\n"
    if stats:
        last_stat = stats[-1]
        stats_text += f"Mean Reward: {last_stat['mean_reward']:.2f}\n"
        stats_text += f"Max Reward:  {last_stat['max_reward']:.2f}\n"
        stats_text += f"Min Reward:  {last_stat['min_reward']:.2f}\n"
        stats_text += f"Policy Loss: {last_stat['policy_loss']:.4f}\n"
        stats_text += f"Value Loss:  {last_stat['value_loss']:.4f}\n"
        stats_text += f"Entropy:     {last_stat['entropy']:.4f}\n"
        stats_text += f"Total Loss:  {last_stat['total_loss']:.4f}\n"
    
    stats_filepath = os.path.join(STATS_DIR, f"stats_iter_{iteration}.txt")
    with open(stats_filepath, "w") as f:
        f.write(stats_text)
    tqdm.tqdm.write(f"Condensed stats saved to {stats_filepath}")

    # 2. Génération des plots (si des stats existent)
    if stats:
        plot_training_stats_img(stats, iteration)

    # 3. Sauvegarde du modèle et des résultats de run
    if model_params:
        model_filepath = os.path.join(MODELS_DIR, f"model_iter_{iteration}.params")
        # Utilisation de `orbax.checkpoint` pour une sauvegarde plus robuste et gestion des états JAX
        # Pour simplifier ici, j'utilise pickle, mais pour des modèles complexes, Orbax est recommandé.
        import pickle
        with open(model_filepath, 'wb') as f:
            pickle.dump(model_params, f)
        tqdm.tqdm.write(f"Model parameters saved to {model_filepath}")

    if run_states:
        run_html_filepath = os.path.join(MODELS_DIR, f"run_iter_{iteration}.html")
        html_str = html.render(sys=run_html['env_sys'], states=run_states, height=480)
        with open(run_html_filepath, "w") as f:
            f.write(html_str)
        tqdm.tqdm.write(f"Run visualization saved to {run_html_filepath}")

# ============================================================================
# PLOT DES STATISTIQUES D'ENTRAÎNEMENT (IMAGÉ)
# ============================================================================
def plot_training_stats_img(stats, iteration):
    """Génère des graphiques de l'entraînement (sauvegardés en PNG)"""
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
    img_filepath = os.path.join(STATS_DIR, f"ppo_training_{iteration}.png")
    plt.savefig(img_filepath)
    plt.close()
    tqdm.tqdm.write(f"Training plot saved to {img_filepath}")

# ============================================================================
# ENTRAÎNEMENT PPO
# ============================================================================
@partial(jax.jit, static_argnames=['batch_size', 'optimizer', 'network'])
def train_epoch_scan(params, opt_state, batch_data, batch_size, rng, optimizer, network):
    """Effectue UNE epoch complète (shuffle + updates) sur GPU"""
    obs, actions, log_probs, returns, advantages = batch_data
    dataset_size = obs.shape[0]
    steps_per_epoch = dataset_size // batch_size
    
    # 1. Shuffle des données
    perm = jax.random.permutation(rng, dataset_size)
    
    # Fonction pour mélanger et reshaper en (Nombre_Batches, Batch_Size, ...)
    def prepare_batch(x):
        shuffled = x[perm]
        # On coupe les données qui dépassent (si dataset_size n'est pas multiple de batch_size)
        trunc_len = steps_per_epoch * batch_size
        shuffled = shuffled[:trunc_len]
        return shuffled.reshape((steps_per_epoch, batch_size) + x.shape[1:])

    obs_b = prepare_batch(obs)
    actions_b = prepare_batch(actions)
    log_probs_b = prepare_batch(log_probs)
    returns_b = prepare_batch(returns)
    advantages_b = prepare_batch(advantages)

    # 2. La boucle d'update (sur GPU via Scan)
    def update_step(carry, batch):
        params, opt_state = carry
        # batch est un tuple (o, a, lp, r, adv) pour UN mini-batch
        loss_fn = lambda p: ppo_loss(p, network, batch)
        (loss_val, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), info

    # Lancement du scan
    (new_params, new_opt_state), infos = jax.lax.scan(
        update_step, 
        (params, opt_state), 
        (obs_b, actions_b, log_probs_b, returns_b, advantages_b)
    )
    
    return new_params, new_opt_state, infos

def train_ppo(
    env,
    network,
    params,
    num_iterations=1500,
    show_stats_every=10,
    save_every=100,
    num_envs=64,
    num_steps=200,
    num_epochs=4,
    batch_size=512,
    learning_rate=1e-4
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
    all_run_states = [] # Stocke les états pour la génération HTML finale
    run_html_data = {'env_sys': env.sys} # Stocke le sys de l'env pour la génération HTML

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
        start_time = time.time()
        # 2. CALCUL DES AVANTAGES (GAE)
        # On aplatit les dimensions Temps et Batch
        def flatten(x):
            return x.reshape((-1,) + x.shape[2:])

        rewards = trajectories['reward']
        values = trajectories['value']
        dones = trajectories['done']
        
        instruction = jnp.ones(4)
        def get_value(o):
            _, _, v = network.apply(params, instruction, o)
            return v
        
        last_value = jax.vmap(get_value)(trajectories['next_obs'][-1]) 
        next_values = jnp.concatenate([values[1:], last_value[None, :]], axis=0)
        
        advantages, returns = compute_gae(rewards, values, next_values, dones)
        
        # GAE
        advantages, returns = compute_gae(rewards, values, next_values, dones)
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8) # norm of advantages
        
        def flatten(x):
            return x.reshape((-1,) + x.shape[2:]) if x.ndim > 2 else x.reshape(-1)

        obs = flatten(trajectories['obs'])
        actions = flatten(trajectories['action'])
        log_probs = flatten(trajectories['log_prob'])
        returns = flatten(returns)
        advantages = flatten(advantages)
        
        tqdm.tqdm.write(f"Calculated advantages, GAE, and rewards in {time.time() - start_time:.2f}s")
        start_time = time.time()
        # 3. OPTIMISATION (plusieurs epochs sur le même batch de données)
        
        batch_data = (obs, actions, log_probs, returns, advantages)

        epoch_losses = []
        
        # Boucle simple sur les epochs (4 itérations, c'est rien pour Python)
        for epoch in range(num_epochs):
            rng, epoch_rng = jax.random.split(rng)
            
            # APPEL DE LA FONCTION SCAN : Tout se passe sur le GPU ici
            params, opt_state, info = train_epoch_scan(
                params, opt_state, batch_data, batch_size, epoch_rng, optimizer, network
            )
            
            # On stocke la moyenne des losses de cette epoch
            avg_loss = {k: jnp.mean(v) for k, v in info.items()}
            epoch_losses.append(avg_loss)

        tqdm.tqdm.write(f"Optimized in {time.time() - start_time:.2f}s")
        
        # 4. LOGGING & SAUVEGARDE
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

        # Évaluation et sauvegarde périodiques
        if iteration % show_stats_every == 0:
            tqdm.tqdm.write(
                f"Iter {iteration:4d} | "
                f"Reward: {mean_reward:6.2f} (min: {min_reward:6.2f}, max: {max_reward:6.2f}) | "
                f"Policy loss: {avg_info['policy_loss']:.4f} | "
                f"Value loss: {avg_info['value_loss']:.4f} | "
                f"Entropy: {avg_info['entropy']:.4f}"
            )

        if iteration % save_every == 0 and iteration > 0:
            # Évaluation pour la sauvegarde HTML
            eval_states, eval_mean_reward = evaluate_model(env, network, params, rng, num_episodes=2, num_steps=num_steps)
            all_run_states.extend(eval_states)

            plot_and_save_stats(stats, iteration, params, all_run_states, run_html_data)
            all_run_states = [] # Réinitialise pour la prochaine sauvegarde

    # Sauvegarde finale des stats, modèle et run
    plot_and_save_stats(stats, num_iterations, params, all_run_states, run_html_data)

    return params, stats

# ============================================================================
# ÉVALUATION DU MODÈLE
# ============================================================================

def evaluate_model(env, network, params, rng, num_episodes=5, num_steps=500):
    """
    Version optimisée qui utilise vmap pour le batching correct.
    """
    start_time = time.time()
    tqdm.tqdm.write(f"Starting evaluation for {num_episodes} episodes...")
    rngs = jax.random.split(rng, num_episodes)
    # Reset de tous les envs en parallèle
    start_states = jax.vmap(env.reset)(rngs)
    instruction = jnp.ones(4)
    
    @jax.jit
    def run_eval(state):
        def step_fn(carry_state, _):
            def get_action_single(o):
                mu, _, _ = network.apply(params, instruction, o)
                return mu
            
            mus = jax.vmap(get_action_single)(carry_state.obs)
            actions = jnp.clip(mus, -1.0, 1.0)
            
            # On step l'environnement en parallèle
            next_state = jax.vmap(env.step)(carry_state, actions)
            rewards = jax.vmap(compute_reward)(next_state)
            
            # On retourne l'état complet pour le HTML et le reward
            return next_state, (next_state.pipeline_state, rewards)

        final_state, (pipeline_states, rewards) = jax.lax.scan(
            step_fn, state, None, length=num_steps
        )
        return pipeline_states, rewards

    # Exécution sur GPU
    pipeline_states, rewards = run_eval(start_states)
    
    # Calculs stats
    total_rewards = jnp.sum(rewards, axis=0)
    mean_reward = jnp.mean(total_rewards)
    
    
    best_idx = jnp.argmax(total_rewards)
    
    
    states_list = []
    
    best_episode_tree = jax.tree_util.tree_map(lambda x: x[:, best_idx], pipeline_states)
    
    for i in range(num_steps):
        s = jax.tree_util.tree_map(lambda x: x[i], best_episode_tree)
        states_list.append(s)
    
    tqdm.tqdm.write(f"Evaluation finished in {time.time() - start_time:.2f}s")
    
    return states_list, float(mean_reward)
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
    num_steps = 200
    # Entraînement
    final_params, stats = train_ppo(
        env=env,
        network=network,
        params=params,
        num_iterations=300,      # Nombre d'itérations PPO
        show_stats_every=20,     # Afficher les stats tous les X itérations
        save_every=50,          # Sauvegarder modèle et stats tous les X itérations
        num_envs=4096,             # Nombre d'environnements parallèles
        num_steps=num_steps,           # Steps par environnement
        num_epochs=4,            # Epochs d'optimisation par iteration
        batch_size=32768,          # Taille des mini-batches
        learning_rate=3e-4       # Learning rate initial
    )

    print("\n✓ Entraînement terminé!")

    # Graphiques finaux et sauvegardes
    # Utiliser le dernier enregistrement pour les plots finaux si save_every est plus petit que num_iterations
    if len(stats) > 0:
        plot_and_save_stats(stats, stats[-1]['iteration'], final_params, [], {'env_sys': env.sys}) # Pas de run states pour le plot final ici

    # Évaluation finale et génération du HTML
    print("\n✓ Évaluation finale du modèle...")
    rng, eval_rng = jax.random.split(rng)
    final_eval_states, final_mean_reward = evaluate_model(env, network, final_params, eval_rng, num_episodes=10, num_steps=500)

    print("\n✓ Génération de la visualisation HTML finale...")
    html_filepath = os.path.join(MODELS_DIR, "best_model.html")
    html_str = html.render(sys=env.sys, states=final_eval_states, height=480)
    with open(html_filepath, "w") as f:
        f.write(html_str)

    print(f"\n✓ Terminé! Statistiques et modèles sauvegardés dans '{STATS_DIR}' et '{MODELS_DIR}'.")
    print(f"Visualisation finale disponible dans {html_filepath}")