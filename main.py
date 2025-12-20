# notebook_visualize.py  (fonctionne dans Jupyter / Colab)
import jax, jax.numpy as jnp
from brax.envs import create
from brax.io import html

env = create('humanoid')               # ou 'ant','humanoid',...
env = env  # si tu veux vectoriser ensuite, vois plus bas

key = jax.random.PRNGKey(0)
state = env.reset(key)

state_history = []
for t in range(100):
    # action aléatoire (exemple) — adapte à ta policy
    a = jax.random.uniform(key, (env.action_size,), minval=-1.0, maxval=1.0)
    state = env.step(state, a)
    # stocke le pipeline_state (ou state.qp selon version)
    state_history.append(state.pipeline_state)
    key, _ = jax.random.split(key)
    if state.done:
        break

# render: on passe le sys et la liste de pipeline_states
html_str = html.render(sys=env.sys, states=state_history, height=480)

with open('index.html', 'w') as f:
    f.write(html_str)