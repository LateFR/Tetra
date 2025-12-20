import jax
import jax.numpy as jnp
from flax import struct

# === ENVIRONNEMENT MINIMALISTE : Chariot 1D ===

@struct.dataclass
class CartState:
    """État du chariot"""
    position: float  # position sur l'axe X
    velocity: float  # vitesse
    time: int        # temps écoulé
    
class SimpleCart1D:
    """
    Environnement ultra-simple :
    - Un chariot sur une ligne (1D)
    - 1 moteur qui pousse gauche (-) ou droite (+)
    - Objectif : aller le plus loin possible à droite
    """
    
    def __init__(self):
        self.dt = 0.02  # pas de temps (20ms)
        self.max_steps = 500
        self.friction = 0.1  # frottement
        
    def reset(self, rng: jax.random.PRNGKey) -> CartState:
        """Réinitialise l'environnement"""
        return CartState(
            position=0.0,
            velocity=0.0,
            time=0
        )
    
    def step(self, state: CartState, action: float) -> CartState:
        """
        Fait avancer la simulation d'un pas de temps
        
        Args:
            state : état actuel
            action : force appliquée (entre -1 et 1)
        
        Returns:
            nouveau state
        """
        # Limiter l'action entre -1 et 1
        action = jnp.clip(action, -1.0, 1.0)
        
        # Physique simple : F = ma => a = F/m (on suppose m=1)
        acceleration = action - self.friction * state.velocity
        
        # Intégration d'Euler
        new_velocity = state.velocity + acceleration * self.dt
        new_position = state.position + new_velocity * self.dt
        
        return CartState(
            position=new_position,
            velocity=new_velocity,
            time=state.time + 1
        )
    
    def get_obs(self, state: CartState) -> jnp.ndarray:
        """
        Observation = ce que la "créature" perçoit
        Ici : juste sa vitesse (elle ne voit pas sa position absolue)
        """
        return jnp.array([state.velocity])
    
    def is_done(self, state: CartState) -> bool:
        """L'épisode est-il terminé ?"""
        return state.time >= self.max_steps