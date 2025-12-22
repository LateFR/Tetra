from flax import linen as nn
from jax import numpy as jnp

class Level1Network(nn.Module):
    """
    Réseau N1 : Contrôleur bas niveau avec double flux
    
    Inputs:
        - instruction_N2 : vecteur d'instruction (pour l'instant on met des zéros)
        - proprio : proprioception (vitesse du chariot, position absolue)
    
    Output:
        - action : force à appliquer (entre -1 et 1)
    """
    
    instruction_dim: int = 4   # taille du vecteur d'instruction
    hidden_dim: int = 128        # neurones cachés (petit réseau)
    output_dim: int = 17        
    @nn.compact
    def __call__(self, instruction_N2, proprio):
        """
        Forward pass du réseau
        
        Args:
            instruction_N2 : jnp.array de taille [instruction_dim]
            proprio : jnp.array de taille 2 (position + vitesse)
        
        Returns:
            action : float (force entre -1 et 1)
        """
        
        # === FLUX COMMANDÉ : Interprète l'instruction de N2 ===
        commanded = nn.Dense(features=self.hidden_dim, name='cmd_dense1')(instruction_N2)
        commanded = nn.tanh(commanded)
        commanded = nn.Dense(features=self.output_dim, name='cmd_output')(commanded)
        commanded = nn.tanh(commanded)  # action entre -1 et 1
        
        # === FLUX RÉFLEXE : Réagit à la proprioception ===
        reflexive = nn.Dense(features=self.hidden_dim, name='ref_dense1')(proprio)
        reflexive = nn.tanh(reflexive)
        reflexive = nn.Dense(features=self.output_dim, name='ref_output')(reflexive)
        reflexive = nn.tanh(reflexive)
        
        # === ATTENTION : Quel flux privilégier ? ===
        # Concatène instruction + proprio pour décider
        attention_input = jnp.concatenate([instruction_N2, proprio])
        attention = nn.Dense(features=4, name='att_dense1')(attention_input)
        attention = nn.tanh(attention)
        attention = nn.Dense(features=self.output_dim, name='att_output')(attention)
        attention = nn.sigmoid(attention)  # poids entre 0 et 1
        
        # === FUSION ===
        # action finale = mélange pondéré des deux flux
        mu = attention * commanded + (1 - attention) * reflexive
        
        log_std = self.param('log_std', nn.initializers.zeros, (self.output_dim,))
        
        value = nn.Dense(1)(attention_input)
        value = nn.tanh(value)
        value = nn.Dense(1)(value)
        value = jnp.squeeze(value, axis=-1)

        return (mu, log_std, value)