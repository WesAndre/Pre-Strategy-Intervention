import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import copy
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import flashbax as fbx
import wandb

from jaxmarl import make
from jaxmarl.wrappers.baselines import (
    LogWrapper,
    CTRolloutManager,
    PrePolicyWrapper,
)
# Note: Ensure you have created laies_agent.py as previously discussed
from agent.laies_agent import LAIESAgent, ScannedRNN

@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict

class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0
    test_returns: float = 0.0

def make_train(config, env):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=config["EPS_DECAY"] * config["NUM_UPDATES"],
    )

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def train(rng):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        
        # 1. LAIES AGENT AND OPTIMIZERS
        network = LAIESAgent(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config["HIDDEN_SIZE"],
            init_scale=config["INIT_SCALE"]
        )

        def create_agent(rng):
            init_batch_x = (
                jnp.zeros((len(wrapped_env.agents), 1, 1, wrapped_env.obs_size)),
                jnp.zeros((len(wrapped_env.agents), 1, 1)),
            )
            init_batch_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(wrapped_env.agents), 1
            )

            params = network.init(rng, init_batch_hs, *init_batch_x, init_batch_hs)
            
            # Separate Learning Rates for Q-network and RND Predictor
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=config["LR"]),
            )

            return CustomTrainState.create(
                apply_fn=network.apply,
                params=params,
                target_network_params=params,
                tx=tx,
            )

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        # BUFFER SETUP
        # (Standard buffer initialization from iql_pre.py)
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        # ... (buffer_state initialization)

        def _update_step(runner_state, unused):
            train_state, buffer_state, test_state, rng = runner_state

            # 2. STEP ENV WITH LAIES REWARD
            def _step_env(carry, _):
                hs, last_obs, last_dones, pre_hs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]

                new_hs, q_vals, (target_feat, pred_feat) = network.apply(
                    train_state.params, hs, _obs, _dones, pre_hs
                )

                # LAIES Reward Math: Curiosity - Laziness Penalty
                # curiosity = ||target_feat - pred_feat||^2
                curiosity = jnp.mean((target_feat - pred_feat)**2, axis=-1).squeeze(1)
                
                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, unbatchify(jnp.argmax(q_vals.squeeze(1), axis=-1))
                )

                # Apply laziness penalty if agent is inactive while team gets reward
                extrinsic = batchify(rewards)
                laziness_penalty = jnp.where(extrinsic > 0, 1.0 / (curiosity + 1e-5), 0.0)
                laies_reward = (config["CURIOSITY_SCALE"] * curiosity) - (config["LAIES_COEFF"] * laziness_penalty)
                
                # Combine total reward
                total_reward = unbatchify(extrinsic + laies_reward)

                timestep = Timestep(obs=last_obs, actions=unbatchify(jnp.argmax(q_vals.squeeze(1), axis=-1)), 
                                    rewards=total_reward, dones=dones, avail_actions={})
                return (new_hs, new_obs, dones, pre_hs, new_env_state, rng), (timestep, infos)

            # ... (Rest of rollout logic)

            # 3. LEARN PHASE: UPDATE Q AND RND NETWORKS
            def _learn_phase(carry, _):
                train_state, rng = carry
                # ... (sampling from buffer)

                def _loss_fn(params):
                    _, q_vals, (target_f, pred_f) = network.apply(params, init_hs, _obs, _dones, init_hs)
                    
                    # RND Loss: Train predictor to match target features
                    rnd_loss = jnp.mean((jax.lax.stop_gradient(target_f) - pred_f)**2)
                    
                    # Standard Q-learning Loss
                    # ... (Standard TD-error calculation from iql_pre.py)
                    
                    return q_loss + rnd_loss, (q_loss, rnd_loss)

                (total_loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                return (train_state, rng), (total_loss, aux)

            # ... (standard update logic)
            return (train_state, buffer_state, test_state, rng), None

    return train

# ... (End of your existing make_train function)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    # Convert Hydra config to standard Python dict
    config = OmegaConf.to_container(config)
    
    # Initialize the environment
    # Note: We check if it's Hanabi to handle specific arguments if needed
    print(f"Loaded Algorithm: {config['alg']['ALG_NAME']}")
    print(f"Target Environment: {config['alg']['ENV_NAME']}")

    # Create the Environment and Training Function
    env = make(config["alg"]["ENV_NAME"], **config["alg"]["ENV_KWARGS"])
    train_vjit = jax.jit(make_train(config["alg"], env))
    
    # Run Training
    print("Starting Training...")
    rng = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
    runner_state = train_vjit(rng)
    print("Training Complete!")

if __name__ == "__main__":
    main()