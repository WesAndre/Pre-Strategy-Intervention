import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import hydra
import chex
import flashbax as fbx
from jaxmarl import make
from jaxmarl.wrappers.baselines import CTRolloutManager
from agent.liir_agent import LIIRAgent, ScannedRNN

@chex.dataclass(frozen=True)
class Timestep:
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    avail_actions: jnp.ndarray

# Custom TrainState to hold BOTH Agent and Proxy parameters
class LIIRTrainState(TrainState):
    target_network_params: any
    proxy_params: any
    proxy_opt_state: any

def make_train(config, env):
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])

    def train(rng):
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        
        # 1. Initialize Network
        network = LIIRAgent(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config["HIDDEN_SIZE"],
            init_scale=config["INIT_SCALE"]
        )

        # 2. Setup Optimizers (Two separate optimizers)
        agent_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=config["LR"])
        )
        proxy_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=config["META_LR"])
        )

        def create_train_state(rng):
            init_obs = jnp.zeros((len(wrapped_env.agents), 1, 1, wrapped_env.obs_size))
            init_dones = jnp.zeros((len(wrapped_env.agents), 1, 1))
            init_hs = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], len(wrapped_env.agents), 1)
            
            # Init all params
            params = network.init(rng, init_hs, init_obs, init_dones)
            
            # Split params into Agent and Proxy
            # Note: Flax stores them in a dict. We treat 'proxy' params separately.
            # For simplicity in this implementation, we will update the whole tree 
            # but mask gradients, or keep them in one TrainState.
            # To strictly separate, we would partition. Here we use one state for simplicity
            # but will apply gradients selectively.
            
            return LIIRTrainState.create(
                apply_fn=network.apply,
                params=params,
                target_network_params=params,
                tx=agent_tx,
                proxy_params=params['proxy'], # Store proxy params separately if needed
                proxy_opt_state=proxy_tx.init(params['proxy'])
            )

        rng, _rng = jax.random.split(rng)
        train_state = create_train_state(_rng)

        # Buffer Setup (Standard)
        dummy_sample = Timestep(
            obs=jnp.zeros((len(wrapped_env.agents), wrapped_env.obs_size)),
            actions=jnp.zeros((len(wrapped_env.agents),), dtype=jnp.int32),
            rewards=jnp.zeros((len(wrapped_env.agents),)),
            dones=jnp.zeros((len(wrapped_env.agents),)),
            avail_actions=jnp.zeros((len(wrapped_env.agents), wrapped_env.max_action_space))
        )
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"],
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(dummy_sample)

        # --- UPDATE STEP ---
        def _update_step(runner_state, unused):
            train_state, buffer_state, env_state, last_obs, last_dones, hs, rng = runner_state

            # A. COLLECT TRAJECTORY
            def _step_env(carry, _):
                hs, obs, dones, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                
                # Get Q values
                obs_batch = obs[:, np.newaxis, np.newaxis, :]
                dones_batch = dones[:, np.newaxis, np.newaxis]
                new_hs, q_vals, _ = network.apply(train_state.params, hs, obs_batch, dones_batch)
                
                # Select Action (Epsilon Greedy)
                eps = config["EPS_START"] # Simplified constant eps
                greedy_a = jnp.argmax(q_vals.squeeze(), axis=-1)
                random_a = jax.random.randint(rng_a, greedy_a.shape, 0, wrapped_env.max_action_space)
                actions = jnp.where(jax.random.uniform(rng_a, greedy_a.shape) < eps, random_a, greedy_a)
                
                # Step Env
                new_obs, new_env_state, rewards, new_dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, {a: actions[i] for i, a in enumerate(env.agents)}
                )
                
                rewards_np = jnp.stack([rewards[a] for a in env.agents])
                dones_np = jnp.stack([new_dones[a] for a in env.agents])
                
                step = Timestep(obs=obs, actions=actions, rewards=rewards_np, dones=dones_np, avail_actions={})
                return (new_hs, new_obs, dones_np, new_env_state, rng), step

            (hs, last_obs, last_dones, env_state, rng), traj = jax.lax.scan(
                _step_env, (hs, last_obs, last_dones, env_state, rng), None, length=config["NUM_STEPS"]
            )
            buffer_state = buffer.add(buffer_state, traj)

            # B. LEARN PHASE (Meta-Gradient)
            def _learn_phase(carry, _):
                ts, rng = carry
                rng, rng_s = jax.random.split(rng)
                batch = buffer.sample(buffer_state, rng_s).experience.first
                
                # 1. INNER LOOP: Calculate Loss using Current Proxy
                def loss_inner(agent_params, proxy_params_fixed):
                    # Merge params for forward pass
                    full_params = agent_params.copy()
                    full_params['proxy'] = proxy_params_fixed
                    
                    # Compute Intrinsic Reward
                    a_one_hot = jax.nn.one_hot(batch.actions, config["MAX_ACTION_SPACE"])
                    _, q_vals, r_in = network.apply(full_params, init_hs, batch.obs[:,None], batch.dones[:,None], a_one_hot[:,None])
                    
                    # Total Reward
                    r_total = batch.rewards + (config["LIIR_COEFF"] * r_in.squeeze())
                    
                    # Q-Learning Loss (TD Error)
                    q_chosen = jnp.take_along_axis(q_vals.squeeze(), batch.actions[..., None], axis=-1).squeeze()
                    # Target (simplified, assuming target net is handled outside for brevity)
                    # For real implementation, you'd use target_network_params here
                    loss = jnp.mean((q_chosen - r_total)**2) 
                    return loss

                # 2. Get Gradient of Inner Loss w.r.t Agent Params
                grads_agent = jax.grad(loss_inner)(ts.params, ts.params['proxy'])
                
                # 3. Simulate Update (Lookahead)
                # theta_prime = theta - lr * grad_theta
                # We need to manually apply the update to get the "lookahead" parameters
                # This is the "differentiable optimization step"
                new_agent_params = jax.tree_map(lambda p, g: p - config["LR"] * g, ts.params, grads_agent)
                
                # 4. OUTER LOOP: Maximize Extrinsic Reward using Lookahead Agent
                def loss_outer(proxy_params_train):
                    # Combine updated agent params with trainable proxy params
                    lookahead_params = new_agent_params.copy()
                    lookahead_params['proxy'] = proxy_params_train
                    
                    # Re-evaluate Q-values with new params on the SAME batch (or separate validation batch)
                    # We want the Agent's Q-values to predict the EXTRINSIC return well
                    _, q_vals_prime, _ = network.apply(lookahead_params, init_hs, batch.obs[:,None], batch.dones[:,None])
                    
                    # We want to minimize the error between Q_prime and EXTRINSIC reward only
                    q_chosen_prime = jnp.take_along_axis(q_vals_prime.squeeze(), batch.actions[..., None], axis=-1).squeeze()
                    
                    # Meta-Loss: How bad is the updated agent at predicting pure extrinsic reward?
                    # If Proxy was good, Agent update made it better at predicting extrinsic reward.
                    meta_loss = jnp.mean((q_chosen_prime - batch.rewards)**2)
                    return meta_loss

                # 5. Update Proxy
                grads_proxy = jax.grad(loss_outer)(ts.params['proxy'])
                updates_proxy, new_proxy_opt = proxy_tx.update(grads_proxy, ts.proxy_opt_state)
                new_proxy_params = optax.apply_updates(ts.params['proxy'], updates_proxy)

                # 6. Real Update for Agent (using the original gradients from step 2)
                updates_agent, new_agent_opt = agent_tx.update(grads_agent, ts.opt_state) # Note: assumes ts has opt_state
                new_params = optax.apply_updates(ts.params, updates_agent)
                
                # Replace the proxy part of new_params with the meta-updated proxy
                new_params['proxy'] = new_proxy_params
                
                # Create new train state
                new_ts = ts.replace(params=new_params, proxy_opt_state=new_proxy_opt)
                return new_ts, rng

            # Execute Learn Phase
            train_state, rng = jax.lax.scan(_learn_phase, (train_state, rng), None, length=config["NUM_UPDATES_PER_STEP"])[0]

            return (train_state, buffer_state, env_state, last_obs, last_dones, hs, rng), None

        # Init loop variables
        rng, init_rng = jax.random.split(rng)
        init_obs, init_env_state = wrapped_env.reset(init_rng)
        init_dones = jnp.zeros((len(env.agents),), dtype=bool)
        init_hs = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], len(env.agents), 1)
        
        runner_state = (train_state, buffer_state, init_env_state, 
                        jnp.stack([init_obs[a] for a in env.agents]), 
                        jnp.stack([init_dones[a] for a in env.agents]), 
                        init_hs, rng)

        runner_state, _ = jax.lax.scan(_update_step, runner_state, None, length=config["NUM_UPDATES"])
        return runner_state

    return train