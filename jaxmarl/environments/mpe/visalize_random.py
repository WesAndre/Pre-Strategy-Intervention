import jax
from jaxmarl import make
from jaxmarl.environments.mpe import MPEVisualizer
from jaxmarl.wrappers.baselines import PrePolicyWrapper

def visualize_random_policy(
    num_agents: int = 5,
    num_landmarks: int = 5,
    max_steps: int = 50,
    save_gif: bool = True,
    gif_name: str = "mpe_random.gif",
    dual_target: bool = False,
):
    """Visualize random policy on AugmentedMPE"""
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    # Create environment
    env = make("AugmentedMPE", 
               num_agents=num_agents, 
               num_landmarks=num_landmarks,
               train_pre_policy=True,
               if_augment_obs=True,
               intrinsic_reward_ratio=0.2,
               zero_index=True,
               dual_target=dual_target)
    
    obs, state = env.reset(key_r)

    state_seq = []
    for step in range(max_steps):
        state_seq.append(state)
        
        # Sample random actions
        key, key_s, key_a = jax.random.split(key, 3)
        key_a = jax.random.split(key_a, env.num_agents)
        actions = {agent: env.action_space(agent).sample(key_a[i]) 
                   for i, agent in enumerate(env.agents)}

        # Step environment
        obs, state, rewards, dones, infos = env.step(key_s, state, actions)
        
        if dones["__all__"]:
            break

    print(f"Collected {len(state_seq)} states")

    # to determine which agents and landmarks to highlight
    if dual_target:
        highlight_agents = [0, num_agents - 1]  # highlight first and last agent
        highlight_landmarks = [0, num_landmarks - 1]  # highlight first and last landmark
    else:
        highlight_agents = [0]  # highlight first agent
        highlight_landmarks = [0]  # highlight first landmark
    
    viz = MPEVisualizer(env, state_seq, highlight_agents=highlight_agents, highlight_landmarks=highlight_landmarks)
    
    if save_gif:
        viz.animate(save_fname=gif_name, view=False)
        print(f"✓ Saved to {gif_name}")
    else:
        viz.animate(view=True)

if __name__ == "__main__":
    # Compare single vs dual target with random policy
    print("Visualizing with SINGLE target...")
    visualize_random_policy(num_agents=5, num_landmarks=5, 
                           gif_name="mpe_5agents_single.gif", 
                           dual_target=False)
    
    print("\nVisualizing with DUAL target...")
    visualize_random_policy(num_agents=5, num_landmarks=5, 
                           gif_name="mpe_5agents_dual.gif", 
                           dual_target=True)