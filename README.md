# A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning

This repository contains the official JAX implementation for the NeurIPS 2025 paper: [A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2510.17697).

Our work introduces a principle for designing targeted interventions that guide multi-agent systems toward desirable outcomes, such as improved coordination and performance, without explicitly programming complex behaviors.

## Table of Contents
- [Installation](#installation)
- [Running Experiments](#running-experiments)
  - [Hanabi Environment](#hanabi-environment)
  - [MPE Environment](#mpe-environment)
- [Visualization of Learned Behavior](#visualization-of-learned-behavior)
- [Citation](#citation)
- [License and Acknowledgements](#license-and-acknowledgements)

## Installation

This project requires Python 3.10 and we recommend `conda` for environment management. The installation is a **two-stage process**: first, you install the hardware-specific JAX library, and second, you install this project's dependencies.

1.  **Clone the repository:**
    ```shell
    git clone https://github.com/iamlilAJ/Pre-Strategy-Intervention.git
    cd Pre-Strategy-Intervention
    ```

2.  **Create and activate the conda environment:**
    ```shell
    conda create -n intervention python=3.10 -y
    conda activate intervention
    ```

3.  **Install JAX for your specific hardware:**

    * **For NVIDIA GPU Users (Recommended):**
        This command installs the exact versions of JAX, a CUDA-enabled jaxlib, and cuDNN that are compatible with this project. The `-f` flag is crucial as it directs `pip` to the official JAX repository to find the GPU-specific packages.
        ```shell
        pip install jax==0.4.25 jaxlib==0.4.25 nvidia-cudnn-cu12==8.9.2.26 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        ```

    * **For CPU-Only Users:**
        If you do not have an NVIDIA GPU, install the CPU-only version of JAX.
        ```shell
        pip install jax==0.4.25 jaxlib==0.4.25
        ```

4.  **Install the project and dependencies:**
    Now that JAX is correctly installed, you can install the rest of the project's dependencies. This command uses the `[algs]` extra to include packages like `optax` and `wandb`.
    ```shell
    pip install -e .[algs]
    ```


## Running Extensions

Here are commands on how to run each of the four extensions.

# 5-agent Hanabi
* **PSI:**
    ```shell
    python baselines/IPPO/ippo_pre.py +alg=ippo alg.ENV_KWARGS.num_agents=5
    ```
* **IR:**
    ```shell
    python baselines/IPPO/ippo_pre.py +alg=intrinsic_reward_ippo alg.ENV_KWARGS.num_agents=5
    ```

# X-agent MPE
Replace X in the following command with the amount of agents

* **PSI:**
    ```shell
    python baselines/QLearning/vdn_pre.py +alg=vdn ++alg.ENV_KWARGS.num_agents=X ++alg.ENV_KWARGS.num_landmarks=X ++alg.ENV_KWARGS.dual_target=True
    ```
* **IR:**
    ```shell
    python baselines/QLearning/vdn_pre.py +alg=intrinsic_reward_vdn ++alg.ENV_KWARGS.num_agents=X ++alg.ENV_KWARGS.num_landmarks=X ++alg.ENV_KWARGS.dual_target=True
    ```

So, for example, running PSI with 50 agents would need the following command:

    python baselines/QLearning/vdn_pre.py +alg=vdn ++alg.ENV_KWARGS.num_agents=50 ++alg.ENV_KWARGS.num_landmarks=50 ++alg.ENV_KWARGS.dual_target=True

# Extrinsic vs Intrinsic return ratio in loss function
To change the weight given to the intrinsic return alter the X in the following command:

* **IR:**
    ```shell
    python baselines/IPPO/ippo_pre.py +alg=intrinsic_reward_ippo alg.ENV_KWARGS.num_agents=X
    ```
The default value is 0.035. In our experiments we used: 0.0, 0.035, 0.075 and 0.2

# Longer training
In this experiment we doubled the number of training steps. To run this use the following commands.

* **PSI:**
    ```shell
    python baselines/IPPO/ippo_pre.py +alg=ippo alg.TOTAL_TIMESTEPS=2e9
    ```
* **IR:**
    ```shell
    python baselines/IPPO/ippo_pre.py +alg=intrinsic_reward_ippo alg.TOTAL_TIMESTEPS=2e9
    ```

From this point the rest of the readme follows the readme of the forked github repo:
## Running Experiments
To reproduce the main results from our paper, run our **Pre-Strategy Intervention** method against the **Standard MARL** and **Intrinsic Reward** baselines described below. All experiments are managed via command-line arguments using Hydra.

Our experiments are organized around three main conditions which can be applied to most algorithms. You can select the condition by modifying the Hydra configuration name (`+alg=...`).

* **Pre-Strategy Intervention (Our Method):** Use the base algorithm name.
    * Example: `+alg=ippo`
* **Standard MARL Baseline:** Add the `base_marl_` prefix to the algorithm name.
    * Example: `+alg=base_marl_ippo`
* **Intrinsic Reward Baseline:** Add the `intrinsic_reward_` prefix to the algorithm name.
    * Example: `+alg=intrinsic_reward_ippo`

Below are the base commands for each supported algorithm and environment. Simply apply the prefixes described above to run the desired baseline.

### Hanabi Environment

* **IPPO:**
    ```shell
    python baselines/IPPO/ippo_pre.py +alg=ippo
    ```
* **MAPPO:**
    ```shell
    python baselines/MAPPO/mappo_pre.py +alg=mappo
    ```
* **PQN-VDN:**
    ```shell
    python baselines/QLearning/pqn_vdn_pre.py +alg=pqn
    ```
* **PQN-IQL:**
    ```shell
    python baselines/QLearning/pqn_iql_pre.py +alg=pqn
    ```

#### 4-Player Version in Hanabi
You can change the number of players by overriding the `num_agents` parameter.

* **To run PQN-VDN with 4 players:**
    ```shell
    python baselines/QLearning/pqn_vdn_pre.py +alg=pqn alg.ENV_KWARGS.num_agents=4
    ```

#### Global Pre-Strategy Intervention (GPSI)
You can change the intervention scope by overriding the `intervene_two_agents` parameter.  For example:
 ```shell
  python baselines/QLearning/pqn_vdn_pre.py +alg=pqn alg.ENV_KWARGS.intervene_two_agents=True
```

     

### MPE Environment

* **IQL:**
    ```shell
    python baselines/QLearning/iql_pre.py +alg=iql
    ```
    *To run the second IQL scenario:*
    ```shell
    python baselines/QLearning/iql_pre.py +alg=iql_scenario_2
    ```
* **VDN:**
    ```shell
    python baselines/QLearning/vdn_pre.py +alg=vdn
    ```
* **QMIX:**
    ```shell
    python baselines/QLearning/qmix_pre.py +alg=qmix
    ```

#### Heterogeneous MPE Setting
This special setting tests our method with heterogeneous agents, where one agent is a significantly faster "sprinter."

* **Intervening on the second agent:**
    ```shell
    python baselines/QLearning/iql_pre.py +alg=heter_iql
    ```
* **To change which agent is the sprinter to the targeted agent):**
    You can override the `accel` parameter on the command line.
    ```shell
    python baselines/QLearning/iql_pre.py +alg=heter_iql alg.ENV_KWARGS.accel='[25.0, 5.0, 5.0]'
    ```
* **To run the baseline for this setting:**
    ```shell
    python baselines/QLearning/iql_pre.py +alg=baseline_heter_iql
    ```


## Visualization of Learned Behavior

| Our Method                                                   | Baseline                                                       |
| :----------------------------------------------------------- | :------------------------------------------------------------- |
| ![MPE Visualization 1](assets/MPE_visualization_1.gif)       | ![Baseline 1](assets/MPE_visualization_baseline_1.gif)         |
| ![MPE Visualization 2](assets/MPE_visualization_2.gif)       | ![Baseline 2](assets/MPE_visualization_baseline_2.gif)         |

In this visualization, the red agent (our intervened agent) has learned a preference for moving towards the yellow landmark. By learning this simple additional desired outcome, the agent team can achieve effective coordination and successfully solve the task.

## Citation

If you use this work in your research, please cite the following paper.

**BibTeX:**
```latex
@misc{liu2025principle,
    title={A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning},
    author={Anjie Liu and Jianhong Wang and Samuel Kaski and Jun Wang and Mengyue Yang},
    year={2025},
    eprint={2510.17697},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

## License and Acknowledgements

This project is licensed under the Apache 2.0 License. 

Our implementation is built upon the excellent [JaxMARL](https://github.com/FLAIROx/JaxMARL) library. We thank the original authors for their significant contributions to the community.
