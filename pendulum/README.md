# Pendulum: Final RL notebooks 

Run / inspect the notebooks in this order:

1. `01_Pendulum_policy_checkpoints_and_action_level_DPO_diagnostic.ipynb`  
   Policy checkpoints, rollout evaluation, and action-level DPO diagnostic. Use this to explain why local action-level supervision helps Pendulum.

2. `02_Pendulum_pure_trajectory_level_DPO_final.ipynb`  
   Main pure trajectory-level DPO experiment. Use this for the final trajectory-level DPO result in the report.

3. `03_Pendulum_PPO_RLHF_K_sweep_and_plot_data.ipynb`  
   PPO-RLHF reward-model training, K sweep, coefficient sweep, and plot data.

Important: action-level DPO is a diagnostic/ablation, not the main DPO result. The main DPO comparison is in the pure trajectory-level DPO notebook.
