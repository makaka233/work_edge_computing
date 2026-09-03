import subprocess
import sys
import csv
import math
import numpy as np
import pytest
import torch
from argparse import Namespace
from pathlib import Path

from edge_drl.agents.drl import HierarchicalPPOAgent
from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig

from train_dual_ppo import (
    _cpu_byte_rng_state,
    apply_pressure_profile,
    apply_training_design,
    demand_seed_for_training_rollout,
    demand_profile_summary,
    effective_replicas_per_stage,
    fast_counterfactual_credit_coefficient,
    finite_weighted_mean,
    load_assignments_for_update,
    load_multiplier_for_rollout,
    load_adjusted_rolling_latency,
    SlowLearningRateController,
    parse_args,
    load_checkpoint,
    rollout_start_minute,
    scenario_seed_for_offset,
    save_checkpoint,
    TrainingRandomStreams,
    use_deterministic_fast_collection,
)


def test_checkpoint_rng_state_is_canonicalized_for_cpu_generator():
    original = torch.get_rng_state()
    noncanonical = original.to(dtype=torch.int64)

    restored = _cpu_byte_rng_state(noncanonical)

    assert restored.device.type == "cpu"
    assert restored.dtype == torch.uint8
    assert restored.is_contiguous()
    torch.set_rng_state(restored)


def test_checkpoint_atomically_restores_episode_replay_and_resume_metadata(tmp_path):
    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=77,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_minutes=60,
        )
    )
    env.reset()
    source = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    # Use the critic's actual input width rather than coupling the test to a
    # hard-coded observation layout.
    critic_width = source.slow_agent.window_critic.network[0].in_features
    source.slow_agent.critic_replay_states = [np.zeros(critic_width, dtype=np.float32)]
    source.slow_agent.critic_replay_returns = [-1.25]
    source.slow_agent.critic_replay_update_ids = [7]
    source.slow_agent.critic_replay_episode_ids = [13]
    source.slow_agent.critic_episode_index = 14
    source.slow_agent.placement_updates_completed = 9
    source.slow_agent.placement_entropy_current_coef = 0.012
    source.fast_agent.entropy_current_coef = 0.008
    source.slow_agent.count_ppo.buffer.add(
        state=np.zeros(3, dtype=np.float32),
        mask=np.ones(4, dtype=bool),
        action=1,
        logprob=-0.5,
        reward=-1.0,
        done=True,
        value=0.25,
    )
    source.slow_agent.placement_ppo.buffer.add(
        state=np.ones(3, dtype=np.float32),
        mask=np.ones(16, dtype=bool),
        action=2,
        logprob=-0.75,
        reward=-1.0,
        done=True,
        value=0.5,
    )
    source.slow_agent.count_action_returns = [-1.0]
    source.slow_agent.placement_action_returns = [-1.2]
    source.slow_agent.count_action_stage_keys = [(0, 0)]
    source.slow_agent.placement_action_stage_keys = [(0, 0)]
    source.slow_agent.count_window_ids = [0]
    source.slow_agent.placement_window_ids = [0]
    source.slow_agent.window_states = [np.zeros(critic_width, dtype=np.float32)]
    source.slow_agent.window_old_values = [0.1]
    source.slow_agent.window_returns = [-1.1]
    source.slow_agent.window_dones = [True]
    source.slow_agent.window_episode_ids = [14]
    streams = TrainingRandomStreams(2026)
    checkpoint_path = tmp_path / "latest.pt"
    metadata = {
        "update": 7,
        "completed_updates": 7,
        "completed_rollouts": 84,
        "completed_episodes": 14,
    }

    save_checkpoint(source, checkpoint_path, metadata, streams)

    assert checkpoint_path.exists()
    assert not checkpoint_path.with_suffix(".pt.tmp").exists()
    restored = HierarchicalPPOAgent.from_env(env, replicas_per_stage=4)
    restored_metadata = load_checkpoint(
        restored,
        checkpoint_path,
        TrainingRandomStreams(999),
    )
    assert restored_metadata == metadata
    assert restored.slow_agent.critic_replay_episode_ids == [13]
    assert restored.slow_agent.critic_episode_index == 14
    assert restored.slow_agent.placement_updates_completed == 9
    assert np.isclose(restored.slow_agent.placement_entropy_current_coef, 0.012)
    assert np.isclose(restored.fast_agent.entropy_current_coef, 0.008)
    assert restored.completed_slow_windows == 1
    assert restored.slow_agent.count_ppo.buffer.actions == [1]
    assert restored.slow_agent.placement_ppo.buffer.actions == [2]
    assert restored.slow_agent.count_action_stage_keys == [(0, 0)]
    assert restored.slow_agent.placement_action_returns == [-1.2]
    assert np.allclose(restored.slow_agent.window_states[0], 0.0)


def test_policy_stability_defaults_reach_the_training_entrypoint(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_dual_ppo.py"])
    args = parse_args()

    assert args.fast_lr == 2e-4
    assert args.fast_load_balanced_updates is True
    assert args.fast_full_batch_kl_stop is True
    assert args.fast_entropy_coef == 0.001
    assert args.fast_entropy_target == 0.7
    assert args.fast_entropy_max_coef == 0.03
    assert args.fast_entropy_adaptation_rate == 0.002
    assert args.fast_entropy_normalized_control is False
    assert args.fast_counterfactual_credit_final_coef == 0.5
    assert args.slow_count_unused_replica_coef == 0.0
    assert args.slow_count_lr == 2e-4
    assert args.slow_placement_entropy_coef == 0.005
    assert args.slow_placement_entropy_final_coef == 0.0035
    assert args.slow_placement_entropy_hold_updates == 64
    assert args.slow_placement_entropy_decay_updates == 64
    assert args.slow_placement_entropy_max_coef == 0.05
    assert args.slow_placement_entropy_adaptation_rate == 0.002
    assert args.slow_count_global_advantage_coef == 0.25
    assert args.slow_placement_global_advantage_coef == 0.35
    assert args.slow_placement_global_attribution_coef == 0.50
    assert args.slow_global_advantage_ev_full == 0.20
    assert args.slow_critic_lr == 5e-4
    assert args.slow_critic_k_epochs == 8
    assert args.slow_critic_replay_windows == 96
    assert args.slow_critic_replay_decay == 0.90
    assert args.slow_critic_holdout_windows == 12
    assert args.slow_critic_holdout_episodes == 2
    assert args.slow_critic_gradient_clip == 5.0
    assert args.slow_idle_replica_coef == 0.05
    assert args.slow_placement_idle_coef == 0.02
    assert args.episode_minutes == 10
    assert args.episode_hours is None
    assert args.scenario_refresh_episodes == 1
    assert args.demand_scenario_pool_size == 32
    assert args.load_sampling_mode == "distribution-random"
    assert args.demand_scenario_schedule == "stream"
    assert args.training_design == "legacy-alternating"
    assert args.save_best is True
    assert args.best_checkpoint_window == 10
    assert args.checkpoint_interval == 20
    assert args.slow_lr_decay is False
    assert args.slow_lr_decay_patience == 10


def test_checkpoint_score_uses_rolling_load_adjusted_latency():
    history = [
        {"latency_s": 0.8 + 0.4 * load, "load_multiplier": load}
        for load in (0.8, 1.0, 1.2, 1.4)
    ]

    score, slope = load_adjusted_rolling_latency(history, window=3, reference_load=1.1)

    assert np.isclose(slope, 0.4)
    assert np.isclose(score, 1.24)


def test_finite_weighted_mean_handles_empty_request_windows():
    values = np.asarray([0.2, 0.4], dtype=np.float64)

    assert np.isclose(finite_weighted_mean(values, np.zeros(2)), 0.3)
    assert np.isclose(finite_weighted_mean(values, np.asarray([1.0, 3.0])), 0.35)


def test_mec_pressure_profile_scales_demand_and_fixed_capacity():
    args = Namespace(
        pressure_profile="mec-moderate",
        active_user_ratio=0.15,
        active_user_request_rate_per_minute=1.5,
        traffic_scale=1.0,
        load_multipliers="1.0",
        task_compute_scale=1.0,
        task_data_scale=1.0,
        node_compute_capacity_scale=1.0,
        wired_link_bandwidth_scale=1.0,
        service_resource_fraction=0.5,
    )

    apply_pressure_profile(args, argv=["--pressure-profile", "mec-moderate"])

    assert args.active_user_ratio == 0.20
    assert args.active_user_request_rate_per_minute == 1.75
    assert args.task_compute_scale == 1.65
    assert args.task_data_scale == 2.5
    assert args.node_compute_capacity_scale == 0.65
    assert args.wired_link_bandwidth_scale == 0.15
    assert args.service_resource_fraction == 0.25
    assert args.deadline_scale == 2.75
    assert args.load_multipliers == "0.8,1.1,1.4,1.7"
    assert args.load_sampling_mode == "distribution-random"
    assert args.load_strata == "0.75:0.95,0.95:1.20,1.20:1.50,1.50:1.85"
    assert args.load_stratum_probabilities == "0.20,0.50,0.25,0.05"
    assert args.training_design == "trajectory-simultaneous"


def test_realistic_constrained_profile_selects_hardware_nodes_and_compute_pressure():
    args = Namespace(
        pressure_profile="mec-realistic-constrained",
        training_design="legacy-alternating",
        edge_node_profile="synthetic-tiered",
        service_workload_profile="legacy-random",
        active_user_ratio=0.15,
        active_user_request_rate_per_minute=1.5,
        traffic_scale=1.0,
        load_multipliers="1.0",
        load_sampling_mode="cyclic",
        load_strata="",
        load_stratum_probabilities="",
        task_compute_scale=1.0,
        task_data_scale=1.0,
        node_compute_capacity_scale=1.0,
        wired_link_bandwidth_scale=1.0,
        service_resource_fraction=0.5,
        deadline_scale=1.0,
    )

    apply_pressure_profile(args, argv=["--pressure-profile", "mec-realistic-constrained"])

    assert args.edge_node_profile == "hardware-constrained"
    assert args.service_workload_profile == "edge-ai-pipelines"
    assert args.training_design == "trajectory-simultaneous"
    assert args.node_compute_capacity_scale == 1.0
    assert args.wired_link_bandwidth_scale == 0.75
    assert args.service_resource_fraction == 0.60
    assert args.task_compute_scale == 1.0
    assert args.task_data_scale == 1.0
    assert args.load_strata == "0.65:0.80,0.80:1.00,1.00:1.20,1.20:1.45"


def test_realistic_constrained_profile_respects_explicit_node_profile_override():
    args = Namespace(
        pressure_profile="mec-realistic-constrained",
        edge_node_profile="synthetic-tiered",
    )

    apply_pressure_profile(
        args,
        argv=[
            "--pressure-profile",
            "mec-realistic-constrained",
            "--edge-node-profile",
            "synthetic-tiered",
        ],
    )

    assert args.edge_node_profile == "synthetic-tiered"


def test_trajectory_training_design_builds_complete_multi_window_batches():
    args = Namespace(
        training_design="trajectory-simultaneous",
        episode_minutes=10,
        joint_training_schedule="alternating",
        fast_windows_per_update=4,
        slow_windows_per_update=16,
        fast_warmup_updates=4,
        slow_warmup_updates=4,
        fast_counterfactual_credit_coef=0.5,
    )

    apply_training_design(args, argv=[])

    assert args.episode_minutes == 60
    assert args.joint_training_schedule == "simultaneous"
    assert args.fast_windows_per_update == 12
    assert args.slow_windows_per_update == 12
    assert args.fast_warmup_updates == 0
    assert args.slow_warmup_updates == 0
    assert args.slow_count_lr == 1e-4
    assert args.slow_placement_lr == 7.5e-5
    assert args.slow_k_epochs == 2
    assert args.fast_entropy_target == 0.18
    assert args.fast_entropy_max_coef == 0.02
    assert args.fast_entropy_normalized_control is True
    assert args.fast_counterfactual_credit_final_coef == 0.40
    assert args.fast_counterfactual_credit_hold_updates == 80
    assert args.fast_counterfactual_credit_decay_updates == 120
    assert args.slow_count_unused_replica_coef == 0.015
    assert np.isclose(fast_counterfactual_credit_coefficient(args, 80), 0.5)
    assert np.isclose(fast_counterfactual_credit_coefficient(args, 140), 0.45)
    assert np.isclose(fast_counterfactual_credit_coefficient(args, 200), 0.4)
    assert args.slow_placement_entropy_target == 1.10
    assert args.slow_placement_entropy_max_coef == 0.015
    assert args.slow_lr_decay is True
    assert args.slow_lr_decay_patience == 40
    assert args.slow_count_min_lr == 5e-5
    assert args.slow_placement_min_lr == 3.75e-5
    assert args.slow_lr_max_reductions == 1


def test_fast_counterfactual_credit_anneals_after_hold_period():
    args = Namespace(
        fast_counterfactual_credit_coef=0.5,
        fast_counterfactual_credit_final_coef=0.2,
        fast_counterfactual_credit_hold_updates=40,
        fast_counterfactual_credit_decay_updates=60,
    )

    assert np.isclose(fast_counterfactual_credit_coefficient(args, 0), 0.5)
    assert np.isclose(fast_counterfactual_credit_coefficient(args, 40), 0.5)
    assert np.isclose(fast_counterfactual_credit_coefficient(args, 70), 0.35)
    assert np.isclose(fast_counterfactual_credit_coefficient(args, 100), 0.2)
    assert np.isclose(fast_counterfactual_credit_coefficient(args, 200), 0.2)


def test_slow_lr_controller_counts_only_completed_slow_updates():
    env = EdgeComputingEnv(
        EdgeEnvConfig(seed=91, num_users=10_000, num_edge_nodes=8, num_service_types=2)
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env,
        replicas_per_stage=3,
        slow_count_lr=1e-4,
        slow_placement_lr=8e-5,
    )
    controller = SlowLearningRateController(
        enabled=True,
        patience=2,
        factor=0.5,
        min_delta=1e-3,
        min_lr=1e-5,
    )

    assert controller.observe(0.20, ready=True, slow_updated=True, agent=agent) is False
    assert controller.observe(0.21, ready=True, slow_updated=False, agent=agent) is False
    assert controller.bad_updates == 0
    assert controller.observe(0.21, ready=True, slow_updated=True, agent=agent) is False
    assert controller.observe(0.22, ready=True, slow_updated=True, agent=agent) is True
    assert np.isclose(controller.optimizer_lr(agent.slow_agent.count_ppo.optimizer), 5e-5)
    assert np.isclose(controller.optimizer_lr(agent.slow_agent.placement_ppo.optimizer), 4e-5)
    assert controller.reductions == 1


def test_slow_lr_controller_gates_each_actor_by_its_own_kl():
    env = EdgeComputingEnv(
        EdgeEnvConfig(seed=92, num_users=10_000, num_edge_nodes=8, num_service_types=2)
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env,
        replicas_per_stage=3,
        slow_count_lr=1e-4,
        slow_placement_lr=8e-5,
    )
    controller = SlowLearningRateController(
        enabled=True,
        patience=2,
        factor=0.5,
        min_delta=1e-3,
        min_lr=1e-5,
        count_min_lr=5e-5,
        placement_min_lr=4e-5,
        kl_floor_fraction=0.1,
        max_reductions=1,
    )
    observations = dict(
        ready=True,
        slow_updated=True,
        agent=agent,
        count_approx_kl=1e-6,
        placement_approx_kl=0.005,
        count_target_kl=0.01,
        placement_target_kl=0.01,
    )

    assert controller.observe(0.20, **observations) is False
    assert controller.observe(0.21, **observations) is False
    assert controller.observe(0.22, **observations) is True

    assert np.isclose(controller.optimizer_lr(agent.slow_agent.count_ppo.optimizer), 1e-4)
    assert np.isclose(controller.optimizer_lr(agent.slow_agent.placement_ppo.optimizer), 4e-5)
    assert controller.actor_metric("count", "reductions") == 0
    assert controller.actor_metric("placement", "reductions") == 1
    assert controller.actor_metric("count", "low_kl_blocks") == 2


def test_slow_lr_controller_restores_floors_from_older_checkpoint_optimizer():
    env = EdgeComputingEnv(
        EdgeEnvConfig(seed=93, num_users=10_000, num_edge_nodes=8, num_service_types=2)
    )
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env,
        replicas_per_stage=3,
        slow_count_lr=2e-5,
        slow_placement_lr=1e-5,
    )
    controller = SlowLearningRateController(
        enabled=True,
        patience=40,
        factor=0.7,
        min_delta=1e-3,
        min_lr=1e-5,
        count_min_lr=5e-5,
        placement_min_lr=3.75e-5,
    )

    assert controller.enforce_minimum_lrs(agent) is True
    assert np.isclose(controller.optimizer_lr(agent.slow_agent.count_ppo.optimizer), 5e-5)
    assert np.isclose(
        controller.optimizer_lr(agent.slow_agent.placement_ppo.optimizer),
        3.75e-5,
    )
    assert controller.enforce_minimum_lrs(agent) is False


def test_explicit_episode_horizon_survives_training_design_expansion():
    args = Namespace(
        training_design="trajectory-simultaneous",
        episode_minutes=90,
        joint_training_schedule="alternating",
        fast_windows_per_update=4,
        slow_windows_per_update=16,
        fast_warmup_updates=4,
        slow_warmup_updates=4,
    )

    apply_training_design(args, argv=["--episode-minutes", "90"])

    assert args.episode_minutes == 90


def test_trajectory_training_rejects_different_fast_and_slow_batches(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_dual_ppo.py",
            "--training-design",
            "trajectory-simultaneous",
            "--fast-windows-per-update",
            "12",
            "--slow-windows-per-update",
            "24",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_pressure_profile_does_not_override_explicit_scale():
    args = Namespace(
        pressure_profile="mec-moderate",
        active_user_ratio=0.20,
        active_user_request_rate_per_minute=1.75,
        traffic_scale=1.0,
        load_multipliers="0.8,1.1,1.4,1.7",
        load_sampling_mode="stratified-random",
        load_strata="",
        load_stratum_probabilities="1.0",
        task_compute_scale=3.0,
        task_data_scale=2.0,
        node_compute_capacity_scale=0.65,
        wired_link_bandwidth_scale=0.35,
        service_resource_fraction=0.45,
    )

    apply_pressure_profile(
        args,
        argv=["--pressure-profile", "mec-moderate", "--task-compute-scale", "3.0"],
    )

    assert args.task_compute_scale == 3.0
    assert args.task_data_scale == 2.5


def test_pressure_profile_does_not_attach_default_strata_to_custom_load_anchors():
    args = Namespace(
        pressure_profile="mec-moderate",
        active_user_ratio=0.15,
        active_user_request_rate_per_minute=1.5,
        traffic_scale=1.0,
        load_multipliers="0.7,1.3",
        load_sampling_mode="stratified-random",
        load_strata="",
        load_stratum_probabilities="1.0",
        task_compute_scale=1.0,
        task_data_scale=1.0,
        node_compute_capacity_scale=1.0,
        wired_link_bandwidth_scale=1.0,
        service_resource_fraction=0.5,
    )

    apply_pressure_profile(
        args,
        argv=["--pressure-profile", "mec-moderate", "--load-multipliers", "0.7,1.3"],
    )

    assert args.load_multipliers == "0.7,1.3"
    assert args.load_strata == ""


def test_master_seed_derives_reproducible_advancing_random_streams():
    first = TrainingRandomStreams(2026)
    second = TrainingRandomStreams(2026)

    first_scenarios = [first.next_demand_seed() for _ in range(16)]
    second_scenarios = [second.next_demand_seed() for _ in range(16)]
    first_environments = [first.next_environment_seed() for _ in range(16)]
    second_environments = [second.next_environment_seed() for _ in range(16)]

    assert first_scenarios == second_scenarios
    assert first_environments == second_environments
    assert len(set(first_scenarios)) == len(first_scenarios)
    assert len(set(first_environments)) == len(first_environments)
    assert first_scenarios != first_environments


def test_distribution_random_loads_follow_target_probabilities():
    args = Namespace(
        seed=2026,
        load_sampling_mode="distribution-random",
        load_multipliers="0.8,1.1,1.4,1.7",
        load_strata="0.75:0.95,0.95:1.20,1.20:1.50,1.50:1.85",
        load_stratum_probabilities="0.20,0.50,0.25,0.05",
    )
    streams = TrainingRandomStreams(args.seed)

    assignments = load_assignments_for_update(
        args,
        update_idx=0,
        rollouts=20_000,
        rng=streams.load_rng,
    )
    frequencies = [
        sum(group == group_id for _, group in assignments) / len(assignments)
        for group_id in range(4)
    ]

    np.testing.assert_allclose(frequencies, [0.20, 0.50, 0.25, 0.05], atol=0.012)


def test_slow_collection_uses_frozen_stochastic_fast_by_default():
    assert not use_deterministic_fast_collection(True, "stochastic")
    assert use_deterministic_fast_collection(True, "deterministic")
    assert not use_deterministic_fast_collection(False, "deterministic")


def test_scenario_refresh_groups_training_episodes():
    args = Namespace(seed=2026, fixed_scenario=False, scenario_refresh_episodes=20)

    assert scenario_seed_for_offset(args, 0, group_by_refresh=True) == 2026
    assert scenario_seed_for_offset(args, 19, group_by_refresh=True) == 2026
    assert scenario_seed_for_offset(args, 20, group_by_refresh=True) == 2027
    assert scenario_seed_for_offset(args, 20, group_by_refresh=False) == 2046


def test_demand_sampling_mode_controls_training_demand_seed():
    episode_args = Namespace(seed=2026, fixed_scenario=False, scenario_refresh_episodes=20, demand_sampling_mode="episode")
    rollout_args = Namespace(seed=2026, fixed_scenario=False, scenario_refresh_episodes=20, demand_sampling_mode="rollout")

    assert demand_seed_for_training_rollout(episode_args, rollout_idx=19, episode_idx=0) == 2026
    assert demand_seed_for_training_rollout(episode_args, rollout_idx=20, episode_idx=1) == 2026
    assert demand_seed_for_training_rollout(rollout_args, rollout_idx=19, episode_idx=0) == 2045
    assert demand_seed_for_training_rollout(rollout_args, rollout_idx=20, episode_idx=1) == 2046


def test_shuffled_demand_pool_reuses_each_seed_once_per_cycle():
    args = Namespace(
        seed=2026,
        fixed_scenario=False,
        scenario_refresh_episodes=2,
        demand_sampling_mode="episode",
        demand_scenario_schedule="shuffled-pool",
        demand_scenario_pool_size=4,
    )

    first_cycle = [
        demand_seed_for_training_rollout(args, rollout_idx=0, episode_idx=episode)
        for episode in range(0, 8, 2)
    ]
    second_cycle = [
        demand_seed_for_training_rollout(args, rollout_idx=0, episode_idx=episode)
        for episode in range(8, 16, 2)
    ]

    assert sorted(first_cycle) == [2026, 2027, 2028, 2029]
    assert sorted(second_cycle) == [2026, 2027, 2028, 2029]
    assert first_cycle == [
        demand_seed_for_training_rollout(args, rollout_idx=0, episode_idx=episode)
        for episode in range(0, 8, 2)
    ]


def test_demand_profile_summary_exposes_policy_independent_difficulty():
    from edge_drl.env.environment import EdgeComputingEnv, EdgeEnvConfig

    env = EdgeComputingEnv(
        EdgeEnvConfig(
            seed=2026,
            physical_seed=2026,
            scenario_seed=2030,
            num_users=10_000,
            num_edge_nodes=16,
            num_service_types=3,
            episode_hours=1,
        )
    )
    env.reset()

    summary = demand_profile_summary(env)

    assert summary["demand_expected_compute_gcycles"] > 0.0
    assert summary["demand_expected_data_mb"] > 0.0
    assert summary["demand_expected_deadline_s"] > 0.0
    assert summary["demand_service_popularity_entropy"] > 0.0


def test_zero_replica_cap_uses_node_count():
    args = Namespace(replicas_per_stage=0, num_edge_nodes=32)
    assert effective_replicas_per_stage(args) == 32

    explicit_args = Namespace(replicas_per_stage=5, num_edge_nodes=32)
    assert effective_replicas_per_stage(explicit_args) == 5


def test_rollout_load_and_start_modes_cycle():
    args = Namespace(
        seed=2026,
        load_multipliers="1.0,1.5,2.0",
        rollout_start_mode="cycle-window",
        eval_rollout_start_mode="same",
        episode_hours=24,
        deployment_interval_minutes=10,
    )

    assert [load_multiplier_for_rollout(args, idx) for idx in range(5)] == [1.0, 1.5, 2.0, 1.0, 1.5]
    assert [rollout_start_minute(args, idx) for idx in range(7)] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


def test_stratified_random_loads_are_balanced_shuffled_and_reproducible():
    args = Namespace(
        seed=2026,
        load_sampling_mode="stratified-random",
        load_multipliers="0.8,1.1,1.4,1.7",
        load_strata="0.75:0.95,0.95:1.20,1.20:1.50,1.50:1.85",
    )

    first = load_assignments_for_update(args, update_idx=0, rollouts=4)
    repeated = load_assignments_for_update(args, update_idx=0, rollouts=4)
    second = load_assignments_for_update(args, update_idx=1, rollouts=4)

    assert first == repeated
    assert first != second
    assert sorted(group for _, group in first) == [0, 1, 2, 3]
    assert [group for _, group in first] != [0, 1, 2, 3]
    bounds = ((0.75, 0.95), (0.95, 1.20), (1.20, 1.50), (1.50, 1.85))
    assert all(bounds[group][0] <= value <= bounds[group][1] for value, group in first)

    slow_batch = load_assignments_for_update(args, update_idx=2, rollouts=16)
    assert [group for _, group in slow_batch].count(0) == 4
    assert [group for _, group in slow_batch].count(1) == 4
    assert [group for _, group in slow_batch].count(2) == 4
    assert [group for _, group in slow_batch].count(3) == 4

    stationary_args = Namespace(
        seed=2026,
        rollout_start_mode="cycle-window",
        eval_rollout_start_mode="same",
        episode_hours=4,
        deployment_interval_minutes=10,
        arrival_profile="stationary",
    )
    assert [rollout_start_minute(stationary_args, idx) for idx in range(3)] == [0.0, 0.0, 0.0]


def test_dual_ppo_entrypoint_writes_log_and_checkpoint(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--rollout-unit",
        "window",
        "--deployment-interval-minutes",
        "1",
        "--requests-per-update",
        "4",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--mean-requests-per-minute",
        "60",
        "--request-aggregation-window-seconds",
        "1",
        "--compute-hotspot-coef",
        "0.08",
        "--link-hotspot-coef",
        "0.04",
        "--eval-baseline",
        "--eval-requests",
        "4",
        "--save-best",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "baseline requests=" in result.stdout
    assert "update=001" in result.stdout
    assert "artificial_cap=none" in result.stdout
    assert "diag_res=" in result.stdout
    assert "slowR=" in result.stdout
    assert (log_dir / "training.csv").exists()
    assert (log_dir / "episode_metrics.csv").exists()
    assert (log_dir / "rollout_metrics.csv").exists()
    assert (save_dir / "last.pt").exists()
    assert (save_dir / "best.pt").exists()

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert "avg_train_resource_penalty" in rows[0]
    assert "avg_diagnostic_resource_penalty" in rows[0]
    assert "slow_window_return" in rows[0]
    assert "slow_window_count" in rows[0]
    assert "slow_critic_explained_variance" in rows[0]
    assert "slow_window_critic_train_explained_variance" in rows[0]
    assert "slow_window_critic_holdout_explained_variance" in rows[0]
    assert "slow_window_critic_replay_size" in rows[0]
    assert "slow_global_advantage_reliability" in rows[0]
    assert "slow_placement_entropy_coef" in rows[0]
    assert "slow_placement_entropy_next_coef" in rows[0]
    assert "slow_count_global_advantage_coef" in rows[0]
    assert "slow_count_global_advantage_configured_coef" in rows[0]
    assert "slow_placement_global_advantage_coef" in rows[0]
    assert "slow_placement_global_advantage_configured_coef" in rows[0]
    assert "slow_placement_updates_completed" in rows[0]
    assert "avg_service_memory_util" in rows[0]
    assert "active_node_rate" in rows[0]
    assert "hot_link_rate" in rows[0]
    assert "fast_optimizer_steps" in rows[0]
    assert "fast_entropy_coef" in rows[0]
    assert "fast_entropy_next_coef" in rows[0]
    assert "fast_samples_seen_fraction" in rows[0]
    assert "fast_min_group_seen_fraction" in rows[0]
    assert "fast_max_group_approx_kl" in rows[0]
    assert "fast_load_approx_kl" in rows[0]
    assert "slow_count_lr" in rows[0]
    assert "slow_placement_lr" in rows[0]
    assert "slow_lr_reductions" in rows[0]
    assert "slow_count_replica_imbalance_fraction" in rows[0]
    assert "slow_count_replica_efficiency_cost" in rows[0]
    assert "fast_counterfactual_credit_coef" in rows[0]

    with (log_dir / "episode_metrics.csv").open(newline="", encoding="utf-8") as handle:
        episode_rows = list(csv.DictReader(handle))
    assert len(episode_rows) == 1
    episode_row = episode_rows[0]
    assert episode_row["episode"] == "1"
    assert episode_row["update_start"] == "1"
    assert episode_row["update_end"] == "1"
    assert episode_row["rollouts_collected"] == "1"
    assert "avg_reward" in episode_row
    assert "total_reward" in episode_row
    assert "avg_latency_s" in episode_row
    assert "p95_latency_s" in episode_row

    with (log_dir / "rollout_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rollout_rows = list(csv.DictReader(handle))
    assert len(rollout_rows) == 1
    assert rollout_rows[0]["rollout"] == "1"
    assert rollout_rows[0]["rollout_in_update"] == "1"
    assert "avg_reward" in rollout_rows[0]
    assert "total_reward" in rollout_rows[0]
    assert "avg_latency_s" in rollout_rows[0]
    assert "p95_latency_s" in rollout_rows[0]
    assert "slow_count_replica_imbalance_fraction" in rollout_rows[0]
    assert "slow_count_replica_efficiency_cost" in rollout_rows[0]
    assert "fast_counterfactual_credit_coef" in rollout_rows[0]


def test_fast_only_entrypoint_writes_mode_tagged_log(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--train-mode",
        "fast-only",
        "--updates",
        "1",
        "--rollout-unit",
        "requests",
        "--requests-per-update",
        "4",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--mean-requests-per-minute",
        "60",
        "--request-aggregation-window-seconds",
        "1",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "train_mode=fast-only" in result.stdout
    assert (log_dir / "training.csv").exists()
    assert (save_dir / "last.pt").exists()


def test_default_entrypoint_writes_under_runs(tmp_path):
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--train-mode",
        "fast-only",
        "--rollout-unit",
        "requests",
        "--requests-per-update",
        "4",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--mean-requests-per-minute",
        "60",
        "--request-aggregation-window-seconds",
        "1",
        "--run-root",
        str(tmp_path / "runs"),
        "--run-name",
        "organized_run",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "runs" in result.stdout
    assert (tmp_path / "runs" / "organized_run" / "logs" / "training.csv").exists()
    assert (tmp_path / "runs" / "organized_run" / "checkpoints" / "last.pt").exists()
    assert (tmp_path / "runs" / "organized_run" / "metadata.json").exists()


def test_episode_rollout_unit_aligns_update_and_episode(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--demand-scenario-schedule",
        "sequential",
        "--rollout-unit",
        "episode",
        "--mean-requests-per-minute",
        "2",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--deployment-interval-minutes",
        "30",
        "--request-aggregation-window-seconds",
        "1",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "rollout_unit=episode" in result.stdout
    assert "update=001 episode=001 demand_seed=2026-2026 load=1.00-1.00 start_min=0-0 rollouts=1 complete=1" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[-1]
    assert row["update"] == "1"
    assert row["episode"] == "1"
    assert row["episode_complete"] == "1"
    assert float(row["episode_fraction"]) >= 1.0
    assert float(row["deployment_updates"]) >= 2.0


def test_window_rollout_unit_allows_multiple_updates_per_episode(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "2",
        "--demand-scenario-schedule",
        "sequential",
        "--slow-warmup-updates",
        "0",
        "--rollout-unit",
        "window",
        "--mean-requests-per-minute",
        "2",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--deployment-interval-minutes",
        "30",
        "--request-aggregation-window-seconds",
        "1",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "rollout_unit=window" in result.stdout
    assert "update=001 episode=001 demand_seed=2026-2026 load=1.00-1.00 start_min=0-0 rollouts=1 complete=0 window=01" in result.stdout
    assert "update=002 episode=001 demand_seed=2026-2026 load=1.00-1.00 start_min=0-0 rollouts=1 complete=1 window=02" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["update"] for row in rows] == ["1", "2"]
    assert [row["episode"] for row in rows] == ["1", "1"]
    assert [row["rollouts_collected"] for row in rows] == ["1", "1"]
    assert [row["window_in_episode"] for row in rows] == ["1", "2"]
    assert rows[0]["episode_complete"] == "0"
    assert rows[1]["episode_complete"] == "1"
    assert "avg_valid_latency_s" in rows[0]
    assert "avg_penalty_latency_s" in rows[0]
    assert "invalid_action_rate" in rows[0]
    assert "avg_node_compute_load" in rows[0]
    assert "load_multiplier" in rows[0]
    assert "start_minute" in rows[0]
    assert "used_replica_rate" in rows[0]
    assert "avg_replica_use_entropy" in rows[0]
    assert "cross_node_stage_transition_rate" in rows[0]
    assert "max_node_memory_util" in rows[0]

    with (log_dir / "episode_metrics.csv").open(newline="", encoding="utf-8") as handle:
        episode_rows = list(csv.DictReader(handle))
    assert len(episode_rows) == 1
    episode_row = episode_rows[0]
    assert episode_row["episode"] == "1"
    assert episode_row["update_start"] == "1"
    assert episode_row["update_end"] == "2"
    assert episode_row["rollouts_collected"] == "2"
    assert episode_row["window_start"] == "1"
    assert episode_row["window_end"] == "2"
    assert episode_row["episode_complete"] == "1.0"
    assert float(episode_row["requests"]) == sum(float(row["requests"]) for row in rows)


def test_window_rollout_demand_sampling_resets_each_update(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "2",
        "--slow-warmup-updates",
        "0",
        "--rollout-unit",
        "window",
        "--demand-sampling-mode",
        "rollout",
        "--demand-scenario-schedule",
        "sequential",
        "--mean-requests-per-minute",
        "2",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--deployment-interval-minutes",
        "30",
        "--request-aggregation-window-seconds",
        "1",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "demand_sampling_mode=rollout" in result.stdout
    assert "update=001 episode=001 demand_seed=2026-2026 load=1.00-1.00 start_min=0-0 rollouts=1 complete=0 window=01" in result.stdout
    assert "update=002 episode=002 demand_seed=2027-2027 load=1.00-1.00 start_min=0-0 rollouts=1 complete=0 window=01" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["episode"] for row in rows] == ["1", "2"]
    assert [row["demand_seed"] for row in rows] == ["2026", "2027"]
    assert [row["demand_seed_end"] for row in rows] == ["2026", "2027"]
    assert [row["window_in_episode"] for row in rows] == ["1", "1"]


def test_rollouts_per_update_batches_independent_demand_samples(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--slow-warmup-updates",
        "0",
        "--rollout-unit",
        "window",
        "--demand-sampling-mode",
        "rollout",
        "--demand-scenario-schedule",
        "sequential",
        "--rollouts-per-update",
        "2",
        "--mean-requests-per-minute",
        "60",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--request-aggregation-window-seconds",
        "1",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "fast_windows_per_update=2" in result.stdout
    assert "slow_windows_per_update=32" in result.stdout
    assert "episode_horizon=10min deployment_windows=1" in result.stdout
    assert "arrival_profile=stationary" in result.stdout
    assert "update=001 episodes=001-002 demand_seed=2026-2027 load=1.00-1.00 start_min=0-0 rollouts=2 complete=1 window=01" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["rollouts_collected"] == "2"
    assert rows[-1]["episodes_collected"] == "2"
    assert rows[-1]["episode_start"] == "1"
    assert rows[-1]["episode_end"] == "2"
    assert rows[-1]["demand_seed"] == "2026"
    assert rows[-1]["demand_seed_end"] == "2027"
    with (log_dir / "rollout_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rollout_rows = list(csv.DictReader(handle))
    assert [row["rollout"] for row in rollout_rows] == ["1", "2"]
    assert [row["rollout_in_update"] for row in rollout_rows] == ["1", "2"]
    assert [row["demand_seed"] for row in rollout_rows] == ["2026", "2027"]


def test_ten_minute_episode_mode_advances_episode_and_demand_pool(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--slow-warmup-updates",
        "0",
        "--rollout-unit",
        "window",
        "--demand-sampling-mode",
        "episode",
        "--demand-scenario-schedule",
        "sequential",
        "--fast-windows-per-update",
        "2",
        "--mean-requests-per-minute",
        "2",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "episode_horizon=10min deployment_windows=1" in result.stdout
    assert "update=001 episodes=001-002 demand_seed=2026-2027" in result.stdout
    assert "rollouts=2 complete=1 window=01" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        row = list(csv.DictReader(handle))[-1]
    assert row["episode_start"] == "1"
    assert row["episode_end"] == "2"
    assert row["episodes_collected"] == "2"
    assert row["episode_complete"] == "1"
    assert row["demand_seed"] == "2026"
    assert row["demand_seed_end"] == "2027"

    with (log_dir / "episode_metrics.csv").open(newline="", encoding="utf-8") as handle:
        episode_rows = list(csv.DictReader(handle))
    assert [episode_row["episode"] for episode_row in episode_rows] == ["1", "2"]
    assert [episode_row["demand_seed"] for episode_row in episode_rows] == ["2026", "2027"]
    assert all(episode_row["update_start"] == "1" for episode_row in episode_rows)
    assert all(episode_row["update_end"] == "1" for episode_row in episode_rows)
    assert all(episode_row["rollouts_collected"] == "1" for episode_row in episode_rows)
    assert all(episode_row["episode_complete"] == "1.0" for episode_row in episode_rows)
    weighted_latency = sum(
        float(episode_row["avg_latency_s"]) * float(episode_row["requests"])
        for episode_row in episode_rows
    ) / sum(float(episode_row["requests"]) for episode_row in episode_rows)
    assert math.isclose(weighted_latency, float(row["avg_latency_s"]))


def test_ten_minute_slow_episode_flushes_exactly_one_window(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--fast-warmup-updates",
        "0",
        "--slow-warmup-updates",
        "1",
        "--slow-windows-per-update",
        "1",
        "--episode-minutes",
        "10",
        "--sampled-seconds-per-window",
        "2",
        "--mean-requests-per-minute",
        "2",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--max-replicas-per-stage",
        "2",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        row = list(csv.DictReader(handle))[-1]
    assert row["training_phase"] == "slow_warmup"
    assert row["episode_complete"] == "1"
    assert row["slow_updated"] == "1"
    assert row["slow_window_count"] == "1.0"
    assert row["slow_placement_updates_completed"] == "1"


def test_rollouts_per_update_batches_pressure_levels_and_start_windows(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--rollout-unit",
        "window",
        "--demand-sampling-mode",
        "rollout",
        "--rollouts-per-update",
        "2",
        "--load-multipliers",
        "1.0,1.5",
        "--load-sampling-mode",
        "cyclic",
        "--rollout-start-mode",
        "cycle-window",
        "--arrival-profile",
        "daily",
        "--mean-requests-per-minute",
        "2",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--deployment-interval-minutes",
        "30",
        "--request-aggregation-window-seconds",
        "1",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "load_multipliers=1.0,1.5" in result.stdout
    assert "rollout_start_mode=cycle-window" in result.stdout
    assert "load=1.00-1.50 start_min=0-30" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["load_multiplier"] == "1.0"
    assert rows[-1]["load_multiplier_end"] == "1.5"
    assert rows[-1]["start_minute"] == "0.0"
    assert rows[-1]["start_minute_end"] == "30.0"


def test_eval_interval_does_not_run_initial_eval_by_default(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--train-mode",
        "fast-only",
        "--rollout-unit",
        "requests",
        "--requests-per-update",
        "4",
        "--eval-interval",
        "1",
        "--eval-requests",
        "4",
        "--eval-seeds",
        "1",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--episode-hours",
        "1",
        "--mean-requests-per-minute",
        "60",
        "--request-aggregation-window-seconds",
        "1",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "update=000" not in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["update"] for row in rows] == ["1"]
    assert rows[0]["eval_avg_latency_s"] != "nan"
    assert rows[0]["seen_eval_avg_latency_s"] == "nan"


def test_training_samples_window_but_eval_runs_full_window(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "1",
        "--slow-warmup-updates",
        "0",
        "--rollout-unit",
        "window",
        "--deployment-interval-minutes",
        "10",
        "--sampled-seconds-per-window",
        "6",
        "--eval-interval",
        "1",
        "--eval-rollout-unit",
        "window",
        "--eval-seeds",
        "1",
        "--mean-requests-per-minute",
        "60",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "temporal_sampling train=6/600s (1.0%) eval=full" in result.stdout
    assert "sampled_steps=6/600" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["settlement_steps"] == "6"
    assert row["logical_steps"] == "600"
    assert float(row["temporal_sampling_fraction"]) == 0.01
    assert float(row["eval_settlement_steps"]) == 600.0
    assert float(row["eval_logical_steps"]) == 600.0
    assert float(row["eval_temporal_sampling_fraction"]) == 1.0


def test_fast_and_slow_ppo_use_independent_window_update_periods(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "3",
        "--rollout-unit",
        "window",
        "--fast-windows-per-update",
        "1",
        "--slow-windows-per-update",
        "2",
        "--joint-training-schedule",
        "simultaneous",
        "--episode-hours",
        "1",
        "--deployment-interval-minutes",
        "1",
        "--mean-requests-per-minute",
        "60",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "slow_update=0 slow_buffer=1/2" in result.stdout
    assert "slow_update=1 slow_buffer=0/2" in result.stdout

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["slow_updated"] for row in rows] == ["0", "1", "0"]
    assert [row["slow_windows_available"] for row in rows] == ["1", "2", "1"]
    assert [row["slow_windows_buffered"] for row in rows] == ["1", "0", "1"]
    assert [row["slow_window_count"] for row in rows] == ["0.0", "2.0", "0.0"]


def test_alternating_schedule_freezes_one_controller_per_phase(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "3",
        "--rollout-unit",
        "window",
        "--fast-windows-per-update",
        "1",
        "--slow-windows-per-update",
        "2",
        "--joint-training-schedule",
        "alternating",
        "--fast-updates-per-cycle",
        "2",
        "--fast-warmup-updates",
        "0",
        "--slow-warmup-updates",
        "0",
        "--episode-hours",
        "1",
        "--deployment-interval-minutes",
        "1",
        "--sampled-seconds-per-window",
        "5",
        "--mean-requests-per-minute",
        "60",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "16",
        "--num-service-types",
        "3",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["training_phase"] for row in rows] == ["fast", "fast", "slow"]
    assert float(rows[0]["fast_loss"]) != 0.0
    assert float(rows[1]["fast_loss"]) != 0.0
    assert float(rows[2]["fast_loss"]) == 0.0
    assert [row["slow_updated"] for row in rows] == ["0", "0", "1"]
    assert float(rows[2]["slow_count_return_std"]) > 0.0


def test_alternating_schedule_runs_fast_then_slow_warmup(tmp_path):
    log_dir = tmp_path / "logs"
    save_dir = tmp_path / "savedModels"
    command = [
        sys.executable,
        "train_dual_ppo.py",
        "--updates",
        "5",
        "--rollout-unit",
        "window",
        "--fast-windows-per-update",
        "1",
        "--slow-windows-per-update",
        "1",
        "--joint-training-schedule",
        "alternating",
        "--fast-updates-per-cycle",
        "2",
        "--fast-warmup-updates",
        "2",
        "--slow-warmup-updates",
        "2",
        "--episode-hours",
        "1",
        "--deployment-interval-minutes",
        "1",
        "--sampled-seconds-per-window",
        "1",
        "--mean-requests-per-minute",
        "600",
        "--num-users",
        "10000",
        "--num-edge-nodes",
        "8",
        "--num-service-types",
        "2",
        "--progress-interval-seconds",
        "0",
        "--log-dir",
        str(log_dir),
        "--save-dir",
        str(save_dir),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    with (log_dir / "training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["training_phase"] for row in rows] == [
        "fast_warmup",
        "fast_warmup",
        "slow_warmup",
        "slow_warmup",
        "fast",
    ]
    assert [row["slow_updated"] for row in rows] == ["0", "0", "1", "1", "0"]
    assert any(float(row["fast_loss"]) != 0.0 for row in rows[:2])
    assert [float(row["fast_loss"]) == 0.0 for row in rows[2:4]] == [True, True]
    assert float(rows[4]["fast_loss"]) != 0.0
    assert "slow_count_effective_replicas_per_stage" in rows[0]
    assert "slow_count_redundant_replica_fraction" in rows[0]
    assert "slow_count_advantage_std" in rows[0]


def test_convergence_analyzer_reads_training_log(tmp_path):
    log_path = tmp_path / "training.csv"
    log_path.write_text(
        "update,requests,avg_latency_s,eval_avg_latency_s,invalid_actions\n"
        "1,10,2.0,nan,0\n"
        "2,10,1.6,nan,0\n"
        "3,10,1.2,nan,0\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "scripts/analyze_convergence.py", str(log_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Convergence report" in result.stdout
    assert "verdict=" in result.stdout
