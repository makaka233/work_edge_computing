import unittest

from edge_drl.agents.deployment import TorchAgentDDeployer
from edge_drl.agents.networks import AgentDActorCritic
from edge_drl.config import load_config
from edge_drl.env.environment import EdgeComputingEnv


class AgentDTest(unittest.TestCase):
    def test_agent_d_generates_feasible_deployment_shape(self):
        cfg = load_config("config/default.yaml")
        env = EdgeComputingEnv(cfg)
        obs = env.reset()
        model = AgentDActorCritic(env.state_dim, env.num_services, env.max_stages, env.num_nodes)
        deployer = TorchAgentDDeployer(
            model,
            env.service_stage_mask,
            env.stage_memory,
            env.stage_storage,
            env.memory_capacity,
            env.storage_capacity,
            cfg["simulation"]["max_service_replicas"],
            deterministic=True,
        )
        deployment = deployer.decide(obs)
        self.assertEqual(deployment.shape, (env.num_services, env.max_stages, env.num_nodes))
        violation = env.deployment_violation(deployment)
        self.assertEqual(violation["missing"], 0.0)


if __name__ == "__main__":
    unittest.main()
