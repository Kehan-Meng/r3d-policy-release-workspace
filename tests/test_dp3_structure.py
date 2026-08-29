from __future__ import annotations

import unittest

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from omegaconf import OmegaConf

from r3d.model.flow_matching.config import FlowMatchingConfig
from r3d.policy.dp3 import DP3


class DP3StructureTest(unittest.TestCase):
    def test_flow_config_schedule(self):
        config = FlowMatchingConfig.from_mapping(
            {
                "consistency_weight": 1.0,
                "consistency_final_weight": 10.0,
                "consistency_schedule": "geometric",
                "consistency_ramp_start_epoch": 200,
                "consistency_ramp_end_epoch": 1400,
                "consistency_schedule_power": 0.5,
            },
            default_time_scale=99,
        )
        self.assertEqual(config.consistency_weight_at(200), 1.0)
        self.assertEqual(config.consistency_weight_at(1400), 10.0)
        self.assertGreater(config.consistency_weight_at(800), 1.0)
        self.assertLess(config.consistency_weight_at(800), 10.0)

    def test_builders_preserve_top_level_state_dict_paths(self):
        shape_meta = OmegaConf.create({
            "obs": {
                "point_cloud": {"shape": [16, 3], "type": "point_cloud"},
                "agent_pos": {"shape": [4], "type": "low_dim"},
            },
            "action": {"shape": [3]},
        })
        policy = DP3(
            shape_meta=shape_meta,
            noise_scheduler=DDIMScheduler(num_train_timesteps=10),
            horizon=4,
            n_action_steps=2,
            n_obs_steps=2,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=16,
            down_dims=(16, 32),
            n_groups=4,
            encoder_output_dim=16,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg={"embed_dim": 8, "feature_mode": "global"},
            use_act=False,
            use_text=False,
            generation_type="diffusion",
        )

        keys = list(policy.state_dict())
        self.assertTrue(any(key.startswith("obs_encoder.") for key in keys))
        self.assertTrue(any(key.startswith("model.") for key in keys))
        self.assertFalse(any(key.startswith("conditioner.") for key in keys))
        self.assertFalse(any(key.startswith("builder.") for key in keys))

        policy.normalizer.fit({
            "point_cloud": torch.randn(8, 2, 16, 3),
            "agent_pos": torch.randn(8, 2, 4),
            "action": torch.randn(8, 4, 3),
        }, last_n_dims=1)
        loss, metrics = policy.compute_loss({
            "obs": {
                "point_cloud": torch.randn(2, 2, 16, 3),
                "agent_pos": torch.randn(2, 2, 4),
            },
            "action": torch.randn(2, 4, 3),
        })
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("bc_loss", metrics)
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in policy.model.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
