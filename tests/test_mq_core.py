from __future__ import annotations

import unittest

import torch

from r3d.model.act.act_former import AffordanceGuidedCompactorTransformer
from r3d.model.act.attention import HeatmapGuidedCrossAttention
from r3d.model.diffusion.diffusion_backbone import ConditionalUnet1D
from r3d.model.flow_matching import compute_consistency_flow_matching_loss


class TestMQCore(unittest.TestCase):
    def test_heatmap_attention_is_normalized_and_differentiable(self):
        module = HeatmapGuidedCrossAttention(
            query_dim=16,
            context_dim=16,
            num_heads=4,
            heatmap_mode="multiply",
            heatmap_gamma=1.0,
        )
        queries = torch.randn(2, 5, 16, requires_grad=True)
        context = torch.randn(2, 11, 16, requires_grad=True)
        heatmap = torch.rand(2, 11, 1)

        output, attention = module(queries, context, heatmap=heatmap)

        self.assertEqual(output.shape, (2, 5, 16))
        self.assertEqual(attention.shape, (2, 4, 5, 11))
        torch.testing.assert_close(
            attention.sum(dim=-1),
            torch.ones(2, 4, 5),
            atol=2e-6,
            rtol=2e-6,
        )
        output.square().mean().backward()
        self.assertIsNotNone(queries.grad)
        self.assertIsNotNone(context.grad)
        self.assertGreater(float(queries.grad.norm()), 0.0)
        self.assertGreater(float(context.grad.norm()), 0.0)

    def test_act_compacts_to_requested_query_count(self):
        module = AffordanceGuidedCompactorTransformer(
            token_dim=16,
            pe_dim=8,
            num_queries=4,
            num_heads=4,
            heatmap_mode="multiply",
            drop_cls_token=False,
        )
        patch_tokens = torch.randn(2, 12, 16, requires_grad=True)
        point_pe = torch.randn(2, 12, 8)
        heatmap = torch.rand(2, 12, 1)

        compact, compact_pe = module(patch_tokens, point_pe, heatmap=heatmap)

        self.assertEqual(compact.shape, (2, 4, 16))
        self.assertEqual(compact_pe.shape, (2, 4, 8))
        compact.mean().backward()
        self.assertGreater(float(patch_tokens.grad.norm()), 0.0)

    def test_one_way_flow_reader_forward_backward(self):
        model = ConditionalUnet1D(
            input_dim=4,
            local_cond_dim=None,
            global_cond_dim=8,
            diffusion_step_embed_dim=8,
            down_dims=(8, 16),
            kernel_size=3,
            n_groups=4,
            condition_type="one_way_transformer",
            transformer_config={
                "embedding_dim": 8,
                "depth": 1,
                "num_heads": 2,
                "mlp_dim": 16,
                "max_n_obs_steps": 2,
                "max_horizon": 4,
            },
        )
        sample = torch.randn(2, 4, 4, requires_grad=True)
        condition = torch.randn(2, 6, 8, requires_grad=True)
        point_pe = torch.randn(2, 6, 8)

        output = model(
            sample,
            torch.tensor([1.0, 2.0]),
            global_cond=condition,
            pc_pe=point_pe,
            n_obs_steps=2,
        )

        self.assertEqual(output.shape, sample.shape)
        output.square().mean().backward()
        self.assertGreater(float(sample.grad.norm()), 0.0)
        self.assertGreater(float(condition.grad.norm()), 0.0)

    def test_flow_training_objective_forward_backward(self):
        model = ConditionalUnet1D(
            input_dim=4,
            local_cond_dim=None,
            global_cond_dim=8,
            diffusion_step_embed_dim=8,
            down_dims=(8, 16),
            kernel_size=3,
            n_groups=4,
            condition_type="one_way_transformer",
            transformer_config={
                "embedding_dim": 8,
                "depth": 1,
                "num_heads": 2,
                "mlp_dim": 16,
                "max_n_obs_steps": 2,
                "max_horizon": 4,
            },
        )
        trajectory = torch.randn(2, 4, 4)
        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        condition = torch.randn(2, 6, 8, requires_grad=True)
        point_pe = torch.randn(2, 6, 8)

        loss, logs = compute_consistency_flow_matching_loss(
            model,
            trajectory,
            condition_mask,
            global_cond=condition,
            pc_pe=point_pe,
            n_obs_steps=2,
            consistency_weight=2.0,
            direct_velocity_weight=1.0,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("flow_f_loss", logs)
        self.assertIn("flow_direct_velocity_loss", logs)
        loss.backward()
        self.assertGreater(float(condition.grad.norm()), 0.0)

if __name__ == "__main__":
    unittest.main()
