import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import sys
import importlib
import inspect
import pathlib
import numpy as np

from pytorch3d.ops import sample_farthest_points
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint


def _installed_pointsam_root() -> str:
    """Return the root of the installed editable or wheel PointSAM package."""
    try:
        import pc_sam
    except ImportError as exc:
        raise ImportError(
            "PointSAM is not installed. Run `pip install -e ./PointSAM`."
        ) from exc
    return str(pathlib.Path(pc_sam.__file__).resolve().parent.parent)


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs
                 ):
        """Initialize PointNet encoder for XYZ+RGB point clouds.

        Args:
            in_channels (int): Feature size of input (3 or 6).
            out_channels (int): Output feature dimension.
            use_layernorm (bool): Whether to use LayerNorm after each MLP layer.
            final_norm (str): Normalization after final projection ('layernorm' or 'none').
            use_projection (bool): Whether to apply the final projection layer.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x, eval):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs
                 ):
        """Initialize PointNet encoder for XYZ-only point clouds.

        Args:
            in_channels (int): Feature size of input (must be 3).
            out_channels (int): Output feature dimension.
            use_layernorm (bool): Whether to use LayerNorm after each MLP layer.
            final_norm (str): Normalization after final projection ('layernorm' or 'none').
            use_projection (bool): Whether to apply the final projection layer.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

    def forward(self, x, eval):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class DP3Encoder(nn.Module):
    def __init__(self,
                 observation_space: Dict,
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 fps_random_config=None,
                 cat_on_token=False
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        state_mlp_size = (64, pointcloud_encoder_cfg['embed_dim'])

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None

        ignored_obs_keys = {self.point_cloud_key, self.rgb_image_key, self.imagination_key}
        self.low_dim_keys = [key for key in observation_space.keys() if key not in ignored_obs_keys]
        if len(self.low_dim_keys) == 0:
            raise RuntimeError("DP3Encoder requires at least one low-dimensional observation key")
        self.low_dim_shapes = {key: observation_space[key] for key in self.low_dim_keys}

        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] low-dim keys: {self.low_dim_keys}", "yellow")
        cprint(f"[DP3Encoder] low-dim shapes: {self.low_dim_shapes}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        feature_mode = pointcloud_encoder_cfg.get('feature_mode', None)
        self.pc_encoder_extract_global_feature = feature_mode != 'pointsam'
        if pointnet_type == "pointsam":
            self.pc_encoder_extract_global_feature = False
        elif pointnet_type == "pointsam_heatmap":
            self.pc_encoder_extract_global_feature = False
        elif pointnet_type in ("pointsam_heatmap_bridge", "pointsam_heatmap_bridge_v4"):
            self.pc_encoder_extract_global_feature = False
        elif pointnet_type in ("pointsam_twowayCA", "pointsam_twowayCA_v4"):
            self.pc_encoder_extract_global_feature = False
        cprint(f"[DP3Encoder] extract_global_feature: {self.pc_encoder_extract_global_feature}", "yellow")

        self.fps_random_config = fps_random_config or {
            'use_random': True,
            'random_start': True,
            'random_noise_scale': 0,
            'shuffle_output': True
        }

        self.cat_on_token = cat_on_token
        pc_output_dim = self.n_output_channels

        if pointnet_type == "pointnet":
            pointnet_cfg = dict(pointcloud_encoder_cfg)
            pointnet_cfg.setdefault('out_channels', out_channel)
            if use_pc_color:
                pointnet_cfg['in_channels'] = 6
                self.extractor = PointNetEncoderXYZRGB(**pointnet_cfg)
            else:
                pointnet_cfg['in_channels'] = 3
                self.extractor = PointNetEncoderXYZ(**pointnet_cfg)
        elif pointnet_type == "uni3d":
            cprint(f"[DP3Encoder] Using Uni3D encoder", "yellow")
            uni3d_config = {
                'pc_model': 'eva02_large_patch14_448',
                'pc_feat_dim': 1024,
                'embed_dim': out_channel,
                'group_size': 32,
                'num_group': 512,
                'patch_dropout': 0.5,
                'drop_path_rate': 0.2,
                'pretrained_pc': None,
                'pc_encoder_dim': 512,
                'use_pretrained_weights': False,
                'pretrained_weights_path': None,
            }
            if pointcloud_encoder_cfg:
                uni3d_config.update(pointcloud_encoder_cfg)
            uni3d_config['fps_random_config'] = self.fps_random_config
            self.extractor = Uni3DPointcloudEncoder(**uni3d_config)
            pc_output_dim = uni3d_config['embed_dim']
        elif pointnet_type == "uni3d_pretrained":
            cprint(f"[DP3Encoder] Using pretrained Uni3D encoder", "yellow")
            uni3d_config = {
                'pc_model': 'eva02_large_patch14_448',
                'pc_feat_dim': 1024,
                'embed_dim': out_channel,
                'group_size': 32,
                'num_group': 512,
                'patch_dropout': 0.5,
                'drop_path_rate': 0.2,
                'pretrained_pc': None,
                'pc_encoder_dim': 512,
                'use_pretrained_weights': True,
                'pretrained_weights_path': 'Uni3D_large/model.pt',
            }
            if pointcloud_encoder_cfg:
                uni3d_config.update(pointcloud_encoder_cfg)
            uni3d_config['fps_random_config'] = self.fps_random_config
            self.extractor = Uni3DPointcloudEncoder(**uni3d_config)
            pc_output_dim = uni3d_config['embed_dim']

        elif pointnet_type == "pointsam":
            cprint(f"[DP3Encoder] Using external Point-SAM encoder", "yellow")
            pointsam_config = dict(pointcloud_encoder_cfg)
            pointsam_config.setdefault('embed_dim', out_channel)
            encoder_variant = pointsam_config.pop('encoder_variant', None)
            use_v2_encoder = (
                encoder_variant == "v2"
                or bool(pointsam_config.get("use_heatmap_model", False))
            )
            encoder_cls = PointSAMPointcloudEncoderV2 if use_v2_encoder else PointSAMPointcloudEncoder
            cprint(f"[DP3Encoder] PointSAM encoder class: {encoder_cls.__name__}", "yellow")
            self.extractor = encoder_cls(**pointsam_config)
            pc_output_dim = pointsam_config.get('out_dim', pointsam_config['embed_dim'])

        elif pointnet_type == "pointsam_heatmap":
            cprint(f"[DP3Encoder] Using PointSAM heatmap encoder", "yellow")
            pointsam_config = dict(pointcloud_encoder_cfg)
            pointsam_config.setdefault('embed_dim', out_channel)
            self.extractor = PointSAMHeatmapPointcloudEncoder(**pointsam_config)
            pc_output_dim = pointsam_config.get('out_dim', pointsam_config['embed_dim'])

        elif pointnet_type in ("pointsam_heatmap_bridge", "pointsam_heatmap_bridge_v4"):
            if pointnet_type == "pointsam_heatmap_bridge_v4":
                cprint(f"[DP3Encoder] Using PointSAMHeatmapBridgeV4", "yellow")
            else:
                cprint(f"[DP3Encoder] Using PointSAMHeatmapBridge", "yellow")
            from r3d.model.vision.pointsam_heatmap_bridge import (
                PointSAMHeatmapBridge,
                PointSAMHeatmapBridgeV4,
            )
            pointsam_config = dict(pointcloud_encoder_cfg)
            pointsam_config.setdefault('embed_dim', out_channel)
            encoder_cls = (
                PointSAMHeatmapBridgeV4
                if pointnet_type == "pointsam_heatmap_bridge_v4"
                else PointSAMHeatmapBridge
            )
            self.extractor = encoder_cls(**pointsam_config)
            # pc_embed_dim is the actual ViT output dim (1024 for Uni3D-Ti);
            # embed_dim (256) is the pc_pe dim only.
            pc_output_dim = getattr(self.extractor, 'pc_embed_dim',
                                    pointsam_config.get('embed_dim', out_channel))

        elif pointnet_type in ("pointsam_twowayCA", "pointsam_twowayCA_v4"):
            if pointnet_type == "pointsam_twowayCA_v4":
                cprint(f"[DP3Encoder] Using PointSAMTwoWayCAV4", "yellow")
            else:
                cprint(f"[DP3Encoder] Using PointSAMTwoWayCA", "yellow")
            from r3d.model.vision.pointsam_twowayCA import (
                PointSAMTwoWayCA,
                PointSAMTwoWayCAV4,
            )
            pointsam_config = dict(pointcloud_encoder_cfg)
            pointsam_config.setdefault('embed_dim', out_channel)
            encoder_cls = (
                PointSAMTwoWayCAV4
                if pointnet_type == "pointsam_twowayCA_v4"
                else PointSAMTwoWayCA
            )
            self.extractor = encoder_cls(**pointsam_config)
            pc_output_dim = getattr(self.extractor, 'pc_embed_dim',
                                    pointsam_config.get('embed_dim', out_channel))

        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.low_dim_mlps = nn.ModuleDict()
        for key in self.low_dim_keys:
            shape = self.low_dim_shapes[key]
            if len(shape) != 1:
                raise RuntimeError(f"Low-dimensional obs '{key}' must be rank-1, got {shape}")
            self.low_dim_mlps[key] = nn.Sequential(
                *create_mlp(shape[0], output_dim, net_arch, state_mlp_activation_fn)
            )

        if self.cat_on_token:
            self.n_output_channels = pc_output_dim
        else:
            self.n_output_channels = pc_output_dim + output_dim * len(self.low_dim_keys)

        cprint(f"[DP3Encoder] Final output dim: {self.n_output_channels}", "yellow")

    def forward(
            self,
            observations: Dict,
            eval=False,
            text=None,
            return_heatmap=False,
            return_pc_embedding=False) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]]
            points = torch.concat([points, img_points], dim=1)

        if self.pointnet_type in ["uni3d", "uni3d_pretrained"]:
            if points.shape[-1] == 3:
                colors = torch.zeros_like(points)
                points = torch.cat([points, colors], dim=-1)
            elif points.shape[-1] > 6:
                points = points[..., :6]

        if not self.pc_encoder_extract_global_feature:
            if return_heatmap:
                pn_feat, pc_pe, heatmap = self.extractor(
                    points, eval, text=text, return_heatmap=True
                )
            else:
                pn_feat, pc_pe = self.extractor(points, eval)
                heatmap = None
        else:
            pn_feat = self.extractor(points, eval)
            heatmap = None

        final_feat = self.attach_low_dim_features(pn_feat, observations)
        if not self.pc_encoder_extract_global_feature:
            if return_heatmap:
                if return_pc_embedding:
                    return final_feat, pc_pe, heatmap, pn_feat
                return final_feat, pc_pe, heatmap
            if return_pc_embedding:
                return final_feat, pc_pe, pn_feat
            return final_feat, pc_pe
        return final_feat

    def attach_low_dim_features(self, pn_feat: torch.Tensor, observations: Dict) -> torch.Tensor:
        """Attach trainable low-dimensional observation features to cached point tokens."""
        low_dim_features = []
        for key in self.low_dim_keys:
            low_dim_value = observations[key]
            low_dim_feat = self.low_dim_mlps[key](low_dim_value)
            if not self.pc_encoder_extract_global_feature:
                if self.cat_on_token:
                    low_dim_feat = low_dim_feat.unsqueeze(1)
                else:
                    low_dim_feat = low_dim_feat.unsqueeze(1).expand(-1, pn_feat.shape[1], -1)
            low_dim_features.append(low_dim_feat)

        features = [pn_feat] + low_dim_features
        if self.cat_on_token:
            final_feat = torch.cat(features, dim=-2)
        else:
            final_feat = torch.cat(features, dim=-1)
        return final_feat

    def output_shape(self):
        return self.n_output_channels


# =============================================================================
# PointSAM encoder components (adapted from Uni3D)
# =============================================================================

def fps(data, number, use_random=True, random_start=True, random_noise_scale=0, shuffle_output=True):
    '''
    Enhanced FPS with randomness options
    Args:
        data: B N 3 (or more channels)
        number: int, number of points to sample
        use_random: bool, whether to enable randomness
        random_start: bool, whether to use random starting point
        random_noise_scale: float, scale of random noise added to distances
        shuffle_output: bool, whether to randomly shuffle the output order
    '''
    xyz_coordinates = data[:, :, :3]
    B, N, _ = xyz_coordinates.shape
    
    if not use_random:
        # Original deterministic FPS
        _, fps_idx = sample_farthest_points(xyz_coordinates, K=number)
    else:
        # Enhanced FPS with randomness
        if random_start:
            # Randomly select starting points for each batch
            start_indices = torch.randint(0, N, (B,), device=data.device)
            
            # Create modified coordinates with random starting points moved to front
            modified_xyz = xyz_coordinates.clone()
            for b in range(B):
                start_idx = start_indices[b]
                # Swap the randomly selected point to the first position
                modified_xyz[b, [0, start_idx]] = modified_xyz[b, [start_idx, 0]]
        else:
            modified_xyz = xyz_coordinates
        
        if random_noise_scale > 0:
            # Add small random noise to coordinates for FPS computation
            noise = torch.randn_like(modified_xyz) * random_noise_scale
            noisy_xyz = modified_xyz + noise
        else:
            noisy_xyz = modified_xyz
        
        # Perform FPS on modified/noisy coordinates
        _, fps_idx = sample_farthest_points(noisy_xyz, K=number)
        
        # If we used random start, we need to map back the indices
        if random_start:
            for b in range(B):
                start_idx = start_indices[b]
                # Map indices back to original positions
                mask_0 = fps_idx[b] == 0
                mask_start = fps_idx[b] == start_idx
                fps_idx[b][mask_0] = start_idx
                fps_idx[b][mask_start] = 0
        
        if shuffle_output:
            # Randomly shuffle the order of selected indices
            for b in range(B):
                perm = torch.randperm(number, device=data.device)
                fps_idx[b] = fps_idx[b][perm]
    
    # Gather the selected points using the (possibly randomized) indices
    fps_data = torch.gather(
        data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    
    return fps_data

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    B, N, _ = batch_pc.shape
    result = torch.clone(batch_pc)
    for b in range(B):
        dropout_ratio = torch.rand(1).item() * max_dropout_ratio  # 0 ~ 0.875
        drop_idx = torch.where(torch.rand(N) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            result[b, drop_idx, :] = batch_pc[b, 0, :].unsqueeze(0)  # set to the first point
    return result


class PatchDropout(nn.Module):
    """
    Patch dropout for Uni3D
    https://arxiv.org/abs/2212.00794
    """
    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x

class KNNGrouper(nn.Module):
    """Group points based on K nearest neighbors.

    A number of points are sampled as centers by farthest point sampling (FPS).
    Each group is formed by the center and its k nearest neighbors.
    """

    def __init__(self, num_groups, group_size, radius=None, centralize_features=False, fps_random_config=None):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.radius = radius
        self.centralize_features = centralize_features
        self.fps_random_config = fps_random_config or {}
        cprint(f"[Group] FPS randomness config: {fps_random_config}", "cyan")

    def forward(self, xyz: torch.Tensor, features: torch.Tensor, use_fps=True):
        """
        Args:
            xyz: [B, N, 3]. Input point clouds.
            features: [B, N, C]. Point features.
            use_fps: bool. Whether to use farthest point sampling.
                If not, `xyz` should already be sampled by FPS.

        Returns:
            dict: {
                features: [B, G, K, 3 + C]. Group features.
                centers: [B, G, 3]. Group centers.
                knn_idx: [B, G, K]. The indices of k nearest neighbors.
            }
        """
        batch_size, num_points, _ = xyz.shape
        with torch.no_grad():
            centers = fps(xyz, self.num_groups, **self.fps_random_config) # B G 3
            _, knn_idx = knn_points(centers, xyz, self.group_size)  # [B, G, K]

        batch_offset = torch.arange(batch_size, device=xyz.device) * num_points
        batch_offset = batch_offset.reshape(-1, 1, 1)
        knn_idx_flat = (knn_idx + batch_offset).reshape(-1)  # [B * G * K]

        nbr_xyz = xyz.reshape(-1, 3)[knn_idx_flat]
        nbr_xyz = nbr_xyz.reshape(batch_size, self.num_groups, self.group_size, 3)
        nbr_xyz = nbr_xyz - centers.unsqueeze(2)  # [B, G, K, 3]
        # NOTE: Follow PointNext to normalize the relative position
        if self.radius is not None:
            nbr_xyz = nbr_xyz / self.radius

        nbr_feats = features.reshape(-1, features.shape[-1])[knn_idx_flat]
        nbr_feats = nbr_feats.reshape(
            batch_size, self.num_groups, self.group_size, features.shape[-1]
        )

        group_feats = torch.cat([nbr_xyz, nbr_feats], dim=-1)
        return dict(
            features=group_feats, centers=centers, knn_idx=knn_idx
        )

class PatchEncoder(nn.Module):
    """Encode point patches following the PointNet structure for segmentation."""

    def __init__(self, in_channels, out_channels, hidden_dims: list[int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # NOTE: The original Uni3D implementation uses BatchNorm1d, while we use LayerNorm.
        self.conv1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dims[0] * 2, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], out_channels),
        )

    def forward(self, point_patches: torch.Tensor):
        # point_patches: [B, L, K, C_in]
        x = self.conv1(point_patches)
        y = torch.max(x, dim=-2, keepdim=True).values
        x = torch.cat([y.expand_as(x), x], dim=-1)
        x = self.conv2(x)  # [B, L, K, C_out]
        y = torch.max(x, dim=-2).values  # [B, L, C_out]
        return y

class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_patches,
        patch_size,
        radius: float = None,
        centralize_features=False,
        fps_random_config=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.grouper = KNNGrouper(
            num_patches,
            patch_size,
            radius=radius,
            centralize_features=centralize_features,
            fps_random_config=fps_random_config
        )

        self.patch_encoder = PatchEncoder(in_channels, out_channels, [128, 512])
        self.fps_random_config = fps_random_config or {}

    def forward(self, coords: torch.Tensor, features: torch.Tensor):
        patches = self.grouper(coords, features)
        patch_features = patches["features"]  # [B, L, K, C_in]
        x = self.patch_encoder(patch_features)
        patches["embeddings"] = x
        return patches


def knn_points(
    query: torch.Tensor,
    key: torch.Tensor,
    k: int,
    sorted: bool = False,
    transpose: bool = False,
):
    """Compute k nearest neighbors.

    Args:
        query: [B, N1, D], query points. [B, D, N1] if @transpose is True.
        key:  [B, N2, D], key points. [B, D, N2] if @transpose is True.
        k: the number of nearest neighbors.
        sorted: whether to sort the results
        transpose: whether to transpose the last two dimensions.

    Returns:
        torch.Tensor: [B, N1, K], distances to the k nearest neighbors in the key.
        torch.Tensor: [B, N1, K], indices of the k nearest neighbors in the key.
    """
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    # Compute pairwise distances, [B, N1, N2]
    distance = torch.cdist(query, key)
    if k == 1:
        knn_dist, knn_ind = torch.min(distance, dim=2, keepdim=True)
    else:
        knn_dist, knn_ind = torch.topk(distance, k, dim=2, largest=False, sorted=sorted)
    return knn_dist, knn_ind


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [-1,1]."""
        # assuming coords are in [-1, 1] and have d_1 x ... x d_n x D shape
        coords = coords @ self.positional_encoding_gaussian_matrix
        # TODO: Why using 2 * np.pi?
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: shape (..., coord_dim), normalized coordinates in [-1, 1].

        Returns:
            torch.Tensor: shape (..., num_pos_feats), positional encoding.
        """
        if (coords < -1 - 1e-6).any() or (coords > 1 + 1e-6).any():
            print("Bounds: ", (coords.min(), coords.max()))
            raise ValueError(f"Input coordinates must be normalized to [-1, 1].")
        # TODO: whether to convert to float?
        return self._pe_encoding(coords)

# PointSAM heatmap integration helpers
class DisabledPointSAMMoEFEB(nn.Module):
    def forward(self, low_image_embeddings):
        return 0.0


class CenterTextSimilarityHeatmapDecoder(nn.Module):
    """PointSAM V2 heatmap decoder evaluated directly on patch-center tokens."""

    def __init__(
            self,
            transformer_dim: int,
            pc_hidden_dim: int = None,
            text_hidden_dim: int = None,
            normalize: bool = True,
            temperature: float = 0.07):
        super().__init__()
        import math

        self.transformer_dim = transformer_dim
        self.normalize = normalize
        pc_hidden_dim = pc_hidden_dim or transformer_dim * 4
        text_hidden_dim = text_hidden_dim or transformer_dim * 4

        self.pc_mlp = nn.Sequential(
            nn.Linear(transformer_dim, pc_hidden_dim),
            nn.GELU(),
            nn.Linear(pc_hidden_dim, transformer_dim),
        )
        self.text_mlp = nn.Sequential(
            nn.Linear(transformer_dim, text_hidden_dim),
            nn.GELU(),
            nn.Linear(text_hidden_dim, transformer_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / temperature))

    def forward(self, pc_embeddings: torch.Tensor, text_tokens: torch.Tensor, **kwargs):
        point_tokens = self.pc_mlp(pc_embeddings)
        text_tokens = self.text_mlp(text_tokens)

        if self.normalize:
            point_tokens = F.normalize(point_tokens, dim=-1)
            text_tokens = F.normalize(text_tokens, dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        return (text_tokens @ point_tokens.transpose(-1, -2)) * logit_scale


class PointSAMPointcloudEncoderV2(nn.Module):
    """V2 adapter for using PointSAM heatmap components inside R3D/DP3.

    R3D's one_way_transformer branch expects patch tokens and a positional
    encoding for the patch centers. Point-SAM's pc_encoder returns patch tokens
    plus patch metadata. V2 can also load the text encoder and heatmap decoder
    from a PointSAM heatmap checkpoint and return patch-center heatmaps.
    """

    _POINTSAM_BUILDER_KWARGS = {
        'drop_path_rate',
        'patch_dropout',
        'num_group',
        'group_size',
        'pc_encoder_dim',
        'moe_feb_num_experts',
        'moe_feb_assignment_factor',
        'moe_feb_num_heads',
        'moe_feb_residual_weight',
        'use_esam_adapter',
        'esam_adapter_mlp_ratio',
        'esam_adapter_scale',
        'use_moe_feb',
    }

    def __init__(self,
                 pointsam_root=None,
                 pointsam_builder='build_uni3d_b_encoder_for_sam',
                 embed_dim=256,
                 out_dim=None,
                 use_pretrained_weights=False,
                 pretrained_weights_path=None,
                 encoder_pretrained_weights_path=None,
                 checkpoint_key_prefix='pc_encoder.',
                 strict_load=False,
                 ignore_mismatched_sizes=True,
                 freeze=False,
                 freeze_uni3d=False,
                 freeze_out_proj=False,
                 use_moe_feb=False,
                 use_esam_adapter=None,
                 esam_adapter_mlp_ratio=None,
                 esam_adapter_scale=None,
                 use_heatmap_model=False,
                 heatmap_config_dir=None,
                 heatmap_config_name='large_heatmap_v2',
                 heatmap_checkpoint_path=None,
                 heatmap_as_prob=True,
                 heatmap_query='patch',
                 **kwargs):
        super().__init__()

        self.embed_dim = out_dim or embed_dim
        self.pe_layer = PositionEmbeddingRandom(self.embed_dim // 2)
        self.heatmap_model = None
        self.heatmap_text_encoder = None
        self.heatmap_decoder = None
        self.heatmap_as_prob = heatmap_as_prob
        if heatmap_query not in ('patch', 'points'):
            raise ValueError(
                "heatmap_query must be either 'patch' or 'points', "
                f"got {heatmap_query}"
            )
        self.heatmap_query = heatmap_query
        self.supports_heatmap = False
        self.freeze_heatmap_components = freeze

        if pointsam_root:
            pointsam_root = os.path.abspath(os.path.expanduser(pointsam_root))
            extra_paths = [
                pointsam_root,
                os.path.join(pointsam_root, 'third_party', 'Pointnet2_PyTorch', 'pointnet2_ops_lib'),
                os.path.join(pointsam_root, 'third_party', 'torkit3d'),
            ]
            for extra_path in extra_paths:
                if not os.path.exists(extra_path):
                    continue
                if extra_path in sys.path:
                    sys.path.remove(extra_path)
                    sys.path.insert(0, extra_path)
                else:
                    sys.path.insert(0, extra_path)

            loaded_pc_sam = sys.modules.get('pc_sam')
            loaded_path = getattr(loaded_pc_sam, '__file__', None)
            if loaded_path is not None:
                loaded_path = os.path.abspath(loaded_path)
                if not loaded_path.startswith(pointsam_root):
                    for module_name in list(sys.modules.keys()):
                        if module_name == 'pc_sam' or module_name.startswith('pc_sam.'):
                            del sys.modules[module_name]

        else:
            pointsam_root = _installed_pointsam_root()

        if use_heatmap_model:
            self._build_heatmap_model(
                pointsam_root=pointsam_root,
                config_dir=heatmap_config_dir,
                config_name=heatmap_config_name,
                checkpoint_path=heatmap_checkpoint_path,
                strict=strict_load,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                encoder_overrides={
                    'use_moe_feb': use_moe_feb,
                    'use_esam_adapter': use_esam_adapter,
                    'esam_adapter_mlp_ratio': esam_adapter_mlp_ratio,
                    'esam_adapter_scale': esam_adapter_scale,
                    **{
                        key: kwargs.get(key)
                        for key in self._POINTSAM_BUILDER_KWARGS
                        if key.startswith('moe_feb_')
                    },
                },
            )
        else:
            module = importlib.import_module('pc_sam.model.uni3d_point_encoder')
            builder = getattr(module, pointsam_builder)

            builder_kwargs = {
                key: kwargs[key]
                for key in self._POINTSAM_BUILDER_KWARGS
                if key in kwargs and kwargs[key] is not None
            }
            builder_kwargs['use_moe_feb'] = use_moe_feb
            if use_esam_adapter is not None:
                builder_kwargs['use_esam_adapter'] = use_esam_adapter
            if esam_adapter_mlp_ratio is not None:
                builder_kwargs['esam_adapter_mlp_ratio'] = esam_adapter_mlp_ratio
            if esam_adapter_scale is not None:
                builder_kwargs['esam_adapter_scale'] = esam_adapter_scale

            self.encoder = builder(out_dim=self.embed_dim, **builder_kwargs)
            cprint(
                f"[PointSAMPointcloudEncoder] builder={pointsam_builder}, out_dim={self.embed_dim}",
                "cyan"
            )

        if not use_moe_feb and hasattr(self.encoder, 'moe_feb'):
            self.encoder.moe_feb = DisabledPointSAMMoEFEB()
            cprint("[PointSAMPointcloudEncoder] MoE-FEB disabled", "yellow")

        encoder_checkpoint_path = encoder_pretrained_weights_path
        if encoder_checkpoint_path is None and use_pretrained_weights:
            encoder_checkpoint_path = pretrained_weights_path

        if use_pretrained_weights and encoder_checkpoint_path:
            self._load_pretrained_weights(
                encoder_checkpoint_path,
                checkpoint_key_prefix=checkpoint_key_prefix,
                strict=strict_load,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )

        if freeze:
            freeze_modules = [self.encoder, self.heatmap_text_encoder, self.heatmap_decoder]
            for module in freeze_modules:
                if module is None:
                    continue
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()
            cprint(f"[{self.__class__.__name__}] froze PointSAM components", "yellow")
        elif freeze_uni3d:
            for name, param in self.encoder.named_parameters():
                keep_trainable = (
                    name.startswith('out_proj.')
                    or name.startswith('moe_feb.')
                    or name.startswith('post_vit_mlp.')
                    or '.MLP_Adapter.' in name
                    or '.Space_Adapter.' in name
                )
                param.requires_grad = keep_trainable
            cprint(
                "[PointSAMPointcloudEncoder] froze Uni3D backbone, kept out_proj/adapters trainable",
                "yellow"
            )

        if freeze_out_proj and hasattr(self.encoder, 'out_proj'):
            for param in self.encoder.out_proj.parameters():
                param.requires_grad = False
            cprint("[PointSAMPointcloudEncoder] froze out_proj", "yellow")

    def _build_heatmap_model(
        self,
        pointsam_root,
        config_dir,
        config_name,
        checkpoint_path,
        strict,
        ignore_mismatched_sizes=False,
        encoder_overrides=None,
    ):
        config_dir = config_dir or os.path.join(pointsam_root, 'configs')
        config_dir = os.path.abspath(os.path.expanduser(config_dir))
        checkpoint_path = (
            self._resolve_checkpoint_path(checkpoint_path)
            if checkpoint_path is not None else None
        )

        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf, open_dict
        import hydra

        global_hydra = GlobalHydra.instance()
        if global_hydra.is_initialized():
            # R3D train.py already runs inside Hydra. PointSAM uses a separate
            # config root, so clear the active Hydra context before composing it.
            global_hydra.clear()
        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = hydra.compose(config_name=config_name, overrides=[])
            OmegaConf.resolve(cfg)

        with open_dict(cfg.model.pc_encoder):
            for key, value in (encoder_overrides or {}).items():
                if value is not None:
                    cfg.model.pc_encoder[key] = value

        self.encoder = hydra.utils.instantiate(cfg.model.pc_encoder)
        self.heatmap_text_encoder = hydra.utils.instantiate(cfg.model.text_prompt_encoder)

        decoder_cfg = cfg.model.mask_decoder
        try:
            self.heatmap_decoder = hydra.utils.instantiate(decoder_cfg)
        except Exception as exc:
            cprint(
                "[PointSAMPointcloudEncoder] failed to instantiate mask_decoder "
                f"target from config ({exc}); falling back to local cosine decoder",
                "yellow",
            )
            self.heatmap_decoder = CenterTextSimilarityHeatmapDecoder(
                transformer_dim=decoder_cfg.transformer_dim,
                pc_hidden_dim=decoder_cfg.get("pc_hidden_dim", None),
                text_hidden_dim=decoder_cfg.get("text_hidden_dim", None),
                normalize=decoder_cfg.get("normalize", True),
                temperature=decoder_cfg.get("temperature", 0.07),
            )

        state_dict = self._read_checkpoint(checkpoint_path) if checkpoint_path else None
        if state_dict is not None:
            self._load_prefixed_module(
                self.encoder,
                state_dict,
                "pc_encoder.",
                strict,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )
            self._load_prefixed_module(
                self.heatmap_text_encoder,
                state_dict,
                "text_prompt_encoder.",
                strict,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )
            self._load_prefixed_module(
                self.heatmap_decoder,
                state_dict,
                "mask_decoder.",
                strict,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )
        else:
            cprint(
                "[PointSAMPointcloudEncoder] no heatmap checkpoint provided; "
                "using initialized pc encoder / text projection / heatmap decoder weights",
                "yellow",
            )

        self.heatmap_model = nn.ModuleDict({
            "text_encoder": self.heatmap_text_encoder,
            "decoder": self.heatmap_decoder,
        })
        self.embed_dim = getattr(self.encoder, 'embed_dim', self.embed_dim)
        self.pe_layer = PositionEmbeddingRandom(self.embed_dim // 2)
        self.supports_heatmap = True
        cprint(
            f"[PointSAMPointcloudEncoder] built PointSAM heatmap components "
            f"decoder={self.heatmap_decoder.__class__.__name__}, "
            f"checkpoint={checkpoint_path}, encoder out_dim={self.embed_dim}",
            "cyan",
        )

    def _load_prefixed_module(
        self,
        module,
        state_dict,
        prefix,
        strict,
        ignore_mismatched_sizes=False,
    ):
        if state_dict is None:
            if strict:
                raise RuntimeError(f"Missing checkpoint state_dict for strict load of {prefix}")
            cprint(f"[PointSAMPointcloudEncoder] skipped {prefix}: no checkpoint", "yellow")
            return None

        module_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not module_state:
            if strict:
                raise RuntimeError(f"No checkpoint keys matched prefix {prefix}")
            cprint(f"[PointSAMPointcloudEncoder] skipped {prefix}: no matched keys", "yellow")
            return None

        skipped = []
        if ignore_mismatched_sizes and not strict:
            current_state = module.state_dict()
            filtered_state = {}
            for key, value in module_state.items():
                current_value = current_state.get(key)
                if current_value is not None and current_value.shape != value.shape:
                    skipped.append((key, tuple(value.shape), tuple(current_value.shape)))
                    continue
                filtered_state[key] = value
            module_state = filtered_state

        result = module.load_state_dict(module_state, strict=strict)
        cprint(
            f"[PointSAMPointcloudEncoder] loaded {prefix} "
            f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}",
            "yellow" if result.missing_keys or result.unexpected_keys else "cyan",
        )
        if skipped:
            cprint(f"  Skipped incompatible {prefix} keys: {skipped[:10]}", "yellow")
        return result

    def _resolve_checkpoint_path(self, checkpoint_path):
        checkpoint_path = os.path.abspath(os.path.expanduser(str(checkpoint_path)))
        if os.path.isdir(checkpoint_path):
            for filename in ('model.safetensors', 'pytorch_model.bin', 'model.pt', 'checkpoint.pt'):
                candidate = os.path.join(checkpoint_path, filename)
                if os.path.exists(candidate):
                    return candidate
        return checkpoint_path

    def _read_checkpoint(self, checkpoint_path):
        checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            cprint(f"[PointSAMPointcloudEncoder] checkpoint not found: {checkpoint_path}", "red")
            return None

        if checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            return load_file(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        for key in ('state_dict', 'model', 'module'):
            if isinstance(checkpoint, dict) and key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
        return checkpoint

    def _extract_encoder_state_dict(self, state_dict, checkpoint_key_prefix):
        if not isinstance(state_dict, dict):
            raise RuntimeError("Point-SAM checkpoint must be a state_dict-like mapping")

        prefixes = checkpoint_key_prefix
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        prefixes = list(prefixes or [])
        prefixes.extend(['module.pc_encoder.', 'model.pc_encoder.', 'pc_encoder.', 'point_encoder.'])

        for prefix in prefixes:
            if not prefix:
                continue
            matched = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if matched:
                return matched, prefix

        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            if key.startswith('point_encoder.'):
                key = key[len('point_encoder.'):]
            if key == 'logit_scale':
                continue
            cleaned[key] = value
        return cleaned, '<raw>'

    def _load_pretrained_weights(self, checkpoint_path, checkpoint_key_prefix, strict, ignore_mismatched_sizes):
        state_dict = self._read_checkpoint(checkpoint_path)
        if state_dict is None:
            return

        encoder_state_dict, used_prefix = self._extract_encoder_state_dict(
            state_dict,
            checkpoint_key_prefix=checkpoint_key_prefix,
        )

        skipped = []
        if ignore_mismatched_sizes and not strict:
            current_state = self.encoder.state_dict()
            filtered_state_dict = {}
            for key, value in encoder_state_dict.items():
                if key not in current_state:
                    skipped.append((key, tuple(value.shape), None))
                    continue
                if current_state[key].shape != value.shape:
                    skipped.append((key, tuple(value.shape), tuple(current_state[key].shape)))
                    continue
                filtered_state_dict[key] = value
            encoder_state_dict = filtered_state_dict

        result = self.encoder.load_state_dict(encoder_state_dict, strict=strict)
        cprint(
            f"[PointSAMPointcloudEncoder] loaded checkpoint from {self._resolve_checkpoint_path(checkpoint_path)} "
            f"using prefix {used_prefix}",
            "cyan"
        )
        cprint(f"  Missing keys: {result.missing_keys}", "yellow")
        cprint(f"  Unexpected keys: {result.unexpected_keys}", "yellow")
        if skipped:
            cprint(f"  Skipped incompatible keys: {skipped[:10]}", "yellow")

    @staticmethod
    def _normalize_text_prompts(text, batch_size):
        if text is None:
            raise ValueError("PointSAM heatmap prediction requires text prompts")
        if isinstance(text, str):
            text = [text] * batch_size
        elif isinstance(text, tuple):
            text = list(text)
        elif not isinstance(text, list):
            text = [str(text)] * batch_size

        if len(text) == 1 and batch_size != 1:
            text = text * batch_size
        if len(text) != batch_size:
            raise ValueError(
                f"text batch size mismatch: got {len(text)} prompts for batch_size={batch_size}"
            )

        normalized = []
        for item in text:
            if isinstance(item, str):
                normalized.append([item])
            elif isinstance(item, (list, tuple)) and len(item) == 1:
                normalized.append([str(item[0])])
            else:
                raise ValueError(
                    "PointSAM expects exactly one text query per point cloud, "
                    f"got {item!r}"
                )
        return normalized

    def _decoder_uses_pointsam_interface(self):
        try:
            params = inspect.signature(self.heatmap_decoder.forward).parameters
        except (TypeError, ValueError):
            return False
        return "aux_inputs" in params

    def _predict_heatmap_from_tokens(self, pts, colors, patch_tokens, patches, text):
        text_prompts = self._normalize_text_prompts(text, pts.shape[0])
        try:
            text_tokens = self.heatmap_text_encoder(
                text_prompts,
                device=pts.device,
                dtype=patch_tokens.dtype,
            )
        except TypeError:
            text_tokens = self.heatmap_text_encoder(text_prompts)

        if self._decoder_uses_pointsam_interface():
            from pc_sam.model.decoder.simple_heatmap_decoder import AuxInputs

            centers = patches["centers"]
            query_coords = centers if self.heatmap_query == "patch" else pts
            aux_inputs = AuxInputs(
                coords=pts,
                features=colors,
                centers=centers,
                query_coords=query_coords,
            )
            heatmap = self.heatmap_decoder(
                pc_embeddings=patch_tokens,
                pc_pe=self.pe_layer(centers),
                text_tokens=text_tokens,
                aux_inputs=aux_inputs,
            )
        else:
            heatmap = self.heatmap_decoder(patch_tokens, text_tokens)

        if heatmap.dim() == 3 and heatmap.shape[1] != 1:
            heatmap = heatmap.mean(dim=1, keepdim=True)
        heatmap = heatmap.transpose(1, 2).contiguous()
        if self.heatmap_as_prob:
            heatmap = heatmap.sigmoid()
        return heatmap

    def forward(self, pcd, eval, text=None, return_heatmap=False):
        if self.freeze_heatmap_components or eval:
            self.encoder.eval()
            if self.heatmap_text_encoder is not None:
                self.heatmap_text_encoder.eval()
            if self.heatmap_decoder is not None:
                self.heatmap_decoder.eval()

        pts = pcd[..., :3].contiguous()
        if pcd.shape[-1] >= 6:
            colors = pcd[..., 3:6].contiguous()
        else:
            colors = torch.zeros_like(pts)

        patch_tokens, patches = self.encoder(pts, colors)
        centers = patches["centers"]
        pc_pe = self.pe_layer(centers)
        if return_heatmap:
            if self.heatmap_model is None:
                raise RuntimeError(
                    "PointSAM heatmap requested, but use_heatmap_model is false"
                )
            heatmap = self._predict_heatmap_from_tokens(
                pts, colors, patch_tokens, patches, text
            )
            return patch_tokens, pc_pe, heatmap
        return patch_tokens, pc_pe


class PointSAMPointcloudEncoder(PointSAMPointcloudEncoderV2):
    """Original PointSAM encoder-only adapter.

    Keep this class for the older Point-SAM encoder path. It builds only the
    pc_encoder, optionally loads encoder weights, and returns patch tokens plus
    patch-center positional encodings. Heatmap/text components are intentionally
    not constructed here.
    """

    def __init__(
            self,
            pointsam_root=None,
            pointsam_builder='build_uni3d_b_encoder_for_sam',
            embed_dim=256,
            out_dim=None,
            use_pretrained_weights=False,
            pretrained_weights_path=None,
            checkpoint_key_prefix='pc_encoder.',
            strict_load=False,
            ignore_mismatched_sizes=True,
            freeze=False,
            freeze_uni3d=False,
            freeze_out_proj=False,
            use_moe_feb=False,
            **kwargs):
        super().__init__(
            pointsam_root=pointsam_root,
            pointsam_builder=pointsam_builder,
            embed_dim=embed_dim,
            out_dim=out_dim,
            use_pretrained_weights=use_pretrained_weights,
            pretrained_weights_path=pretrained_weights_path,
            checkpoint_key_prefix=checkpoint_key_prefix,
            strict_load=strict_load,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            freeze=freeze,
            freeze_uni3d=freeze_uni3d,
            freeze_out_proj=freeze_out_proj,
            use_moe_feb=use_moe_feb,
            use_heatmap_model=False,
            **kwargs,
        )
        self.supports_heatmap = False


def resolve_pointsam_heatmap_config_name(heatmap_model_variant="simple", heatmap_config_name=None):
    if heatmap_config_name:
        return heatmap_config_name

    normalized = str(heatmap_model_variant or "simple").lower().replace("-", "_")
    if normalized == "simple":
        return "large_heatmap_r3d_simple"
    if normalized in ("cross", "cross_attention", "crossattn", "cross_attn"):
        return "large_heatmap_r3d_cross_attention"
    raise ValueError(
        "heatmap_model_variant must be 'simple' or 'cross_attention', "
        f"got {heatmap_model_variant!r}"
    )


class PointSAMHeatmapPointcloudEncoder(nn.Module):
    """Standalone PointSAM heatmap adapter for R3D."""

    _POINTSAM_BUILDER_KWARGS = {
        'drop_path_rate',
        'patch_dropout',
        'num_group',
        'group_size',
        'pc_encoder_dim',
        'use_esam_adapter',
        'esam_adapter_mlp_ratio',
        'esam_adapter_scale',
        'use_post_vit_mlp',
        'post_vit_mlp_ratio',
        'post_vit_mlp_hidden_dim',
    }

    def __init__(
            self,
            pointsam_root=None,
            pointsam_builder='build_uni3d_ti_encoder_for_sam',
            embed_dim=256,
            out_dim=None,
            heatmap_model_variant='simple',
            heatmap_config_dir=None,
            heatmap_config_name=None,
            heatmap_as_prob=True,
            load_encoder_checkpoint=False,
            encoder_checkpoint_path=None,
            load_pointsam_checkpoint=False,
            pointsam_checkpoint_path=None,
            strict_load=False,
            ignore_mismatched_sizes=True,
            freeze=False,
            freeze_uni3d=False,
            freeze_out_proj=False,
            use_esam_adapter=None,
            esam_adapter_mlp_ratio=None,
            esam_adapter_scale=None,
            clip_name=None,
            **kwargs):
        super().__init__()

        self.embed_dim = out_dim or embed_dim
        self.heatmap_as_prob = heatmap_as_prob
        self.freeze_heatmap_components = freeze
        self.supports_heatmap = True
        self._pointsam_builder = pointsam_builder
        self._pointsam_clip_name = clip_name
        self._pointsam_builder_overrides = {
            key: kwargs[key]
            for key in self._POINTSAM_BUILDER_KWARGS
            if key in kwargs and kwargs[key] is not None
        }
        self._pointsam_builder_overrides['out_dim'] = out_dim or embed_dim
        if use_esam_adapter is not None:
            self._pointsam_builder_overrides['use_esam_adapter'] = use_esam_adapter
        if esam_adapter_mlp_ratio is not None:
            self._pointsam_builder_overrides['esam_adapter_mlp_ratio'] = esam_adapter_mlp_ratio
        if esam_adapter_scale is not None:
            self._pointsam_builder_overrides['esam_adapter_scale'] = esam_adapter_scale

        config_name = resolve_pointsam_heatmap_config_name(
            heatmap_model_variant=heatmap_model_variant,
            heatmap_config_name=heatmap_config_name,
        )
        pointsam_root = self._prepare_pointsam_import(pointsam_root)
        self._build_heatmap_model(
            pointsam_root=pointsam_root,
            config_dir=heatmap_config_dir,
            config_name=config_name,
            checkpoint_path=pointsam_checkpoint_path if load_pointsam_checkpoint else None,
            strict=strict_load,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )

        if load_encoder_checkpoint:
            self._load_encoder_checkpoint(
                encoder_checkpoint_path,
                strict=strict_load,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )

        if freeze:
            for module in (self.encoder, self.heatmap_text_encoder, self.heatmap_decoder):
                if module is None:
                    continue
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            cprint("[PointSAMHeatmapPointcloudEncoder] froze PointSAM heatmap components", "yellow")
        elif freeze_uni3d:
            for name, param in self.encoder.named_parameters():
                keep_trainable = (
                    name.startswith('out_proj.')
                    or name.startswith('post_vit_mlp.')
                    or '.MLP_Adapter.' in name
                    or '.Space_Adapter.' in name
                )
                param.requires_grad = keep_trainable
            cprint(
                "[PointSAMHeatmapPointcloudEncoder] froze Uni3D backbone, kept out_proj/adapters trainable",
                "yellow",
            )

        if freeze_out_proj and hasattr(self.encoder, 'out_proj'):
            for param in self.encoder.out_proj.parameters():
                param.requires_grad = False
            cprint("[PointSAMHeatmapPointcloudEncoder] froze out_proj", "yellow")

        cprint(
            f"[PointSAMHeatmapPointcloudEncoder] ready config={config_name}, out_dim={self.embed_dim}",
            "cyan",
        )

    @staticmethod
    def _prepare_pointsam_import(pointsam_root):
        """
        R3D 工程能够直接导入pc_sam.*
        """
        if not pointsam_root:
            return _installed_pointsam_root()
        pointsam_root = os.path.abspath(os.path.expanduser(pointsam_root))
        extra_paths = [
            pointsam_root,
            os.path.join(pointsam_root, 'third_party', 'Pointnet2_PyTorch', 'pointnet2_ops_lib'),
            os.path.join(pointsam_root, 'third_party', 'torkit3d'),
        ]
        for extra_path in extra_paths:
            if not os.path.exists(extra_path):
                continue
            if extra_path in sys.path:
                sys.path.remove(extra_path)
            sys.path.insert(0, extra_path)

        loaded_pc_sam = sys.modules.get('pc_sam')
        loaded_path = getattr(loaded_pc_sam, '__file__', None)
        if loaded_path is not None:
            loaded_path = os.path.abspath(loaded_path)
            if not loaded_path.startswith(pointsam_root):
                for module_name in list(sys.modules.keys()):
                    if module_name == 'pc_sam' or module_name.startswith('pc_sam.'):
                        del sys.modules[module_name]
        return pointsam_root

    def _build_heatmap_model(
        self,
        pointsam_root,
        config_dir,
        config_name,
        checkpoint_path,
        strict,
        ignore_mismatched_sizes=False,
        encoder_overrides=None,
    ):
        config_dir = config_dir or os.path.join(pointsam_root, 'configs')
        config_dir = os.path.abspath(os.path.expanduser(config_dir))
        checkpoint_path = (
            self._resolve_checkpoint_path(checkpoint_path)
            if checkpoint_path is not None else None
        )

        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf, open_dict
        import hydra

        global_hydra = GlobalHydra.instance()
        if global_hydra.is_initialized():
            global_hydra.clear()
        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = hydra.compose(config_name=config_name, overrides=[])
            OmegaConf.resolve(cfg)

        builder_overrides = dict(encoder_overrides or {})
        builder_overrides.update(getattr(self, '_pointsam_builder_overrides', {}))
        with open_dict(cfg.model.pc_encoder):
            cfg.model.pc_encoder._target_ = (
                f"pc_sam.model.encoder.uni3d_point_encoder.{self._pointsam_builder}"
            )
            for key, value in builder_overrides.items():
                if value is not None:
                    cfg.model.pc_encoder[key] = value

        with open_dict(cfg.model.mask_decoder):
            if 'transformer_dim' in cfg.model.mask_decoder:
                cfg.model.mask_decoder.transformer_dim = self.embed_dim
        with open_dict(cfg.model.text_prompt_encoder):
            if 'embed_dim' in cfg.model.text_prompt_encoder:
                cfg.model.text_prompt_encoder.embed_dim = self.embed_dim
            if self._pointsam_clip_name:
                cfg.model.text_prompt_encoder.clip_name = self._pointsam_clip_name

        self.pointsam_model = hydra.utils.instantiate(cfg.model)
        self.encoder = self.pointsam_model.pc_encoder
        self.heatmap_text_encoder = self.pointsam_model.text_prompt_encoder
        self.heatmap_decoder = self.pointsam_model.mask_decoder

        state_dict = self._read_checkpoint(checkpoint_path) if checkpoint_path else None
        if state_dict is not None:
            for module_name, module in (
                ('pc_encoder', self.encoder),
                ('text_prompt_encoder', self.heatmap_text_encoder),
                ('mask_decoder', self.heatmap_decoder),
            ):
                self._load_prefixed_module(
                    module,
                    state_dict,
                    f"{module_name}.",
                    strict,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                )
        else:
            cprint("[PointSAMHeatmapPointcloudEncoder] no full PointSAM checkpoint loaded", "yellow")

        self.heatmap_model = self.pointsam_model
        self.embed_dim = getattr(self.encoder, 'embed_dim', self.embed_dim)
        self.supports_heatmap = True
        cprint(
            "[PointSAMHeatmapPointcloudEncoder] built PointSAM heatmap model "
            f"decoder={self.heatmap_decoder.__class__.__name__}, "
            f"checkpoint={checkpoint_path}, encoder out_dim={self.embed_dim}",
            "cyan",
        )

    def _resolve_checkpoint_path(self, checkpoint_path):
        checkpoint_path = os.path.abspath(os.path.expanduser(str(checkpoint_path)))
        if os.path.isdir(checkpoint_path):
            for filename in ('model.safetensors', 'pytorch_model.bin', 'model.pt', 'checkpoint.pt'):
                candidate = os.path.join(checkpoint_path, filename)
                if os.path.exists(candidate):
                    return candidate
        return checkpoint_path

    def _read_checkpoint(self, checkpoint_path):
        checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            cprint(f"[PointSAMHeatmapPointcloudEncoder] checkpoint not found: {checkpoint_path}", "red")
            return None

        if checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            return load_file(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        for key in ('state_dict', 'model', 'module'):
            if isinstance(checkpoint, dict) and key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
        return checkpoint

    def _load_prefixed_module(
            self,
            module,
            state_dict,
            prefix,
            strict,
            ignore_mismatched_sizes=False):
        if state_dict is None:
            if strict:
                raise RuntimeError(f"Missing checkpoint state_dict for strict load of {prefix}")
            cprint(f"[PointSAMHeatmapPointcloudEncoder] skipped {prefix}: no checkpoint", "yellow")
            return None

        module_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not module_state:
            if strict:
                raise RuntimeError(f"No checkpoint keys matched prefix {prefix}")
            cprint(f"[PointSAMHeatmapPointcloudEncoder] skipped {prefix}: no matched keys", "yellow")
            return None

        skipped = []
        if ignore_mismatched_sizes and not strict:
            current_state = module.state_dict()
            filtered_state = {}
            for key, value in module_state.items():
                current_value = current_state.get(key)
                if current_value is not None and current_value.shape != value.shape:
                    skipped.append((key, tuple(value.shape), tuple(current_value.shape)))
                    continue
                filtered_state[key] = value
            module_state = filtered_state

        result = module.load_state_dict(module_state, strict=strict)
        cprint(
            f"[PointSAMHeatmapPointcloudEncoder] loaded {prefix} "
            f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}",
            "yellow" if result.missing_keys or result.unexpected_keys else "cyan",
        )
        if skipped:
            cprint(f"  Skipped incompatible {prefix} keys: {skipped[:10]}", "yellow")
        return result

    def _extract_encoder_state_dict(self, state_dict):
        for prefix in ('module.pc_encoder.', 'model.pc_encoder.', 'pc_encoder.', 'point_encoder.'):
            matched = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if matched:
                return matched, prefix

        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            if key.startswith('point_encoder.'):
                key = key[len('point_encoder.'):]
            if key == 'logit_scale':
                continue
            cleaned[key] = value
        return cleaned, '<raw>'

    def _load_encoder_checkpoint(self, checkpoint_path, strict, ignore_mismatched_sizes):
        if checkpoint_path is None:
            if strict:
                raise RuntimeError("load_encoder_checkpoint=True but encoder_checkpoint_path is missing")
            cprint("[PointSAMHeatmapPointcloudEncoder] skipped encoder checkpoint: no path", "yellow")
            return

        state_dict = self._read_checkpoint(checkpoint_path)
        if state_dict is None:
            if strict:
                raise RuntimeError("load_encoder_checkpoint=True but checkpoint is missing")
            return

        encoder_state_dict, used_prefix = self._extract_encoder_state_dict(state_dict)
        skipped = []
        if ignore_mismatched_sizes and not strict:
            current_state = self.encoder.state_dict()
            filtered_state_dict = {}
            for key, value in encoder_state_dict.items():
                current_value = current_state.get(key)
                if current_value is None:
                    skipped.append((key, tuple(value.shape), None))
                    continue
                if current_value.shape != value.shape:
                    skipped.append((key, tuple(value.shape), tuple(current_value.shape)))
                    continue
                filtered_state_dict[key] = value
            encoder_state_dict = filtered_state_dict

        result = self.encoder.load_state_dict(encoder_state_dict, strict=strict)
        cprint(
            f"[PointSAMHeatmapPointcloudEncoder] loaded encoder checkpoint from {self._resolve_checkpoint_path(checkpoint_path)} "
            f"using prefix {used_prefix}",
            "cyan",
        )
        cprint(f"  Missing keys: {result.missing_keys}", "yellow")
        cprint(f"  Unexpected keys: {result.unexpected_keys}", "yellow")
        if skipped:
            cprint(f"  Skipped incompatible keys: {skipped[:10]}", "yellow")

    @staticmethod
    def _normalize_text_prompts(text, batch_size):
        if text is None:
            raise ValueError("PointSAM heatmap prediction requires text prompts")
        if isinstance(text, str):
            text = [text] * batch_size
        elif isinstance(text, tuple):
            text = list(text)
        elif not isinstance(text, list):
            text = [str(text)] * batch_size

        if len(text) == 1 and batch_size != 1:
            text = text * batch_size
        if len(text) != batch_size:
            raise ValueError(
                f"text batch size mismatch: got {len(text)} prompts for batch_size={batch_size}"
            )

        normalized = []
        for item in text:
            if isinstance(item, str):
                normalized.append([item])
            elif isinstance(item, (list, tuple)) and len(item) == 1:
                normalized.append([str(item[0])])
            else:
                raise ValueError(
                    "PointSAM heatmap encoder expects exactly one text query per point cloud, "
                    f"got {item!r}"
                )
        return normalized

    def forward(self, pcd, eval, text=None, return_heatmap=False):
        if self.freeze_heatmap_components or eval:
            self.encoder.eval()
            if self.heatmap_text_encoder is not None:
                self.heatmap_text_encoder.eval()
            if self.heatmap_decoder is not None:
                self.heatmap_decoder.eval()

        pts = pcd[..., :3].contiguous()
        if pcd.shape[-1] >= 6:
            colors = pcd[..., 3:6].contiguous()
        else:
            colors = torch.zeros_like(pts)

        if not return_heatmap:# 不需要heatmap，就不需要文本。只进行点云vit。
            patch_tokens, _, pc_pe = self.pointsam_model.encode_points(pts, colors)
            return patch_tokens, pc_pe

        text_prompts = self._normalize_text_prompts(text, pts.shape[0]) #保证 每个 batch 样本都有且只有一条文本，并且格式统一成 PointSAM 能吃的 [B,1] 文本列表。
        patch_tokens, pc_pe, patch_heatmap_logits = self.pointsam_model.forward_r3d_heatmap(
            coords=pts,
            features=colors,
            text=text_prompts,
        )
        patch_heatmap = (
            patch_heatmap_logits.sigmoid()
            if self.heatmap_as_prob
            else patch_heatmap_logits
        )
        return patch_tokens, pc_pe, patch_heatmap

class Uni3DPointcloudEncoder(nn.Module):
    """
    Uni3D point cloud encoder.
    Supports both pretrained weight loading and training from scratch.
    """
    def __init__(self,
                 pc_model='eva02_large_patch14_448',
                 pc_feat_dim=1024,
                 embed_dim=1024,
                 group_size=32,
                 num_group=512,
                 patch_dropout=0.5,
                 drop_path_rate=0.2,
                 pretrained_pc=None,
                 pc_encoder_dim=512,
                 use_pretrained_weights=False,
                 pretrained_weights_path=None,
                 normalization_type="batch_norm",
                 feature_mode="pointsam",
                 extract_global_feature=True,
                 fps_random_config=None,
                 freeze=False,
                 **kwargs):
        super().__init__()

        # vit backbone
        self.transformer = timm.create_model(pc_model, checkpoint_path=pretrained_pc, drop_path_rate=drop_path_rate)
        self.transformer_dim = self.transformer.embed_dim
        self.embed_dim = embed_dim
        self.num_group = num_group
        self.use_pretrained_weights = use_pretrained_weights
        self.freeze = freeze

        self.patch_embed = PatchEmbed(in_channels=6, out_channels=512, num_patches=num_group, patch_size=group_size, fps_random_config=fps_random_config)

        # 7 = xyz + rgb + dist
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.transformer_dim)
        )

        self.extract_global_feature = feature_mode != 'pointsam'

        # for pointsam output pc_pe
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Patch dropout
        self.patch_dropout = PatchDropout(patch_dropout, exclude_first_token=(feature_mode=="cls")) if patch_dropout > 0. else nn.Identity()
        # Project transformer output to embedding dim
        self.out_proj = nn.Linear(self.transformer_dim, self.embed_dim)
        self.patch_proj = nn.Linear(self.patch_embed.out_channels, self.transformer_dim)
        self.feature_mode = feature_mode

        if self.feature_mode == "cls":
            cprint(f"[Uni3DPointcloudEncoder] use cls token", "red")

            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.transformer_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.transformer_dim))
        elif self.feature_mode == "max_pooling":
            cprint(f"[Uni3DPointcloudEncoder] use max pooling", "red")
        else:  # pointsam
            cprint(f"[Uni3DPointcloudEncoder] use pointsam, do not extract global feature", "red")

        # Load pretrained weights if specified
        if use_pretrained_weights:
            self._load_pretrained_weights_selective(pretrained_weights_path, normalization_type)
        else:
            cprint(f"[Uni3DPointcloudEncoder] Using random initialization (training from scratch)", "red")

        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
            cprint("[Uni3DPointcloudEncoder] froze all Uni3D parameters", "yellow")

    def _load_pretrained_weights_selective(self, pretrained_weights_path, normalization_type):
        """
        Selectively load pretrained weights based on normalization_type.

        Args:
            pretrained_weights_path: Path to pretrained weights
            normalization_type: Normalization type ("batch_norm", "layer_norm", "none")
        """
        load_weight_path = pretrained_weights_path
        if not os.path.exists(load_weight_path):
            cprint(f"[Uni3DPointcloudEncoder] Pretrained weights file not found: {load_weight_path}", "red")
            return

        # Load pretrained weights
        from safetensors.torch import load_file
        checkpoint = load_file(os.path.join(load_weight_path, "model.safetensors"))
        # Remap key names
        processed_state_dict = {}
        for key in list(checkpoint.keys()):
            if key.startswith('pc_encoder.'):
                new_key = key.replace('pc_encoder.', '')
                processed_state_dict[new_key] = checkpoint[key]
        missing_keys, unexpected_keys = self.load_state_dict(processed_state_dict, strict=False)
        cprint(f"  Missing keys: {missing_keys}", "yellow")
        cprint(f"  Unexpected keys: {unexpected_keys}", "yellow")

        cprint(f"[Uni3DPointcloudEncoder] Pretrained weights loaded: {load_weight_path}", "red")

    def forward(self, pcd, eval):
        eval = eval or self.freeze
        # Apply point cloud dropout (data augmentation)
        if not eval:
            pcd = random_point_dropout(pcd, max_dropout_ratio=0.8)

        pts = pcd[..., :3].contiguous()
        colors = pcd[..., 3:].contiguous()
        # Group points into patches and get embeddings
        patches = self.patch_embed(pts, colors)
        if isinstance(patches, list):
            patch_embed = patches[-1]["embeddings"]
            centers = patches[-1]["centers"]
        else:
            patch_embed = patches["embeddings"]  # [B, L, D]
            centers = patches["centers"]  # [B, L, 3]
        patch_embed = self.patch_proj(patch_embed)

        # Add positional embedding
        pos_embed = self.pos_embed(centers)

        if self.feature_mode == "cls":

            # prepare cls
            cls_tokens = self.cls_token.expand(patch_embed.size(0), -1, -1)  
            cls_pos = self.cls_pos.expand(pos_embed.size(0), -1, -1) 

            # final input
            patch_embed = torch.cat((cls_tokens, patch_embed), dim=1)
            pos_embed = torch.cat((cls_pos, pos_embed), dim=1)
        
        x = patch_embed + pos_embed
        # patch dropout
        if not eval:
            x = self.patch_dropout(x)
            x = self.transformer.pos_drop(x)

        for block in self.transformer.blocks:
            x = block(x)

        if self.extract_global_feature:

            # Extract features based on whether CLS token is used
            if self.feature_mode == "cls":
                # Use CLS token (first token) for classification
                x = self.transformer.norm(x[:, 0, :])
            elif self.feature_mode == "max_pooling":
                # Use global max pooling over all patch tokens
                x = self.transformer.norm(torch.max(x, dim=1)[0])
        else: 
            # pointsam, do not extract global feature
            x = self.transformer.norm(x)
        
        x = self.transformer.fc_norm(x)
        x = self.out_proj(x)

        if not self.extract_global_feature:
            pc_pe = self.pe_layer(centers)
            return x, pc_pe
        else:
            return x
