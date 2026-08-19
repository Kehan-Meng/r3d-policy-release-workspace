import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils

import logging

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

# https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py 
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist    


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        logging.info("patch dropout prob is {}".format(prob))

    def forward(self, x, return_indices=False):
        # Patch dropout is a training-only augmentation.  Returning the kept
        # patch indices lets dense prediction heads keep centers/KNN metadata
        # aligned with the surviving tokens.
        if not self.training or self.prob == 0.:
            return (x, None) if return_indices else x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch, device=x.device)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens, device=x.device)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        if return_indices:
            return x, patch_indices_keep
        return x


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
            knn_idx: B G M (indices for downsampling point heatmaps to patches)
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        knn_idx = knn_point(self.group_size, xyz, center) # B G M  ← save original index
        assert knn_idx.size(1) == self.num_group
        assert knn_idx.size(2) == self.group_size
        idx = knn_idx
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features, knn_idx

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 6)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


#wzy
class Adapter(nn.Module):
    """E-SAM adapter MLP."""

    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
#wzy


class PointcloudEncoder(nn.Module):
    def __init__(self, point_transformer, args):
        super().__init__()
        #===wzy===
        # args is a SimpleNamespace in Uni3DPointEncoderForSAM; EasyDict is unused.
        #===wzy===
        self.trans_dim = args.pc_feat_dim # 768
        self.embed_dim = args.embed_dim # 512
        self.group_size = args.group_size # 32
        self.num_group = args.num_group # 512
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dim =  args.pc_encoder_dim # 256
        self.encoder = Encoder(encoder_channel = self.encoder_dim)
       
        # bridge encoder and transformer
        self.encoder2trans = nn.Linear(self.encoder_dim,  self.trans_dim)
        
        # bridge transformer and clip embedding
        self.trans2embed = nn.Linear(self.trans_dim,  self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(args.patch_dropout) if args.patch_dropout > 0. else nn.Identity()
        self.visual = point_transformer

        #wzy
        self.use_esam_adapter = getattr(args, "use_esam_adapter", False)
        if self.use_esam_adapter:
            esam_adapter_mlp_ratio = getattr(args, "esam_adapter_mlp_ratio", 0.25)
            esam_adapter_scale = float(getattr(args, "esam_adapter_scale", 0.5))
            for blk in self.visual.blocks:
                blk.MLP_Adapter = Adapter(
                    self.trans_dim,
                    mlp_ratio=esam_adapter_mlp_ratio,
                    skip_connect=False,
                )
                blk.Space_Adapter = Adapter(
                    self.trans_dim,
                    mlp_ratio=esam_adapter_mlp_ratio,
                    skip_connect=True,
                )
                blk.esam_adapter_scale = esam_adapter_scale
        #wzy


    def forward(self, pts, colors):
        # divide the point cloud in the same form. This is important
        _, center, features, knn_idx = self.group_divider(pts, colors)

        # encoder the input cloud patches
        group_input_tokens = self.encoder(features)  #  B G N
        group_input_tokens = self.encoder2trans(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = x + pos
        # x = x.half()
        
        # A non-zero patch dropout keeps the CLS token and randomly removes
        # patch tokens during training.  Dense heatmap interpolation also
        # needs the corresponding centers/KNN groups to be filtered.
        if isinstance(self.patch_dropout, PatchDropout):
            x, patch_indices_keep = self.patch_dropout(x, return_indices=True)
            if patch_indices_keep is not None:
                center = torch.gather(
                    center,
                    dim=1,
                    index=patch_indices_keep.unsqueeze(-1).expand(
                        -1, -1, center.shape[-1]
                    ),
                )
                knn_idx = torch.gather(
                    knn_idx,
                    dim=1,
                    index=patch_indices_keep.unsqueeze(-1).expand(
                        -1, -1, knn_idx.shape[-1]
                    ),
                )
        else:
            x = self.patch_dropout(x)

        x = self.visual.pos_drop(x)

        # ModuleList not support forward
        #wzy
        for i, blk in enumerate(self.visual.blocks):
            if self.use_esam_adapter:
                attn_out = blk.attn(blk.norm1(x))
                if blk.gamma_1 is not None:
                    attn_out = blk.gamma_1 * attn_out
                attn_out = blk.Space_Adapter(attn_out)
                x = x + blk.drop_path1(attn_out)

                xn = blk.norm2(x)
                mlp_out = blk.mlp(xn)
                if blk.gamma_2 is not None:
                    mlp_out = blk.gamma_2 * mlp_out
                x = x + blk.drop_path2(
                    mlp_out + blk.esam_adapter_scale * blk.MLP_Adapter(xn)
                )
            else:
                x = blk(x)
        #wzy
        # Preserve the ViT CLS embedding in the same way as original Uni3D,
        # while keeping all patch tokens for dense heatmap prediction.
        cls_embedding = self.visual.norm(x[:, 0, :])
        cls_embedding = self.visual.fc_norm(cls_embedding)
        cls_embedding = self.trans2embed(cls_embedding)  # [B, embed_dim]

        #===wzy===
        x = self.visual.norm(x[:, 1:, :])
        #===wzy===
        x = self.visual.fc_norm(x)
        x = self.trans2embed(x)
        #===wzy===
        patches = {
            "centers": center,
            "knn_idx": knn_idx,
            "cls_embedding": cls_embedding,
        }
        return x, patches
        #===wzy===


#===wzy===
class Uni3DPointEncoderForSAM(PointcloudEncoder):
    """Concrete Uni3D point encoder class used directly by Hydra configs."""

    def __init__(
        self,
        #===wzy===
        # Uni3D ViT backbone.
        pc_model: str,
        pc_feat_dim: int,
        pretrained_pc: str = "",
        drop_path_rate: float = 0.0,

        # Point grouping and local point encoder.
        embed_dim: int = 1024,
        num_group: int = 512,
        group_size: int = 64,
        pc_encoder_dim: int = 512,
        patch_dropout: float = 0.0,

        # E-SAM adapters attached to every Uni3D ViT block.
        use_esam_adapter: bool = True,
        esam_adapter_mlp_ratio: float = 0.25,
        esam_adapter_scale: float = 0.5,
        #===wzy===
    ):
       
        import timm
        from types import SimpleNamespace

        args = SimpleNamespace(
            #===wzy===
            # Uni3D ViT backbone.
            pc_model=pc_model,
            pc_feat_dim=pc_feat_dim,
            pretrained_pc=pretrained_pc,
            drop_path_rate=drop_path_rate,

            # Point grouping and local point encoder.
            embed_dim=embed_dim,
            num_group=num_group,
            group_size=group_size,
            pc_encoder_dim=pc_encoder_dim,
            patch_dropout=patch_dropout,

            # E-SAM adapters attached to every Uni3D ViT block.
            use_esam_adapter=use_esam_adapter,
            esam_adapter_mlp_ratio=esam_adapter_mlp_ratio,
            esam_adapter_scale=esam_adapter_scale,
            #===wzy===
        )

        #通过timm创建好 nn.Module 模型实例。即point_transformer = EVA02Tiny模型对象，真正被使用的是blocks、pos_drop、norm、fc_norm等
        point_transformer = timm.create_model(
            args.pc_model,
            checkpoint_path=args.pretrained_pc,
            drop_path_rate=args.drop_path_rate, #随机丢弃残差分支
        )
        super().__init__(
            point_transformer=point_transformer,
            args=args,
        )
#===wzy===
