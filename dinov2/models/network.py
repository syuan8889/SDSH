import torch.nn as nn
from torchvision import models
from functools import partial
import timm.models.vision_transformer
from timm.models.vision_transformer import vit_base_patch16_224
from timm.models.vision_transformer import _load_weights
import torch

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class PatchAttention(nn.Module):
    """计算patch tokens的注意力权重，类似SE模块的加权池化机制。
    Args:
        embed_dim (int): 输入的embedding维度
        hidden_dim (int): 中间层维度
        num_patches (int): patch的数量，默认为196 (14x14)
    """

    def __init__(self, embed_dim, hidden_dim=768, num_patches=196):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # 通过全连接层生成中间表示
            nn.GELU(),
            nn.Linear(hidden_dim, num_patches),
            nn.Sigmoid()  # 使用Sigmoid来获得权重
        )
        self.norm = nn.LayerNorm(embed_dim * 2, eps=1e-6)
    def forward(self, cls_token, patch_tokens):
        """
        Args:
            cls_token (torch.Tensor): [B, embed_dim]
            patch_tokens (torch.Tensor): [B, num_patches, embed_dim]

        Returns:
            torch.Tensor: [B, embed_dim*2] 包含cls token和加权平均的patch信息
        """
        # 计算全局特征
        patch_avg = patch_tokens.mean(dim=1)  # [B, embed_dim]
        global_feat = torch.cat([cls_token, patch_avg], dim=1)  # [B, embed_dim*2]
        # 计算注意力权重（每个patch的权重）
        attention_weights = self.attention_net(global_feat)  # [B, num_patches]
        attention_weights = attention_weights.unsqueeze(-1)  # [B, num_patches, 1]
        # 加权平均patch tokens
        weighted_patches = (patch_tokens * attention_weights).mean(dim=1)  # [B, embed_dim]
        # 拼接结果
        output = torch.cat([cls_token, weighted_patches], dim=1)  # [B, embed_dim*2]
        return self.norm(output)


class ViT(nn.Module):
    def __init__(self, hash_bit, supervised_pretrain=True):
        super(ViT, self).__init__()
        self.global_pool = False # ys TODO True or False
        # model_vit = vit_base_patch16(drop_path_rate=0.1, global_pool=self.global_pool)
        if supervised_pretrain:  # load from supervised pretrained vit
            model_vit = vit_base_patch16_224(pretrained=False, drop_path_rate=0.1)
            pretrain_vit = '/mnt/8TDisk1/zhenglab/yuanshuai/vit-models-1k/ViT-B_16_sam.npz'
            _load_weights(model_vit, checkpoint_path=pretrain_vit )
            if self.global_pool:  # set model to support GAP
                model_vit.global_pool = self.global_pool
                norm_layer = partial(nn.LayerNorm, eps=1e-6)
                embed_dim = model_vit.embed_dim
                model_vit.fc_norm = norm_layer(embed_dim)
                del model_vit.norm  # remove the original norm

        else:  # load from pretrained mae
            model_vit = vit_base_patch16(drop_path_rate=0.1, global_pool=self.global_pool)
            checkpoint = torch.load(
                '/home/ouc/data1/qiaoshishi/deep_models_downloaded/mae_models/mae_pretrain/mae_pretrain_vit_base.pth',
                map_location='cpu')
            print("Load pre-trained checkpoint from: MAE")
            checkpoint_model = checkpoint['model']

            state_dict = model_vit.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model_vit, checkpoint_model)

            # load pre-trained model
            msg = model_vit.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            # if self.global_pool:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            # else:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        self.patch_embed = model_vit.patch_embed
        self.cls_token = model_vit.cls_token
        self.pos_embed = model_vit.pos_embed
        self.pos_drop = model_vit.pos_drop
        self.blocks = model_vit.blocks
        if self.global_pool:
            self.fc_norm = model_vit.fc_norm
        else:
            self.norm = model_vit.norm

        self.hash_layer = nn.Linear(model_vit.num_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        self.embed_dim = model_vit.embed_dim
        # TODO add hth hash module
        self.patch_attn = PatchAttention(embed_dim=self.embed_dim,
                                         hidden_dim=self.embed_dim, num_patches=196)
        self.hash_layer_1 = nn.Linear(self.embed_dim * 2, 256)
        self.hash_layer_2 = nn.Linear(self.embed_dim * 2, 256)
        self.hash_layer_3 = nn.Linear(self.embed_dim * 2, 256)
        self.hash_layer_4 = nn.Linear(self.embed_dim * 2, 256)
        self.patch_projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(in_features=self.embed_dim * 2, out_features=256),
        )

        self.hash_layer_global = nn.Linear(256 * 5, hash_bit)
    def forward_hash(self, x):
        v_e = self.patch_attn(x[:, 0, :], x[:, 1:, :])
        u_h_1 = self.hash_layer_1(v_e)
        u_h_2 = self.hash_layer_2(v_e)
        u_h_3 = self.hash_layer_3(v_e)
        u_h_4 = self.hash_layer_4(v_e)
        u_patch_pool = self.patch_projection(v_e[:, self.embed_dim:])
        fused_tensor = torch.stack([u_h_1, u_h_2, u_h_3, u_h_4, u_patch_pool], dim=1)
        u_fused = self.hash_layer_global(fused_tensor.view(-1, 256 * 5))

        return fused_tensor[:, :4, :], u_fused, v_e, u_patch_pool

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        # y = self.hash_layer(outcome)
        u_h, u_fused, v_e, u_patch_pool = self.forward_hash(x)

        return {
            "x_u_fused": u_fused,
            "x_u_h": u_h,
            "x_ve": v_e,
            "x_u_patch_pool": u_patch_pool
        }
