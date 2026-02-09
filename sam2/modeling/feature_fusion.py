# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.sam2_utils import LayerNorm2d, MLP


class FeatureFusion(nn.Module):
    """Fuses encoder layer 0 and layer 1 features via separate projections and addition."""

    def __init__(self, input_dim_layer0, input_dim_layer1, hidden_dim):
        super().__init__()
        # Separate 1x1 convs for each layer
        self.process_hiera_feat_0 = nn.Sequential(
            nn.Conv2d(input_dim_layer0, hidden_dim, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, input_dim_layer0, kernel_size=3, stride=1, padding=1)
        )
        self.process_hiera_feat_1 = nn.Sequential(
            nn.Conv2d(input_dim_layer1, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, input_dim_layer1 // 2, kernel_size=3, stride=1, padding=1),
            # Reduce channels to 32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Upsample to [1, 32, 256, 256]
        )
        self.norm = LayerNorm2d(input_dim_layer0)
        self.act = nn.GELU()

    def forward(self, layer0_feat, layer1_feat):
        """
        Args:
            layer0_feat: [B, C0, H, W] - Higher resolution features
            layer1_feat: [B, C1, H/2, W/2] - Lower resolution features
        Returns:
            fused: [B, output_dim, H, W] - Fused features
        """
        # Project layer0
        layer0_proj = self.process_hiera_feat_0(layer0_feat)


        layer1_proj = self.process_hiera_feat_1(layer1_feat)

        # Add projections
        fused = layer0_proj + layer1_proj

        # Normalize (BCHW → BHWC → BCHW)
        fused = self.norm(fused)

        return self.act(fused)


class PrototypeExtractor(nn.Module):
    """Extracts prototypes using learnable query embeddings via cross-attention."""

    def __init__(self, num_prototypes, feature_dim, num_heads):
        super().__init__()
        self.num_prototypes = num_prototypes

        # Learnable query embeddings
        self.prototype_queries = nn.Parameter(
            torch.randn(num_prototypes, feature_dim)
        )
        nn.init.trunc_normal_(self.prototype_queries, std=0.02)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, fused_features):
        """
        Args:
            fused_features: [B, C, H, W] - Fused encoder features
        Returns:
            prototypes: [B, num_prototypes, C] - Extracted prototypes
        """
        B, C, H, W = fused_features.shape

        # Flatten spatial dimensions
        feat_flat = fused_features.flatten(2).permute(0, 2, 1)  # [B, HW, C]

        # Expand queries for batch
        queries = self.prototype_queries.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention: queries attend to features
        attn_out, _ = self.cross_attn(query=queries, key=feat_flat, value=feat_flat)
        queries = self.norm1(queries + attn_out)

        # Feedforward
        ffn_out = self.ffn(queries)
        prototypes = self.norm2(queries + ffn_out)

        return prototypes


class PrototypeConditioningModule(nn.Module):
    """Conditions current features on previous frame prototypes via cross-attention."""

    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.output_dim = output_dim

        # Project input to output dimension
        self.input_proj = nn.Conv2d(input_dim, output_dim, 1)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            output_dim, num_heads, batch_first=True
        )

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, combined_features, prototypes):
        """
        Args:
            combined_features: [B, input_dim, H, W] - Concatenated fused features and masks
            prototypes: [B, num_proto, C] or None - Prototypes from previous frame
        Returns:
            feat_enhanced: [B, output_dim, H, W] - Enhanced features
        """
        if prototypes is None:
            # First frame: return zeros
            B, _, H, W = combined_features.shape
            return torch.zeros(
                B,
                self.output_dim,
                H,
                W,
                device=combined_features.device,
                dtype=combined_features.dtype,
            )

        # Project input
        feat_proj = self.input_proj(combined_features)
        B, C, H, W = feat_proj.shape

        # Flatten spatial dimensions
        feat_flat = feat_proj.flatten(2).permute(0, 2, 1)  # [B, HW, C]

        # Cross-attention: features (query) attend to prototypes (key/value)
        attn_out, _ = self.cross_attn(
            query=feat_flat, key=prototypes, value=prototypes
        )

        # Residual + normalization
        feat_enhanced = self.norm(feat_flat + attn_out)

        # Reshape back to spatial
        feat_enhanced = feat_enhanced.permute(0, 2, 1).view(B, C, H, W)

        return feat_enhanced


class MaskRefinementHead(nn.Module):
    """
    Generates refined masks using hypernetwork approach.
    Refine tokens generate per-pixel weights via MLP, then matrix multiply with enhanced features.
    """

    def __init__(self, hidden_dim, transformer_dim, num_refine_tokens):
        """
        Args:
            hidden_dim: Dimension of enhanced features (typically 32 = transformer_dim // 8)
            transformer_dim: Transformer dimension (typically 256)
            num_refine_tokens: Number of refine tokens
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project upscaled_embedding to hidden_dim
        # upscaled_embedding is already at hidden_dim (transformer_dim // 8)
        # Just add a projection for potential dimension adjustment
        self.upscaled_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, hidden_dim, 3, 1, 1)
        )

        # LayerNorm and activation for enhanced features (same as SAM decoder)
        # This normalizes the feature values to prevent exploding logits
        self.layer_norm = LayerNorm2d(hidden_dim)
        self.activation = nn.GELU()

        # Hypernetwork MLP: projects refine tokens from transformer_dim to hidden_dim
        # Similar to SAM's output_hypernetworks_mlps
        self.refine_token_mlp = MLP(transformer_dim,transformer_dim,transformer_dim // 8, 3)

    def forward(self, cross_attn_result, upscaled_embedding, refine_tokens_out):
        """
        Args:
            cross_attn_result: [B, hidden_dim, H, W] - Prototype-conditioned features
            upscaled_embedding: [B, hidden_dim, H, W] - Upscaled features from decoder
            refine_tokens_out: [B, N, transformer_dim] - Refine tokens from transformer
        Returns:
            refine_masks: [B, 1, H, W] - Refined mask predictions
        """
        B = upscaled_embedding.size(0)
        C = self.hidden_dim
        H, W = upscaled_embedding.shape[-2:]

        # Project upscaled_embedding
        upscaled_proj = self.upscaled_proj(upscaled_embedding)  # [B, C, H_up, W_up]

        # Add to create enhanced features
        enhanced_feat = upscaled_proj + cross_attn_result  # [B, C, H, W]

        # Apply LayerNorm and activation (same as SAM decoder to control value range)
        enhanced_feat = self.layer_norm(enhanced_feat)
        enhanced_feat = self.activation(enhanced_feat)

        # Process refine tokens through hypernetwork MLP
        # Average over N tokens first
        refine_tokens_avg = refine_tokens_out.mean(dim=1)  # [B, transformer_dim]

        # Project to hidden_dim (hypernetwork weights)
        hyper_weights = self.refine_token_mlp(
            refine_tokens_avg
        )  # [B, hidden_dim=32]

        # Flatten enhanced features for matrix multiplication
        enhanced_feat_flat = enhanced_feat.view(B, C, H * W)  # [B, C, H*W]

        # Hypernetwork-style prediction: weights @ features
        # hyper_weights: [B, C] needs to be [B, 1, C] for correct batch matmul
        # enhanced_feat_flat: [B, C, H*W]
        # Result: [B, 1, H*W]
        hyper_weights = hyper_weights.unsqueeze(1)  # [B, C] -> [B, 1, C]
        refine_masks_flat = hyper_weights @ enhanced_feat_flat  # [B, 1, C] @ [B, C, H*W] = [B, 1, H*W]

        # Reshape to spatial dimensions
        refine_masks = refine_masks_flat.view(B, -1, H, W)

        return refine_masks
