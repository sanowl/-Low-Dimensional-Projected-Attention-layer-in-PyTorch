import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class LowDimProjectedAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        r: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_cache: bool = False
    ):
        """
        Initialize the Low-Dimensional Projected Attention layer.

        Args:
            d_model (int): Dimension of the model (e.g., 512, 768).
            r (int): Dimension for low-dimensional projection.
            n_heads (int): Number of attention heads.
            dropout (float, optional): Dropout probability on attention weights. Default: 0.1
            bias (bool, optional): If set to False, the projection layers will not learn an additive bias. Default: True
            use_cache (bool, optional): If True, enables caching of key and value tensors for faster inference. Default: False
        """
        super(LowDimProjectedAttention, self).__init__()

        # Validate dimensions
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        assert r % n_heads == 0, (
            f"r ({r}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.r = r
        self.n_heads = n_heads
        self.d_k = r // n_heads  # Dimension per head in the low-dimensional space
        self.use_cache = use_cache

        # Projection layers for queries, keys, and values
        self.query_proj = nn.Linear(d_model, r, bias=bias)
        self.key_proj = nn.Linear(d_model, r, bias=bias)
        self.value_proj = nn.Linear(d_model, r, bias=bias)

        # Output projection layer
        self.output_proj = nn.Linear(r, d_model, bias=bias)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Scaling factor for attention scores
        self.scale = math.sqrt(self.d_k)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameters using Xavier uniform initialization.
        """
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.query_proj.bias is not None:
            nn.init.constant_(self.query_proj.bias, 0.)
            nn.init.constant_(self.key_proj.bias, 0.)
            nn.init.constant_(self.value_proj.bias, 0.)
            nn.init.constant_(self.output_proj.bias, 0.)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None
    ) -> torch.Tensor:
        """
        Forward pass for the Low-Dimensional Projected Attention layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Attention mask of shape (batch_size, 1, 1, seq_len).
            cache (dict, optional): Dictionary to store cached keys and values.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
            Optional[dict]: Updated cache with keys and values (if caching is used).
        """
        batch_size, seq_len, _ = x.size()

        # Project inputs to the low-dimensional space
        # Shape after projection: (batch_size, seq_len, r)
        Q = self.query_proj(x)  # (batch_size, seq_len, r)
        K = self.key_proj(x)    # (batch_size, seq_len, r)
        V = self.value_proj(x)  # (batch_size, seq_len, r)

        # Reshape and transpose for multi-head attention
        # New shape: (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # If caching is enabled and cache is provided, concatenate cached keys and values
        if self.use_cache and cache is not None:
            if 'keys' in cache and 'values' in cache:
                K = torch.cat([cache['keys'], K], dim=2)  # Concatenate on seq_len dimension
                V = torch.cat([cache['values'], V], dim=2)
            cache['keys'] = K
            cache['values'] = V

        # Compute scaled dot-product attention
        # Attention scores: (batch_size, n_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)  # Apply dropout to attention weights

        # Compute attention output
        # Shape: (batch_size, n_heads, seq_len, d_k)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate multi-head outputs
        # Reshape to (batch_size, seq_len, r)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.r)

        # Project back to the original model dimension
        output = self.output_proj(attn_output)  # (batch_size, seq_len, d_model)

        if self.use_cache:
            return output, cache
        else:
            return output
