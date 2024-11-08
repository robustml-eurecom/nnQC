import torch
import torch.nn as nn
import math
import xformers

class DualCrossAttention(nn.Module):
    """
    A cross attention layer.

    Args:
        query_dim: number of channels in the query.
        cross_attention_dim: number of channels in the context.
        num_attention_heads: number of heads to use for multi-head attention.
        num_head_channels: number of channels in each head.
        dropout: dropout probability to use.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        num_attention_heads: int = 8,
        num_head_channels: int = 64,
        dropout: float = 0.0,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention
        inner_dim = num_head_channels * num_attention_heads
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = 1 / math.sqrt(num_head_channels)
        self.num_heads = num_attention_heads

        self.upcast_attention = upcast_attention
        '''
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        '''
        # Query projection is always needed
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        
        # Single k,v for self-attention
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        
        # Separate k,v for mask/image cross-attention
        self.to_q_mask = nn.Linear(self.cross_attention_dim, inner_dim, bias=False)
        #self.to_k_mask = nn.Linear(self.cross_attention_dim, inner_dim, bias=False)
        #self.to_v_mask = nn.Linear(self.cross_attention_dim, inner_dim, bias=False)
        self.to_k_image = nn.Linear(self.cross_attention_dim, inner_dim, bias=False)
        self.to_v_image = nn.Linear(self.cross_attention_dim, inner_dim, bias=False)

        self.mixing_param = nn.Parameter(torch.tensor(0.5))

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))


    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide hidden state dimension to the multiple attention heads and reshape their input as instances in the batch.
        """
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, dim // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine the output of the attention heads back into the hidden state dimension."""
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // self.num_heads, self.num_heads, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // self.num_heads, seq_len, dim * self.num_heads)
        return x

    def _memory_efficient_attention_xformers(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        x = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        return x

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype=dtype)

        x = torch.bmm(attention_probs, value)
        return x

    '''def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        query = self.to_q(x)
        context = context if context is not None else x
        key = self.to_k(context)
        value = self.to_v(context)

        # Multi-Head Attention
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(query, key, value)
        else:
            x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = x.to(query.dtype)

        return self.to_out(x)
    '''

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        query = self.to_q(x)
        query = self.reshape_heads_to_batch_dim(query)
        context = context if context is not None else x
        
        if context.shape[2] == (self.cross_attention_dim * 2) and context is not None:
            # Case 3: Dual-condition cross-attention
            image_emb, mask_emb = torch.split(context, context.shape[2] // 2, dim=-1)
            # Process both conditions
            query_mask = self.to_q_mask(mask_emb)
            #key_mask = self.to_k_mask(mask_emb)
            #value_mask = self.to_v_mask(mask_emb)
            key_image = self.to_k_image(image_emb)
            value_image = self.to_v_image(image_emb)

            query_mask = self.reshape_heads_to_batch_dim(query_mask)
            #key_mask = self.reshape_heads_to_batch_dim(key_mask)
            #value_mask = self.reshape_heads_to_batch_dim(value_mask)
            key_image = self.reshape_heads_to_batch_dim(key_image)
            value_image = self.reshape_heads_to_batch_dim(value_image)
            
            if self.use_flash_attention:
                dual_condition = self._memory_efficient_attention_xformers(query_mask, key_image, value_image)
            else:
                dual_condition = self._attention(query_mask, key_image, value_image)
                
            dual_condition = self.reshape_batch_dim_to_heads(dual_condition)
            dual_condition = dual_condition.to(query_mask.dtype)
            
            key_dual_condition = self.to_k(dual_condition)
            value_dual_condition = self.to_v(dual_condition)
            
            if self.use_flash_attention:
                x = self._memory_efficient_attention_xformers(query, key_dual_condition, value_dual_condition)
            else:
                x = self._attention(query, key_dual_condition, value_dual_condition)
                
            #if self.use_flash_attention:
            #    x_mask = self._memory_efficient_attention_xformers(query, key_mask, value_mask)
            #    x_image = self._memory_efficient_attention_xformers(query, key_image, value_image)
            #else:
            #    x_mask = self._attention(query, key_mask, value_mask)
            #    x_image = self._attention(query, key_image, value_image)

            ## Mix the attention outputs
            #mix_weight = torch.clamp(self.mixing_param, 0.0, 1.0)
            #x = mix_weight * x_mask + (1 - mix_weight) * x_image
            
            # Early return for dual-condition case
            x = self.reshape_batch_dim_to_heads(x)
            x = x.to(query.dtype)
            return self.to_out(x)
        
        else:
            # Case 2: Regular cross-attention
            key = self.to_k(context)
            value = self.to_v(context)

        # Process single attention case (self-attention or regular cross-attention)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(query, key, value)
        else:
            x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = x.to(query.dtype)
        return self.to_out(x)
    
    
class StandardCrossAttention(nn.Module):
    """
    A cross attention layer.

    Args:
        query_dim: number of channels in the query.
        cross_attention_dim: number of channels in the context.
        num_attention_heads: number of heads to use for multi-head attention.
        num_head_channels: number of channels in each head.
        dropout: dropout probability to use.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        num_attention_heads: int = 8,
        num_head_channels: int = 64,
        dropout: float = 0.0,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention
        inner_dim = num_head_channels * num_attention_heads
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = 1 / math.sqrt(num_head_channels)
        self.num_heads = num_attention_heads

        self.upcast_attention = upcast_attention
    
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))


    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide hidden state dimension to the multiple attention heads and reshape their input as instances in the batch.
        """
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, dim // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine the output of the attention heads back into the hidden state dimension."""
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // self.num_heads, self.num_heads, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // self.num_heads, seq_len, dim * self.num_heads)
        return x

    def _memory_efficient_attention_xformers(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        x = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        return x

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype=dtype)

        x = torch.bmm(attention_probs, value)
        return x

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        query = self.to_q(x)
        context = context if context is not None else x
        key = self.to_k(context)
        value = self.to_v(context)

        # Multi-Head Attention
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(query, key, value)
        else:
            x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = x.to(query.dtype)

        return self.to_out(x)
    