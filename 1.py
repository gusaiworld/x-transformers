from __future__ import annotations
from typing import Callable
from x_transformers.x_transformers import (DynamicPositionBias,AlibiPositionalBias,LayerNorm,AdaptiveLayerNorm,ScaleNorm,RMSNorm,AdaptiveRMSNorm,SimpleRMSNorm,LayerScale,Attention,equals,not_equals,Scale,
                                           FeedForward,TransformerWrapper,pad_at_dim,AttentionLayers,ConcatCombine,calc_z_loss,Residual,GRUGating,softclamp,ShiftTokens,GLU,AdaptiveLayerScale)
import math
from random import random, randrange
from packaging import version

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList, ModuleDict
from torch.amp import autocast

from functools import partial, wraps
from collections import namedtuple
from contextlib import nullcontext
from dataclasses import dataclass

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack

from x_transformers.attend import Attend, Intermediates
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

import tqdm
import torch
import torch.optim as optim
from functools import partial, wraps
from x_transformers.x_transformers import XTransformer


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

def exists(val):
    return val is not None

@dataclass
class LayerIntermediates:
    hiddens:            list[Tensor] | None = None   # all hiddens, before the final norm (in pre-norm architecture)
    last_hidden:        Tensor | None = None         # very last hidden after all attention layers, after the final norm
    attn_intermediates: list[Intermediates] | None = None
    layer_hiddens:      list[Tensor] | None = None
    attn_z_loss:        Tensor | None = None
    mems:               Tensor | None = None
    memory_tokens:      Tensor | None = None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def first(it):
    return it[0]

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

def is_empty(x):
    return len(x) == 0

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth

def divisible_by(num, den):
    return (num % den) == 0

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def at_most_one_of(*bools):
    return sum(map(int, bools)) <= 1



def pick_and_pop(keys, d):
    values = tuple(d.pop(key) for key in  keys)
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return tuple(return_val)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    prefix_len = len(prefix)
    kwargs_without_prefix = {key[prefix_len:]: value for key, value in kwargs_with_prefix.items()}
    return kwargs_without_prefix, kwargs

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast('cuda', enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[-seq_len:, :]
    scale = scale[-seq_len:, :] if isinstance(scale, torch.Tensor) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t, t_unrotated), dim = -1)

    return out.type(orig_dtype)


def masked_mean(t, mask = None, dim = 1):
    if not exists(mask):
        return t.mean(dim = dim)

    dims_append = (1,) * (t.ndim - mask.ndim)
    mask = mask.reshape(*mask.shape, *dims_append)

    num = (t * mask).sum(dim = dim)
    den = mask.sum(dim = dim).clamp(min = 1.)
    return num / den

class ReluSquared(Module):
    def forward(self, x):
        return F.relu(x) ** 2

def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        mask_value = max_neg_value(logits)
        logits = logits.masked_fill(~mask, mask_value)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = torch.arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

class ReluSquared(Module):
    def forward(self, x):
        return F.relu(x) ** 2

# embedding

class TokenEmbedding(Module):
    def __init__(self, dim, num_tokens, l2norm_embed = False):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x.long())
        return l2norm(token_emb) if self.l2norm_embed else token_emb

    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.emb.weight, std=1e-5)
            return
        nn.init.kaiming_normal_(self.emb.weight)

class AbsolutePositionalEmbedding(Module):
    def __init__(self, dim, max_seq_len, l2norm_embed = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        if exists(seq_start_pos):
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb

class ScaledSinusoidalEmbedding(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert divisible_by(dim, 2)
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale

class always():
    def __init__(self, val):
        self.val = val
    def __call__(self, *args, **kwargs):
        return self.val

class LayerNorm(Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()
        self.unit_offset = unit_offset

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.gamma = nn.Parameter(torch.ones(dim))
        nn.init.constant_(self.gamma, 1. - float(unit_offset))

    def forward(self, x):
        normed = self.ln(x)
        gamma = self.gamma + float(self.unit_offset)
        return normed * gamma

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device = device)
        return self.forward(t)

    @autocast('cuda', enabled = False)
    def forward(self, t):
        max_pos = t.max() + 1

        freqs = torch.einsum('i , j -> i j', t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
        freqs = torch.stack((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, '... d r -> ... (d r)')

        if not exists(self.scale):
            return freqs, 1.

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.stack((scale, scale), dim = -1)
        scale = rearrange(scale, '... d r -> ... (d r)')

        return freqs, scale

class RelativePositionBias(Module):
    def __init__(self, scale, causal = False, num_buckets = 32, max_distance = 128, heads = 8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        device = self.device
        q_pos = torch.arange(j - i, j, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = einx.subtract('j, i -> i j', k_pos, q_pos)
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')
        return bias * self.scale
'''
class AttentionLayers(Module):
    def __init__(
        self,
        dim,
        depth = None,
        heads = 8,
        causal = False,
        cross_attend = False,
        only_cross = False,
        use_scalenorm = False,
        use_rmsnorm = False,
        use_simple_rmsnorm = False,
        use_adaptive_layernorm = False,
        use_adaptive_rmsnorm = False,
        use_adaptive_layerscale = False, # paired with use_adaptive_layernorm for ada-ln-zero from DiT paper
        norm_add_unit_offset = True,
        dim_condition = None,
        adaptive_condition_mlp = False,
        adaptive_condition_mlp_expansion = 4,
        alibi_pos_bias = False,
        alibi_num_heads = None,
        rel_pos_bias = False,
        rel_pos_num_buckets = 32,
        rel_pos_max_distance = 128,
        dynamic_pos_bias = False,
        dynamic_pos_bias_log_distance = False,
        dynamic_pos_bias_mlp_depth = 2,
        dynamic_pos_bias_norm = False,
        rotary_pos_emb = False,
        rotary_emb_dim = None,
        rotary_xpos = False,
        rotary_interpolation_factor = 1.,
        rotary_xpos_scale_base = 512,
        rotary_base_rescale_factor = 1.,
        weight_tie_layers = False,
        custom_layers: tuple[str, ...] | None = None,
        layers_execute_order: tuple[int, ...] | None = None,
        sandwich_coef = None,
        par_ratio = None,
        residual_attn = False,
        cross_residual_attn = False,
        macaron = False,
        pre_norm = True,
        pre_norm_has_final_norm = True,
        gate_residual = False,
        scale_residual = False,
        scale_residual_constant = 1.,
        shift_tokens = 0,
        sandwich_norm = False,
        softclamp_output = False,
        softclamp_output_value = 30.,
        resi_dual = False,
        resi_dual_scale = 1.,
        zero_init_branch_output = False,
        layer_dropout = 0.,
        cross_attn_tokens_dropout = 0.,
        disable_abs_pos_emb = None,
        use_layerscale = False,
        layerscale_init_value = 0.,
        unet_skips = False,
        reinject_input = False,         # seen first in DEQ paper https://arxiv.org/abs/1909.01377, but later used in a number of papers trying to achieve depthwise generalization https://arxiv.org/abs/2410.03020v1
        add_value_residual = False,     # resformer from Zhou et al - https://arxiv.org/abs/2410.17897v1
        rel_pos_kwargs: dict = dict(),
        **kwargs
    ):
        super().__init__()
        rotary_pos_emb = rotary_pos_emb or rotary_xpos

        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim('attn_', kwargs)
        cross_attn_kwargs, kwargs = groupby_prefix_and_trim('cross_attn_', kwargs)

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)
        data_dependent_alibi = attn_kwargs.get('data_dependent_alibi', False)
        neutreno_value_residual = attn_kwargs.get('neutreno_value_residual', False)

        assert len(kwargs) == 0, f'unrecognized kwargs passed in {kwargs.keys()}'

        add_value_residual |= neutreno_value_residual

        self.dim = dim
        self.causal = causal
        self.layers = ModuleList([])

        self.disable_abs_pos_emb = default(disable_abs_pos_emb, (rel_pos_bias or rotary_pos_emb))

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)

        assert not (rotary_xpos and not causal), 'rotary xpos is not compatible with bidirectional attention'
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim, use_xpos = rotary_xpos, scale_base = rotary_xpos_scale_base, interpolation_factor = rotary_interpolation_factor, base_rescale_factor = rotary_base_rescale_factor) if rotary_pos_emb else None

        assert at_most_one_of(alibi_pos_bias, rel_pos_bias, data_dependent_alibi), 'you can only choose one of Alibi positional bias, data dependent Alibi (forgetting transformers), or T5 relative positional bias'
        assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'

        # relative positional bias

        flash_attn = attn_kwargs.get('flash', False)
        assert at_most_one_of(rel_pos_bias, dynamic_pos_bias, alibi_pos_bias), 'you can only choose up to one of t5, alibi, or dynamic positional bias'

        self.rel_pos = None

        if rel_pos_bias:
            assert not flash_attn, 'flash attention not compatible with t5 relative positional bias'
            self.rel_pos = RelativePositionBias(scale = dim_head ** 0.5, causal = causal, heads = heads, num_buckets = rel_pos_num_buckets, max_distance = rel_pos_max_distance, **rel_pos_kwargs)
        elif dynamic_pos_bias:
            assert not flash_attn, 'flash attention not compatible with dynamic positional bias'
            self.rel_pos = DynamicPositionBias(dim = dim // 4, heads = heads, log_distance = dynamic_pos_bias_log_distance, depth = dynamic_pos_bias_mlp_depth, norm = dynamic_pos_bias_norm, **rel_pos_kwargs)
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert alibi_num_heads <= heads, 'number of ALiBi heads must be less than the total number of heads'
            self.rel_pos = AlibiPositionalBias(heads = alibi_num_heads, total_heads = heads, **rel_pos_kwargs)

        assert at_most_one_of(sandwich_norm, resi_dual), 'either sandwich norm or resiDual is selected, but not both'
        assert not (not pre_norm and sandwich_norm), 'sandwich norm cannot be used when not using prenorm'

        if resi_dual:
            pre_norm = False

        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.resi_dual = resi_dual
        assert 0 < resi_dual_scale <= 1., 'resiDual prenorm residual must be scaled by a factor greater than 0 and less than or equal to 1.'
        self.resi_dual_scale = resi_dual_scale

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        assert not (flash_attn and (residual_attn or cross_residual_attn)), 'flash attention is not compatible with residual attention'

        self.cross_attend = cross_attend

        # determine norm

        assert at_most_one_of(use_scalenorm, use_rmsnorm, use_simple_rmsnorm, use_adaptive_layernorm, use_adaptive_rmsnorm), 'you can only use either scalenorm, rmsnorm, adaptive layernorm, adaptive rmsnorm, or simple rmsnorm'

        norm_need_condition = False
        dim_condition = default(dim_condition, dim)
        dim_condition_mult = 1

        if adaptive_condition_mlp:
            dim_condition_mult = adaptive_condition_mlp_expansion

        if use_scalenorm:
            norm_class = ScaleNorm
        elif use_rmsnorm:
            norm_class = RMSNorm
        elif use_simple_rmsnorm:
            norm_class = SimpleRMSNorm
        elif use_adaptive_layernorm:
            norm_need_condition = True
            norm_class = partial(AdaptiveLayerNorm, dim_condition = dim_condition * dim_condition_mult)
        elif use_adaptive_rmsnorm:
            norm_need_condition = True
            norm_class = partial(AdaptiveRMSNorm, dim_condition = dim_condition * dim_condition_mult)
        else:
            norm_class = LayerNorm

        norm_fn = partial(norm_class, dim)

        if not norm_need_condition and norm_add_unit_offset:
            # researcher Ohad Rubin shares in a blog post by adding an offset to gammas, they can be subjected to weight decay safely
            norm_fn = partial(norm_fn, unit_offset = True)

        self.norm_need_condition = norm_need_condition
        self.dim_condition = dim_condition

        # determine default block layer type order

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        # determine post branch wrapper

        assert at_most_one_of(use_layerscale, use_adaptive_layerscale)

        post_branch_fn = None
        post_branch_fn_needs_condition = False

        if use_layerscale:
            post_branch_fn = partial(LayerScale, dim = dim, init_value = layerscale_init_value)
        elif use_adaptive_layerscale:
            post_branch_fn = partial(AdaptiveLayerScale, dim = dim, dim_condition = dim_condition * dim_condition_mult)
            post_branch_fn_needs_condition = True

        self.post_branch_fn_needs_condition = post_branch_fn_needs_condition

        if exists(post_branch_fn) and not post_branch_fn_needs_condition and norm_add_unit_offset:
            post_branch_fn = partial(post_branch_fn, unit_offset = True)

        # setup mlp for conditioning

        self.need_condition = norm_need_condition or post_branch_fn_needs_condition

        self.adaptive_mlp = nn.Identity()

        if self.need_condition and adaptive_condition_mlp:
            self.adaptive_mlp = nn.Sequential(
                LinearNoBias(dim_condition, dim_condition * dim_condition_mult),
                nn.SiLU()
            )

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, 'zero_init_output':  True}
            ff_kwargs = {**ff_kwargs, 'zero_init_output':  True}

        # setup weight tying, which is a special case of `layer_execute_order`

        assert not (exists(layers_execute_order) and exists(custom_layers) and exists(depth)), 'depth should not be passed in if using custom layers and custom layer execution order'

        assert not (weight_tie_layers and any([*map(exists, (custom_layers, par_ratio, sandwich_coef))]))

        if weight_tie_layers:
            assert exists(depth), 'depth must be passed in with `weight_tie_layers` = True'
            assert not exists(layers_execute_order)
            layers_execute_order = tuple(range(len(default_block))) * depth
            depth = 1

        # calculate layer block order

        len_default_block = 1

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn  = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            assert exists(depth), '`depth` must be passed in for `Decoder` or `Encoder`'
            layer_types = default_block * depth
            len_default_block = len(default_block)

        self.layer_types = layer_types
        self.layers_execute_order = default(layers_execute_order, tuple(range(len(layer_types))))

        assert all([i < len(self.layer_types) for i in self.layers_execute_order])

        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        # set the depth

        depth = default(depth, len(self.layers_execute_order))
        self.depth = depth

        # stochastic depth

        self.layer_dropouts = cast_tuple(layer_dropout, len(layer_types))

        # structured dropout for cross attending

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        # optional soft clamping just before the final norm
        # used in gemma 2

        self.softclamp_output = softclamp_output
        self.softclamp_output_value = softclamp_output_value

        # whether it has post norm

        self.final_norm = norm_fn() if pre_norm or resi_dual else nn.Identity()

        # whether unet or not

        self.unet_skips = unet_skips
        num_skips = self.depth // len_default_block

        assert not (unet_skips and num_skips == 0), 'must have depth of at least 2 for unet skip connections'

        skip_indices = [i * len_default_block for i in range(num_skips)]

        self.skip_combines = ModuleList([])

        # whether there is reinjection of input at every layer

        self.reinject_input = reinject_input
        self.reinject_input_proj = nn.Linear(dim, dim, bias = False) if reinject_input else None

        # add the value from the first self attention block to all latter projected self attention values as a residual

        self.add_value_residual = add_value_residual

        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(zip(self.layer_types, shift_tokens)):

            # `ind` is the index of each module - attention, feedforward, cross attention
            # but `block_ind` refers to the typical enumeration of a transformer block (attn + ff + [optional] cross attn)

            block_begin = divisible_by(ind, len_default_block)
            block_ind = ind // len_default_block

            is_last_layer = ind == (len(self.layer_types) - 1)

            # attention, cross attention, feedforward

            if layer_type == 'a':
                layer = Attention(dim, heads = heads, causal = causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads = heads, **{**attn_kwargs, **cross_attn_kwargs})
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            if exists(post_branch_fn):
                layer = post_branch_fn(layer)

            residual_fn = GRUGating if gate_residual else Residual

            residual = residual_fn(dim, scale_residual = scale_residual, scale_residual_constant = scale_residual_constant)

            # handle unet skip connection

            skip_combine = None
            is_latter_half = block_begin and block_ind >= (self.depth / 2)

            if self.unet_skips and is_latter_half:
                skip_combine = ConcatCombine(dim, skip_indices.pop())

            # all normalizations of the layer

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = norm_fn() if sandwich_norm else None
            post_main_norm = norm_fn() if not pre_norm else None

            norms = ModuleList([
                pre_branch_norm,
                post_branch_norm,
                post_main_norm
            ])

            self.skip_combines.append(skip_combine)

            self.layers.append(ModuleList([
                norms,
                layer,
                residual
            ]))

        # determine whether can cache kv

        self.can_cache_kv = all([module.can_cache_kv for module in self.modules() if isinstance(module, Attention) ])

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        self_attn_kv_mask = None,
        mems = None,
        mem_masks = None,
        seq_start_pos: Tensor | None = None,
        cache: LayerIntermediates | None = None,
        cache_age = 1,
        return_hiddens = False,
        rotary_pos_emb = None,
        pos = None,
        attn_bias = None,
        condition = None,
        in_attn_cond = None, # https://arxiv.org/abs/2105.04090
        layers_execute_order: tuple[int, ...] | None = None
    ):
        assert not (self.cross_attend ^ exists(context)), 'context must be passed in if cross_attend is set to True'
        assert not (exists(condition) ^ self.need_condition), 'condition needs to be passed in if using adaptive layernorm or vice versa'

        # handle condition

        if exists(condition):
            assert condition.shape[-1] == self.dim_condition, f'expected condition dimension of {self.dim_condition} but received {condition.shape[-1]}'

            assert condition.ndim in {2, 3}

            if condition.ndim == 2:
                condition = rearrange(condition, 'b d -> b 1 d')

            condition = self.adaptive_mlp(condition)

        # setup maybe layernorm kwarg

        norm_kwargs = dict()

        if self.norm_need_condition:
            norm_kwargs.update(condition = condition)

        # maybe post branch fn conditioning (DiT paper's ada-ln-zero)

        block_forward_kwargs = dict()

        if self.post_branch_fn_needs_condition:
            block_forward_kwargs.update(condition = condition)

        # initialize accums

        hiddens = []
        layer_hiddens = []
        intermediates = []

        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers
        mem_masks = mem_masks.copy() if exists(mem_masks) else [None] * self.num_attn_layers

        # handle left padded sequences

        if exists(seq_start_pos):
            seq_arange = torch.arange(x.shape[-2], device = x.device, dtype = torch.long)
            left_pad_mask = seq_arange >= seq_start_pos[..., None]

            if exists(self_attn_kv_mask):
                self_attn_kv_mask = self_attn_kv_mask & left_pad_mask
            else:
                self_attn_kv_mask = left_pad_mask

        # rotary positions

        if not exists(rotary_pos_emb) and exists(self.rotary_pos_emb):
            maybe_mem = mems[0]
            mem_len = maybe_mem.shape[1] if exists(maybe_mem) else 0

            if not exists(pos):
                pos = torch.arange(x.shape[1] + mem_len, device = x.device) - mem_len

            rotary_pos_emb = self.rotary_pos_emb(pos)

        # assume cached key / values

        attn_cache = []

        if exists(cache):
            assert not self.training and self.causal and not any([*map(exists, (mask, attn_mask))])

            if exists(context):
                context = context[:, :0]

            if cache_age > 0:
                x = x[:, -cache_age:] # for spec decoding, may be greater than 1

            attn_cache = cache.attn_intermediates

        iter_attn_cache = iter(attn_cache)

        # outer residual - for resiDual paper

        outer_residual = x * self.resi_dual_scale

        # get layers to be executed

        layer_variables = (
            self.layer_types,
            self.skip_combines,
            self.layers,
            self.layer_dropouts
        )

        # able to override the layers execution order on forward, for trying to depth extrapolate

        layers_execute_order = default(layers_execute_order, self.layers_execute_order)

        layer_variables = tuple(tuple(layer_variable[i] for i in layers_execute_order) for layer_variable in layer_variables)

        # derived input for reinjection if needed

        if self.reinject_input:
            assert not exists(in_attn_cond)
            inp_inject = self.reinject_input_proj(x)

        elif exists(in_attn_cond):
            # handle in-attention conditioning, which serves the same purpose of having the network learn the residual
            inp_inject = in_attn_cond if in_attn_cond.ndim == 3 else rearrange(in_attn_cond, 'b d -> b 1 d')

        # store all hiddens for skips

        skip_hiddens = []

        # for value residuals

        first_self_attn_inter = None
        first_cross_attn_inter = None

        # go through the attention and feedforward layers

        for ind, (layer_type, skip_combine, (norm, block, residual_fn), layer_dropout) in enumerate(zip(*layer_variables)):
            is_last = ind == (len(self.layers) - 1)

            # handle skip connections

            skip_hiddens.append(x)

            if exists(skip_combine):
                x = skip_combine(x, skip_hiddens)

            # layer dropout

            if self.training and layer_dropout > 0. and random() < layer_dropout:
                continue

            if layer_type == 'a':
                if return_hiddens:
                    hiddens.append(x)

                layer_mem = mems.pop(0) if mems else None
                layer_mem_mask = mem_masks.pop(0) if mem_masks else None

            if layer_type == 'c':
                if self.training and self.cross_attn_tokens_dropout > 0.:
                    context, context_mask = dropout_seq(context, context_mask, self.cross_attn_tokens_dropout)

            inner_residual = x

            if return_hiddens:
                layer_hiddens.append(x)

            pre_norm, post_branch_norm, post_main_norm = norm

            if self.need_condition:
                pre_norm = maybe(partial)(pre_norm, **norm_kwargs)
                post_branch_norm = maybe(partial)(post_branch_norm, **norm_kwargs)
                post_main_norm = maybe(partial)(post_main_norm, **norm_kwargs)

            if self.reinject_input:
                x = x + inp_inject

            if exists(pre_norm):
                x = pre_norm(x)

                if layer_type == 'a' and exists(layer_mem):
                    layer_mem = pre_norm(layer_mem)

            block = partial(block, **block_forward_kwargs)

            # handle maybe value residuals

            maybe_self_attn_value_residual = None
            maybe_cross_attn_value_residual = None

            if self.add_value_residual:
                if exists(first_self_attn_inter):
                    maybe_self_attn_value_residual = first_self_attn_inter.values

                if exists(first_cross_attn_inter):
                    maybe_cross_attn_value_residual = first_cross_attn_inter.values

            # forward depending on layer type

            if layer_type == 'a':
                out, inter = block(x, mask = mask, context_mask = self_attn_kv_mask, attn_mask = attn_mask, rel_pos = self.rel_pos, pos = pos, rotary_pos_emb = rotary_pos_emb, prev_attn = prev_attn, cache = next(iter_attn_cache, None), mem = layer_mem, mem_mask = layer_mem_mask, attn_bias = attn_bias, value_residual = maybe_self_attn_value_residual, return_intermediates = True)
            elif layer_type == 'c':
                out, inter = block(x, context = context, mask = mask, context_mask = context_mask, prev_attn = prev_cross_attn, cache = next(iter_attn_cache, None), value_residual = maybe_cross_attn_value_residual, return_intermediates = True)
            elif layer_type == 'f':
                out = block(x)

            # store first self or cross attention intermediate for value residual

            if not exists(first_self_attn_inter) and layer_type == 'a':
                first_self_attn_inter = inter

            if not exists(first_cross_attn_inter) and layer_type == 'c':
                first_cross_attn_inter = inter

            if self.resi_dual:
                outer_residual = outer_residual + out * self.resi_dual_scale

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, inner_residual)

            if layer_type in ('a', 'c') and return_hiddens:
                inter.layer_type = layer_type
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            layer_hiddens.append(x)

        if self.softclamp_output:
            x = softclamp(x, self.softclamp_output_value)

        final_norm = self.final_norm

        if self.need_condition:
            final_norm = maybe(partial)(final_norm, **norm_kwargs)

        if self.resi_dual:
            x = x + final_norm(outer_residual)
        else:
            x = final_norm(x)

        if not return_hiddens:
            return x

        intermediates = LayerIntermediates(
            hiddens = hiddens,
            last_hidden = x,
            attn_intermediates = intermediates,
            layer_hiddens = layer_hiddens,
        )

        return x, intermediates
'''

class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal = False, **kwargs)

class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(causal = True, **kwargs)

class transformerwrap(Module):
    def __init__(self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers: AttentionLayers,
        embed_num_tokens: dict[str, int] = dict(),
        emb_dim = None,
        max_mem_len = 0,
        shift_mem_down = 0,
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        tie_embedding = False,
        logits_dim = None,
        return_only_embed = False,
        num_output_heads = 1,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False,
        l2norm_embed = False,
        recycling = False,            # from Jumper et al. - Alphafold2
        train_max_recycle_steps = 4,  # saw a benefit for language modeling up to 3 recycling steps, so let's default this to 4
        emb_frac_gradient = 1.,       # GLM-130B and Cogview successfully used this, set at 0.1
        attn_z_loss_weight = 1e-4,
        average_pool_embed = False,
        use_cls_token = False,
        num_cls_tokens = 1,
        squeeze_out_last_dim = False,
        token_emb: TokenEmbedding | None = None,
        mixture_of_softmax = False,
        mixture_of_softmax_k = 4,
        sigsoftmax_logits = False,
        to_logits: Module | None = None,):
        super().__init__()
        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)
        self.num_tokens=num_tokens,
        self.max_seq_len=max_seq_len,
        #self.attn_layers=attn_layers,
        self.emb_dim=emb_dim,
        self.l2norm_embed = l2norm_embed
        self.return_only_embed = return_only_embed,

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.post_emb_norm = LayerNorm(emb_dim) if post_emb_norm else nn.Identity()

        self.embeds = None
        if not exists(token_emb):
            token_emb = TokenEmbedding(emb_dim, num_tokens, l2norm_embed=l2norm_embed)

        self.token_emb = token_emb

        if use_abs_pos_emb:
            self.pos_emb=AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed = l2norm_embed)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(emb_dim)
        else:
            self.pos_emb = always(0)

        if len(embed_num_tokens) > 0:
            self.embeds = ModuleDict(
                {f'{name}_embed': nn.Embedding(num_tokens, emb_dim) for name, num_tokens in embed_num_tokens.items()})

        self.post_emb_norm = LayerNorm(emb_dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers

        self.init_()

        self.recycling = recycling
        self.train_max_recycle_steps = train_max_recycle_steps

        self.average_pool_embed = average_pool_embed

        self.to_mixture = None
        self.combine_mixture = None

        if mixture_of_softmax:
            assert num_output_heads == 1

            self.to_mixture = Sequential(
                LinearNoBias(dim, dim * mixture_of_softmax_k),
                Rearrange('... (k d) -> ... k d', k=mixture_of_softmax_k)
            )

            self.combine_mixture = LinearNoBias(dim, mixture_of_softmax_k)

        logits_dim = default(logits_dim, num_tokens)
        self.has_multiple_heads = num_output_heads > 1

        if return_only_embed:
            self.to_logits = None
        elif tie_embedding:
            assert isinstance(token_emb, TokenEmbedding), 'can only tie embedding if using `TokenEmbedding`'
            self.to_logits = lambda t: t @ self.token_emb.emb.weight.t()
        elif num_output_heads > 1:
            self.to_logits = ModuleList([LinearNoBias(dim, logits_dim) for _ in range(num_output_heads)])
        else:
            self.to_logits = LinearNoBias(dim, logits_dim) if not exists(to_logits) else to_logits

    def init_(self):
        if hasattr(self.token_emb, 'init_'):
            self.token_emb.init_()

        if self.l2norm_embed:
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)

    def forward(  self,
        x,
        return_embeddings = False,
        return_logits_and_embeddings = False,
        return_intermediates = False,
        mask = None,
        return_mems = False,
        return_attn = False,
        mems = None,
        mem_masks = None,
        recycle_steps = None,
        pos = None,
        prepend_embeds = None,
        prepend_mask = None,
        embed_ids: dict[str, Tensor] = dict(),
        sum_embeds = None,
        return_attn_z_loss = False,
        attn_z_loss_weight = 1e-4,
        seq_start_pos = None,
        cache: LayerIntermediates | None = None,
        token_emb_kwargs = dict(),
        to_logits_kwargs = dict(),
        **kwargs,):
        orig_mask=mask

        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = self.pos_emb(x, pos=pos, seq_start_pos=seq_start_pos) if not external_pos_emb else pos
        x = self.token_emb(x, **token_emb_kwargs) + pos_emb

        assert not (exists(self.embeds) ^ (
                    len(embed_ids) > 0)), '`embed_num_tokens` must be defined on `TransformerWrapper`'

        if exists(self.embeds):
            assert len(embed_ids) == len(self.embeds)

            for name, embed_id in embed_ids.items():
                embed_key = f'{name}_embed'

                assert embed_key in self.embeds
                embed = self.embeds[embed_key](embed_id)

                x = x + embed

        x = self.post_emb_norm(x)

        x = self.emb_dropout(x)

        x = self.project_emb(x)


        if not self.recycling:
            assert not exists(recycle_steps) or recycle_steps == 1, 'you did not train with recycling'

            # regular

            attended, intermediates = self.attn_layers(x, mask=mask, mems=mems, mem_masks=mem_masks, cache=cache,
                                                       return_hiddens=True, seq_start_pos=seq_start_pos, **kwargs)

        else:
            # recycling

            recycle_steps = default(recycle_steps,
                                    (randrange(self.train_max_recycle_steps) + 1) if self.training else None)
            assert exists(
                recycle_steps) and recycle_steps > 0, '`recycle_steps` must be provided on forward if recycling is turned on and not training'

            for i in range(recycle_steps):
                first_step = i == 0
                last_step = i == (recycle_steps - 1)

                context = nullcontext if last_step else torch.no_grad

                with context():
                    maybe_recycled = self.recycled_proj(attended.detach()) if not first_step else 0.

                    attended, intermediates = self.attn_layers(x + maybe_recycled, mask=mask, mems=mems,
                                                               mem_masks=mem_masks, cache=cache, return_hiddens=True,
                                                               seq_start_pos=seq_start_pos, **kwargs)

        x = attended

        if self.average_pool_embed:
            x = masked_mean(x, mask = orig_mask, dim = 1)

        combine_mixture = None

        if exists(self.to_mixture):
            combine_mixture = self.combine_mixture(x).softmax(dim=-1)
            x = self.to_mixture(x)

        if not return_embeddings:
            if self.has_multiple_heads:
                logits = tuple(fn(x, **to_logits_kwargs) for fn in self.to_logits)
            else:
                logits = self.to_logits(x, **to_logits_kwargs)

        if exists(combine_mixture):
            with autocast('cuda', enabled = False):
                prob = logits.softmax(dim = -1)
                mos = einsum('... k d, ... k -> ... d', prob, combine_mixture)
                logits = log(mos)

        if return_logits_and_embeddings:
            out = (logits, x)
        elif return_embeddings:
            out = x
        else:
            out = logits
        if return_intermediates:
            return out, intermediates

        if return_attn_z_loss:
            pre_softmax_attns = [t.pre_softmax_attn for t in intermediates.attn_intermediates]
            intermediates.attn_z_loss = calc_z_loss(pre_softmax_attns, weight=attn_z_loss_weight)
            return_intermediates = True


        if return_attn:
            attn_maps = [t.post_softmax_attn for t in intermediates.attn_intermediates]
            return out, attn_maps

        return out

class xtransformer(Module):
    def __init__(self,
        *,dim,
                 tie_token_emb=False,
                 ignore_index=-100,
                 pad_value=0,
                 cross_attn_tokens_dropout=0.,
         **kwargs):
        super().__init__()
        enc_kwargs, kwargs = groupby_prefix_and_trim('enc_', kwargs)# num_token , max_seq_length , depth , head
        dec_kwargs, kwargs = groupby_prefix_and_trim('dec_', kwargs)

        assert 'dim' not in enc_kwargs and 'dim' not in dec_kwargs, 'dimension of either encoder or decoder must be set with `dim` keyword'
        enc_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], enc_kwargs)
        enc_transformer_kwargs['emb_dropout'] = enc_kwargs.pop('emb_dropout', 0)
        enc_transformer_kwargs['num_memory_tokens'] = enc_kwargs.pop('num_memory_tokens', None)
        enc_transformer_kwargs['scaled_sinu_pos_emb'] = enc_kwargs.pop('scaled_sinu_pos_emb', False)
        enc_transformer_kwargs['use_abs_pos_emb'] = enc_kwargs.pop('use_abs_pos_emb', True)

        dec_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec_kwargs)
        dec_transformer_kwargs['emb_dropout'] = dec_kwargs.pop('emb_dropout', 0)
        dec_transformer_kwargs['scaled_sinu_pos_emb'] = dec_kwargs.pop('scaled_sinu_pos_emb', False)
        dec_transformer_kwargs['use_abs_pos_emb'] = dec_kwargs.pop('use_abs_pos_emb', True)

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout

        self.encoder = transformerwrap(**enc_transformer_kwargs,return_only_embed = True,attn_layers = Encoder(dim = dim, **enc_kwargs))
        self.decoder = transformerwrap(**dec_transformer_kwargs,attn_layers = Decoder(dim=dim , cross_attend = True, **dec_kwargs))
        if tie_token_emb:
            self.decoder.token_emb = self.encoder.token_emb
        self.decoder = AutoregressiveWrapper(self.decoder, ignore_index=-100, pad_value=0)

    @torch.no_grad()
    def generate(self, seq_in, seq_out_start, seq_len, mask = None, attn_mask = None, **kwargs):
        encodings = self.encoder(seq_in, mask = mask, attn_mask = attn_mask, return_embeddings = True)
        return self.decoder.generate(seq_out_start, seq_len, context = encodings, context_mask = mask, **kwargs)

    def forward(self,src,tgt,mask = None,attn_mask = None,src_prepend_embeds = None):
        enc_out = self.encoder(src,mask=mask, attn_mask = attn_mask,  return_embeddings = True)
        if exists(src_prepend_embeds) and exists(mask):
            mask = pad_at_dim(mask, (src_prepend_embeds.shape[-2], 0), dim = -1, value = True)

        if self.training and self.cross_attn_tokens_dropout > 0:
            enc, mask = dropout_seq(enc_out, mask, self.cross_attn_tokens_dropout)
        dec_out = self.decoder(tgt,context = enc_out,mask=mask)
        return dec_out
def main():
    # constants

    NUM_BATCHES = int(1e4)
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    GENERATE_EVERY = 100
    NUM_TOKENS = 16 + 2
    ENC_SEQ_LEN = 32
    DEC_SEQ_LEN = 64 + 1

    # helpers

    def cycle():
        while True:
            prefix = torch.ones((BATCH_SIZE, 1)).long().cuda()
            src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long().cuda()
            tgt = torch.cat((prefix, src, src), 1)
            src_mask = torch.ones(BATCH_SIZE, src.shape[1]).bool().cuda()
            yield (src, tgt, src_mask)

    # instantiate model

    model = XTransformer(
        dim=512,
        tie_token_emb=True,
        return_tgt_loss=True,
        enc_num_tokens=NUM_TOKENS,
        enc_depth=3,
        enc_heads=8,
        enc_max_seq_len=ENC_SEQ_LEN,
        dec_num_tokens=NUM_TOKENS,
        dec_depth=3,
        dec_heads=8,
        dec_max_seq_len=DEC_SEQ_LEN
    ).cuda()

    # optimizer

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training

    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
        model.train()

        src, tgt, src_mask = next(cycle())

        loss = model(src, tgt, mask=src_mask)
        loss.backward()
        print(f'{i}: {loss.item()}')

        optim.step()
        optim.zero_grad()

        if i != 0 and i % GENERATE_EVERY == 0:
            model.eval()
            src, _, src_mask = next(cycle())
            src, src_mask = src[:1], src_mask[:1]
            start_tokens = (torch.ones((1, 1)) * 1).long().cuda()

            sample = model.generate(src, start_tokens, ENC_SEQ_LEN, mask=src_mask)
            incorrects = (src != sample).abs().sum()

            print(f"input:  ", src)
            print(f"predicted output:  ", sample)
            print(f"incorrects: {incorrects}")


if __name__ == '__main__':
    LinearNoBias = partial(nn.Linear, bias=False)
    DEFAULT_DIM_HEAD = 64
    main()
