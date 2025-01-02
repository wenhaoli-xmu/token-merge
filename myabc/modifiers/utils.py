import torch
import torch.utils.checkpoint
from torch import distributed as dist

from transformers.models.llama.modeling_llama import rotate_half
from functools import partial
from copy import deepcopy


def do_projection(proj, states, num_heads, head_dim):
    return proj(states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)


def apply_rotary_pos_emb(mat, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    mat_embed = (mat * cos) + (rotate_half(mat) * sin)

    return mat_embed


def new_posid(num_token: int, device, dtype, bsz):
    appendix = torch.arange(num_token, device=device)
    appendix = appendix[None,:].expand(bsz, -1)
    return appendix


def check_and_apply_qk_rope(query, key, cos, sin, pos=0):
    batch_size, num_heads, num_query, head_dim = query.shape
    num_kv = key.shape[-2]

    assert key.shape == (batch_size, num_heads, num_kv, head_dim)

    new_posid_spec = partial(new_posid, device=query.device, dtype=query.dtype, bsz=batch_size)
    pos_list = new_posid_spec(max(num_kv, pos))

    Q = apply_rotary_pos_emb(query, cos, sin, pos_list[:,-num_query:])
    K = apply_rotary_pos_emb(key, cos, sin, pos_list[:,-num_kv:])

    return Q, K


class ScoreHead(torch.nn.Module):
    def __init__(self, hidden_size, kernel_size, dtype=None, device=None):
        super().__init__()
        self.hidden_size = hidden_size

        if dtype is None:
            dtype = torch.bfloat16
        
        if device is None:
            device = 'cuda'

        self.conv = torch.nn.Conv1d(
            in_channels=hidden_size, 
            out_channels=64, 
            kernel_size=kernel_size, 
            dtype=dtype,
            device=device)
    
        self.linear = torch.nn.Linear(
            in_features=64, 
            out_features=1,
            dtype=dtype,
            device=device)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.constant_(self.conv.bias, 0.0)
        torch.nn.init.constant_(self.linear.bias, -1.0)


    def forward(self, x):
        # assertions
        assert x.ndim == 3 and x.shape[-1] == self.hidden_size, f"`x` should be 3 dimensional tensor with last dimension {self.hidden_size} but got: {x.shape}"
        assert x.shape[0] == 1, f"only support batch size 1 currently"

        logits = self.conv(x.transpose(-1,-2)).transpose(-1,-2)
        logits = torch.nn.functional.silu(logits)
        logits = self.linear(logits).squeeze(0)

        return logits



def slerp(t: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor, DOT_THRESHOLD: float = 0.9995, eps: float = 1e-8) -> torch.Tensor:
    """
    Spherical linear interpolation between two vectors.

    Args:
        t (torch.Tensor): Interpolation parameter. Float tensor between 0.0 and 1.0.
        v0 (torch.Tensor): Starting vector.
        v1 (torch.Tensor): Final vector.
        DOT_THRESHOLD (float): Threshold for considering the two vectors as colinear. Default is 0.9995.
        eps (float): Small value to avoid division by zero. Default is 1e-8.

    Returns:
        torch.Tensor: Interpolated vector between v0 and v1.
    """
    dtype = v0.dtype

    v0 = v0.double()
    v1 = v1.double()

    # Copy the vectors to reuse them later
    v0_copy = v0.clone().detach()
    v1_copy = v1.clone().detach()

    # Normalize the vectors to get the directions and angles
    v0 = normalize_torch(v0, eps)
    v1 = normalize_torch(v1, eps)

    # Dot product with the normalized vectors
    dot = torch.sum(v0 * v1, dim=-1)
    dot = torch.minimum(dot, torch.full_like(dot, fill_value=0.999))

    return slerp_torch(dot, t, v0_copy, v1_copy).type(dtype)



def lerp_torch(t: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """
    Linearly interpolate between two vectors (optimized for torch.Tensor).

    Args:
        t (torch.Tensor): Interpolation parameter. Float tensor between 0.0 and 1.0.
        v0 (torch.Tensor): Starting vector.
        v1 (torch.Tensor): Final vector.

    Returns:
        torch.Tensor: Interpolated vector between v0 and v1.
    """
    return (1 - t).unsqueeze(-1).expand_as(v0) * v0 + t.unsqueeze(-1).expand_as(v1) * v1



def slerp_torch(dot: torch.Tensor, t: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation between two vectors (optimized for torch.Tensor).

    Args:
        dot (torch.Tensor): Dot product of the two vectors.
        t (torch.Tensor): Interpolation parameter. Float tensor between 0.0 and 1.0.
        v0 (torch.Tensor): Starting vector.
        v1 (torch.Tensor): Final vector.

    Returns:
        torch.Tensor: Interpolated vector between v0 and v1.
    """

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)

    # Finish the slerp algorithm
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = (s0.unsqueeze(-1).expand_as(v0) * v0) + (s1.unsqueeze(-1).expand_as(v1) * v1)

    return res

def normalize_torch(v: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Normalize a vector (optimized for torch.Tensor).

    Args:
        v (torch.Tensor): Input vector.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Normalized vector.
    """
    norm_v = torch.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v



def prune_labels(labels, mask):
    mask = torch.tensor(mask, dtype=torch.bool, device=labels.device)
    other_labels, last_label = labels[:, :-1], labels[:, -1:]
    other_labels = other_labels[:, ~mask]
    return torch.cat([other_labels, last_label], dim=-1)


def merge(hidden_states, mask, merge_method):

    _mask = deepcopy(mask)

    length = hidden_states.shape[-2]
    hidden_states = list(hidden_states.squeeze(0).chunk(length,dim=0))
    merged = []

    assert len(_mask) == length - 1
    _mask.insert(0, False)
    while len(_mask) > 0:

        m = _mask.pop(0)
        x = hidden_states.pop(0)

        assert m is False

        while len(_mask) > 0 and _mask[0] is True:
            _mask.pop(0)
            y = hidden_states.pop(0)

            if merge_method == 'avg':
                x = (x + y) / 2
            elif merge_method == 'add':
                x = x + y
            elif merge_method == 'slerp':
                x = slerp(0.5, x, y)
            elif merge_method == 'drop':
                ...
            elif merge_method == 'max':
                x = torch.maximum(x, y)
            elif merge_method == 'absmax':
                x = torch.where(torch.abs(x) > torch.abs(y), x, y)
            else:
                raise NotImplementedError(f"{merge_method}")
        
        merged.append(x)

    return torch.cat(merged, dim=0).unsqueeze(0)