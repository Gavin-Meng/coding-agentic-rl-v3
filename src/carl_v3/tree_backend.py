"""Port the V2.1 Ampere attention adapter to the locked TRL kernel seam."""

from types import SimpleNamespace


def select_sm80_backend():
    from torch.nn.attention.varlen import varlen_attn
    from trl.trainer import async_grpo_trainer as trl

    def flash_attn_varlen_func(query, key, value, *, cu_seqlens_q, cu_seqlens_k,
                              max_seqlen_q, max_seqlen_k, softmax_scale, causal,
                              window_size, softcap=None):
        if softcap:
            raise ValueError("The Ampere CUDA varlen adapter has no softcap")
        if causal and window_size == (-1, -1):
            window_size = (-1, 0)
        return varlen_attn(query, key, value, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k, scale=softmax_scale,
                           window_size=window_size, enable_gqa=query.shape[1] != key.shape[1])

    adapter = SimpleNamespace(flash_attn_varlen_func=flash_attn_varlen_func)
    trl._flash_attn = lambda: adapter
