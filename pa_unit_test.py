import random
from typing import List, Optional, Tuple, Union
import itertools
import torch
import aiter
import pytest
from aiter.test_common import checkAllclose, perftest, tensor_dump, tensor_load
from aiter import pertoken_quant
from aiter import dtypes
from enum import Enum
from einops import rearrange
import argparse

uniform_range = (-1, 1)
class PAVariant(Enum):
    Shomy = 1
    Asm = 2
    Naive = 3


def get_kv_cache_torch_dtype(
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype

def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    x = 16 // torch_dtype.itemsize
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(
            size=value_cache_shape, dtype=torch_dtype, device=device
        )
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches

def run_aiter(
    query,
    key_cache,
    value_cache,
    kv_indptr,
    kv_page_indices,
    kv_last_page_lens,
    max_seq_len,
    kv_cache_dtype,
    kv_cache_layout,
    num_kv_heads,
    scale,
    alibi_slopes,
    logits_soft_cap,
    k_scale,
    v_scale,
):
    # copied from ops.PagedAttention.forward_decode()
    _PARTITION_SIZE_ROCM = 256
    fp8_out_scale = None

    num_seqs, num_heads, head_size = query.shape
    block_size = key_cache.shape[2 if kv_cache_layout == "HND" else 1]

    output = torch.empty_like(query)
    max_num_partitions = (
        max_seq_len + _PARTITION_SIZE_ROCM - 1
    ) // _PARTITION_SIZE_ROCM
    assert _PARTITION_SIZE_ROCM % block_size == 0

    # will use single workspace buffer to accommodate following 3 intermediate tensors:
    #   1. tmp_output (shape=(num_seqs, num_heads, max_num_partitions, head_size), dtype=output.dtype)
    #   2. exp_sums (shape=(num_seqs, num_heads, max_num_partitions), dtype=float32)
    #   3. max_logits (shape=(num_seqs, num_heads, max_num_partitions), dtype=float32)

    nbyes_per_qo_elem = torch.finfo(output.dtype).bits // 8
    workspace_buffer = torch.empty(
        (num_seqs * num_heads * max_num_partitions * head_size) * nbyes_per_qo_elem
        + 2 * (num_seqs * num_heads * max_num_partitions) * 4,
        dtype=torch.uint8,
        device=output.device,
    )
    print(f"[DEBUG] num_seqs={num_seqs}, num_heads={num_heads}, "
                f"max_num_partitions={max_num_partitions}, head_size={head_size}, "
                f"nbyes_per_qo_elem={nbyes_per_qo_elem}, "
                f"_PARTITION_SIZE_ROCM={_PARTITION_SIZE_ROCM}, output.dtype={output.dtype}")

    cpa_fp8_out = False
    if fp8_out_scale is not None:
        output = torch.empty_like(output, dtype=dtypes.fp8)
        cpa_fp8_out = True
    torch.ops.aiter.paged_attention_ragged(
        output,
        workspace_buffer,
        query,
        key_cache,
        value_cache,
        scale,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        block_size,
        max_num_partitions,
        alibi_slopes,
        kv_cache_dtype,
        kv_cache_layout,
        logits_soft_cap,
        k_scale,
        v_scale,
        fp8_out_scale if cpa_fp8_out else None,
        _PARTITION_SIZE_ROCM,
    )
    if cpa_fp8_out:
        return output.view(num_seqs, num_heads * head_size)
    else:
        return output

def test_paged_attention(
    ctx_lens: int,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    kv_cache_layout: str,
    logits_soft_cap: float,
    pa_variant: PAVariant,
    quant_cache_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_default_device(device)  

    # Using default kv_scale
    k_scale = v_scale = torch.tensor([1.0], dtype=dtypes.fp32)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=dtypes.fp32)
    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    max_seq_len = ctx_lens
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = max_num_blocks_per_seq * num_seqs


    # prepare inputs & golden output
  
    query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    query.uniform_(*uniform_range)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(
        num_blocks,
        block_size,
        1,
        num_kv_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        seed,
        device,
    )

    key_cache, value_cache = key_caches[0], value_caches[0]

    # Create the block tables.
    block_tables = rearrange(
        torch.randperm(num_blocks, dtype=dtypes.i32, device=device),
        "(b nblocks) -> b nblocks",
        b=num_seqs,
    )

    # prepare flashinfer format-compatible parameters
    # TODO: pass list of context_length instead
    def convert_to_kv_indptr_last_page_lens(fixed_context_length):
        def get_num_blocks(context_length):
            return (context_length + block_size - 1) // (block_size)

        def get_last_page_len(context_length):
            return (
                context_length % block_size
                if context_length % block_size > 0
                else block_size
            )

        context_lengths = [fixed_context_length] * num_seqs
        num_blocks_list = [
            get_num_blocks(context_length) for context_length in context_lengths
        ]
        last_page_lens = [
            get_last_page_len(context_length) for context_length in context_lengths
        ]

        return torch.tensor([0] + num_blocks_list).cumsum(
            dim=0, dtype=torch.int
        ), torch.tensor(last_page_lens, dtype=torch.int)

    def convert_to_page_indices(block_tables, kv_indptr):
        elements_per_row = kv_indptr[1:] - kv_indptr[:-1]
        col_indices = torch.arange(block_tables.size(1)).expand(
            block_tables.size(0), -1
        )

        return block_tables[col_indices < elements_per_row.unsqueeze(1)]

    kv_indptr, kv_last_page_lens = convert_to_kv_indptr_last_page_lens(ctx_lens)
    kv_page_indices = convert_to_page_indices(block_tables, kv_indptr)

    # generate golden output
    key_cache_new = rearrange(key_cache, "b h d1 s d2 -> b h s (d1 d2)")
    value_cache_new = rearrange(value_cache, "b h d s -> b h s d")

    if kv_cache_layout == "NHD":
        key_cache_new = rearrange(key_cache_new, "b h s d -> b s h d")
        value_cache_new = rearrange(value_cache_new, "b h s d -> b s h d")

    # Warmup 5 iter
    for i in range(5):
        _ = run_aiter(
            query,
            key_cache_new.contiguous(),
            value_cache_new.contiguous(),
            kv_indptr,
            kv_page_indices,
            kv_last_page_lens,
            max_seq_len,
            kv_cache_dtype,
            kv_cache_layout,
            num_kv_heads,
            scale,
            alibi_slopes,
            logits_soft_cap,
            k_scale,
            v_scale,
        )

    out_golden = run_aiter(
        query,
        key_cache_new.contiguous(),
        value_cache_new.contiguous(),
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        max_seq_len,
        kv_cache_dtype,
        kv_cache_layout,
        num_kv_heads,
        scale,
        alibi_slopes,
        logits_soft_cap,
        k_scale,
        v_scale,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test Paged Attention ragged.",
    )
    parser.add_argument(
        "-c",
        "--ctx_len",
        type=int,
        default=2048,
        help="""Context length.
    e.g. -c 128""",
    )
    parser.add_argument(
        "-p",
        "--pa_variant",
        type=str,
        choices=[member.name for member in PAVariant],
        default=[PAVariant.Shomy, PAVariant.Asm],
        nargs="*",
        help="It is not used. Just place an empty str",
    )
    parser.add_argument(
        "-q",
        "--quant_cache_dtype",
        type=str,
        choices=["none", "fp8", "i8"],
        default=["none", "fp8", "i8"],
        nargs="*",
        help="""Quantization cache dtype.
        e.g. -q fp8""",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=512,
        help="number of seqs",
    )
    torch.set_printoptions(sci_mode=False)
    args = parser.parse_args()
    args.quant_cache_dtype = [
        None if i == "none" else dtypes.d_dtypes[i] for i in args.quant_cache_dtype
    ]

    ctx_len = args.ctx_len
    pa_variant = args.pa_variant
    quant_cache_dtype = args.quant_cache_dtype
    print(f"[DEBUG pa_unit_test.py] ctx_len={ctx_len}, pa_variant={pa_variant}, quant_cache_dtype={quant_cache_dtype}")

    test_paged_attention(
        ctx_len, 
        args.n,
        (6, 1),   # num_heads: query and KV
        128,      # head_size
        False,    # use_alibi
        1,        # block_size
        dtypes.bf16, # dtype
        "auto",   # kv_cache_dtype
        "HND",    # kv_cache_layout
        30.0,      # logits_soft_cap
        pa_variant,
        quant_cache_dtype,
        0,        # seed
        "cuda:0", # device
    )

'''
# Even if the input length is 256, I use "context length = 2048"
# since I would like to know the performance of the kernel when 
# the KV cache is longer than prefill 256 tokens.
python pa_unit_test.py -n 512 -c 2048

# E2E
RCCL_MSCCL_ENABLE=0 SGLANG_USE_AITER=1 SGLANG_INT4_WEIGHT=1  python -m sglang.bench_one_batch \
	--batch-size 512 --input 256 --output 2048 --tp 8 --quantization fp8 --trust-remote-code \
    --model /data/huggingface/hub/amd/grok-1-W4A8KV8  \
	--tokenizer-path /data/huggingface/hub/Xenova/grok-1-tokenizer  \
    --attention-backend aiter 


'''