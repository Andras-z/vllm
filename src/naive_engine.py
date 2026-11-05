"""naive_engine.py — 手写 TinyLlama 推理：加载权重 + 手写 Llama forward + KV Cache + 采样

对应 W9（阶段 D 第一步）。目标不是快，而是「把每一层都摊开看清楚」：

  - 自己按 safetensors 名字取权重，不靠 AutoModel
  - 手写 RMSNorm / RoPE / GQA attention / SwiGLU FFN（W1-W3 的零件在这里合体）
  - 手写 KV Cache（按 (layer, kv_head, seq) 追加），先不做分页
  - 手写 greedy / temperature 采样 + 逐 token 生成循环

对照验证：
  - 与 transformers 的 LlamaForCausalLM 在同一 prompt 上的 logits 比对
  - 开启/关闭 KV Cache 的 logits 一致性（验证缓存没有算错）

用法：
    python code/naive_engine.py --model <本地模型目录>
    python code/naive_engine.py --model ... --prompt "Hello" --max-new-tokens 20
    python code/naive_engine.py --model ... --verify        # 与 transformers 比对

不需要 vLLM；只需要 torch + safetensors + transformers（只作参照）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import torch.nn.functional as F


# 配置
@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 5632
    num_hidden_layers: int = 22
    num_attn_heads: int = 32
    num_key_value_heads: int = 4
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048
    tie_word_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attn_heads

    @property
    def num_kv_groups(self) -> int:
        return self.num_attn_heads // self.num_key_value_heads

    @classmethod
    def from_json(cls, json_path: str) -> "LlamaConfig":
        with open(os.path.join(json_path, "config.json"), encoding="utf-8") as f:
            data = json.load(f)
        keep = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**keep)

# 手写算子
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm：只除均方根，不减均值。y = x / sqrt(mean(x^2) + eps) * w"""
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight

def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """预计算 cos/sin 表：shape [seq_len, head_dim/2]"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arrange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [seq_len, head_dim/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)  # [seq_len, head_dim/2], [seq_len, head_dim/2]

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D]；cos/sin: [T, D/2]（已按当前位置切好）

    Llama 用「相邻两维配对」（rotate_half 风格）：把 D 拆成前后两半再交叉。
    """
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    c, s = cos[None, None, :, :], sin[None, None, :, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)  # [B, H, T, D]


# 权重加载
def load_weights(model_dir: str) -> dict[str, torch.Tensor]:
    """按 safetensors 名字加载权重，返回 {name: tensor}"""
    from safetensors.torch import load_file

    path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(path):
        # 分片模型：合并所有 shard
        import glob

        merged = {}
        for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            merged.update(load_file(f))
        return merged
    return load_file(path)

# KV Cache（朴素版：预分配 [L, H_kv, max_seq, D]）
class NaiveKVCache:
    def __init__(self, cfg: LlamaConfig, max_seq: int, device, dtype):
        self.k = torch.zeros(
            cfg.num_hidden_layers, cfg.num_key_value_heads, max_seq, cfg.head_dim,
            device=device, dtype=dtype,
        )
        self.v = torch.zeros_like(self.k)
        self.max_seq = max_seq
        self.seq_len = 0

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """k/v: [B=1, H_kv, T_new, D] → 写入并返回 [1, H_kv, seq_len+T_new, D]"""
        t_new = k.shape[2]
        end = self.seq_len + t_new
        if end > self.max_seq:
            raise RuntimeError(f"序列超过 max_seq={self.max_seq}（本 Demo 不做淘汰）")
        self.k[layer, :, self.seq_len:end] = k[0]
        self.v[layer, :, self.seq_len:end] = v[0]
        return (
            self.k[layer, :, :end].unsqueeze(0),  # [1, H_kv, seq_len+T_new, D] 
            self.v[layer, :, :end].unsqueeze(0)   # [1, H_kv, seq_len+T_new, D]
        )

    def advance(self, t_new: int):
        """更新 seq_len"""
        self.seq_len += t_new

    def reset(self):
        """清空缓存"""
        self.seq_len = 0

# 手写 Llama forward
class NaiveLlama:
    def __init__(self, cfg: LlamaConfig, weights: dict[str, torch.Tensor], device="cuda", dtype=torch.float16):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.weights = {k: v.to(device=device, dtype=dtype) for k, v in weights.items()}

        # 只保留需要的名字，避免整份 dict 一直驻留
        need = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        for i in range(cfg.num_hidden_layers):
            p = f"model.layers.{i}."
            need |= {
                p + "self_attn.q_proj.weight",
                p + "self_attn.k_proj.weight",
                p + "self_attn.v_proj.weight",
                p + "self_attn.o_proj.weight",
                p + "mlp.gate_proj.weight",
                p + "mlp.down_proj.weight",
                p + "mlp.up_proj.weight",
                p + "input_layernorm.weight",
                p + "post_attention_layernorm.weight",
            }
        self.weights = {k: v for k, v in self.weights.items() if k in need}
        missing = need - set(self.weights)
        if missing:
            raise KeyError(f"缺少权重：{sorted(missing)[:5]} ...")

        self.cos, self.sin = build_rope_cache(
            cfg.max_position_embeddings, cfg.head_dim, cfg.rope_theta, device, dtype
        )

    # ---- 单层 ----
    def _attn(self, x, layer, cache: NaiveKVCache | None, start_pos: int):
        """单层 attention + KV Cache"""
        cfg = self.cfg
        p = f"model.layers.{layer}.self_attn."
        B, T, _ = x.shape
        H, Hkv, D = cfg.num_attn_heads, cfg.num_key_value_heads, cfg.head_dim

        # Q/K/V
        q = F.linear(x, self.weights[p + "q_proj.weight"]).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        k = F.linear(x, self.weights[p + "k_proj.weight"]).view(B, T, Hkv, D).transpose(1, 2)  # [B, Hkv, T, D]
        v = F.linear(x, self.weights[p + "v_proj.weight"]).view(B, T, Hkv, D).transpose(1, 2)  # [B, Hkv, T, D]

        cos = self.cos[start_pos : start_pos + T]
        sin = self.sin[start_pos : start_pos + T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k_all, v_all = cache.append(layer, k, v)  # [1, Hkv, seq_len+T, D]
        else:
            k_all, v_all = k, v

        # GQA：把 kv 头复制 groups 份以匹配 q 头
        g = cfg.num_kv_groups
        k_rep = k_all.repeat_interleave(g, dim=1)  # [1, H, seq_len+T, D]
        v_rep = v_all.repeat_interleave(g, dim=1)  # [1, H, seq_len+T, D]

        # 因果掩码：只在 T>1（prefill / 多 token）时需要
        Tk = k_rep.shape[2]
        attn = (q @ k_rep.transpose(-1, -2)) / math.sqrt(D)  # [B, H, T, Tk]
        if T > 1:
            # query 的位置是 start_pos..start_pos+T-1，key 是 0..Tk-1
            qpos = torch.arange(start_pos, start_pos + T, device=x.device)[:, None]
            kpos = torch.arange(Tk, device=x.device)[None, :]
            attn = attn.masked_fill(kpos > qpos, float("-inf"))
        probs = torch.softmax(attn.float(), dim=-1).to(x.dtype)  # [B, H, T, Tk]
        out = probs @ v_rep  # [B, H, T, D]
        out = out.transpose(1, 2).reshape(B, T, H * D)  # [B, T, H*D]
        return F.linear(out, self.weights[p + "o_proj.weight"])  # [B, T, hidden]

    def _mlp(self, x, layer):
        """单层 MLP"""
        p = f"model.layers.{layer}.mlp."
        gate = F.linear(x, self.weights[p + "gate_proj.weight"])
        up = F.linear(x, self.weights[p + "up_proj.weight"])
        down = F.linear(F.silu(gate) * up, self.weights[p + "down_proj.weight"])
        return down

    # 整模型
    def forward(self, input_ids: torch.Tensor, cache: NaiveKVCache | None = None,
                start_pos: int = 0, logits_last_only: bool = True) -> torch.Tensor:
        """input_ids: [B, T] → logits: [B, T, vocab]"""
        cfg = self.cfg
        x = F.embedding(input_ids, self.weights["model.embed_tokens.weight"])  # [B, T, hidden]

        for i in range(cfg.num_hidden_layers):
            p = f"model.layers.{i}."
            h = rms_norm(x, self.weights[p + "input_layernorm.weight"], cfg.rms_norm_eps)
            x = x + self._attn(h, i, cache, start_pos)
            h = rms_norm(x, self.weights[p + "post_attention_layernorm.weight"], cfg.rms_norm_eps)
            x = x + self._mlp(h, i)

        x = rms_norm(x, self.weights["model.norm.weight"], cfg.rms_norm_eps)
        if logits_last_only:
            x = x[:, -1:]  # [B, 1, hidden]
        return F.linear(x, self.weights["lm_head.weight"]).float()

# 采样
def sample(logits: torch.Tensor, temperature: float = 0.0, top_k: int = 0) -> int:
    """logits: [vocab] → 返回采样的 token id"""
    if temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1).squeeze(0)  # [vocab]
    return int(torch.multinomial(probs, num_samples=1).item())

# 生成循环（prefill + decode，带 KV Cache）
@torch.no_grad()
def generate(
    model: NaiveLlama, promt_ids: list[int], max_new_tokens: int = 20,
    temperature: float = 0.0, top_k: int = 0, verbose: bool = True):
    cfg = model.cfg
    device = model.device
    cache = NaiveKVCache(cfg, cfg.max_position_embeddings, device, model.dtype)

    ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)  # [1, T]
    t0 = time.perf_counter()

    logits = model.forward(ids, cache=cache, start_pos=0)  # prefill
    cache.advance(ids.shape[1])
    t_prefill = time.perf_counter() - t0

    out_ids = []
    t_decode0 = time.perf_counter()
    for step in range(max_new_tokens):
        nxt = sample(logits, temperature=temperature, top_k=top_k)
        out_ids.append(nxt)
        if verbose:
            print(f"  step {step+1:>2}: token={nxt}", flush=True)
        if step == max_new_tokens - 1:
            break
        cur = torch.tensor([[nxt]], device=device, dtype=torch.long)  # [1, 1]
        logits = model.forward(cur, cache=cache, start_pos=cache.seq_len)  # decode
        cache.advance(1)
    t_decode = time.perf_counter() - t_decode0

    n_dec = max(len(out_ids) - 1, 1)
    if verbose:
        print(f"\n  prefill {len(prompt_ids)} tok: {t_prefill*1e3:.1f} ms")
        print(f"  decode  {len(out_ids)} tok: {t_decode*1e3:.1f} ms "
              f"({t_decode/n_dec*1e3:.1f} ms/token)")
    return out_ids

# 验证：与 transformers 比对
@torch.no_grad()
def verify(model: NaiveLlama, model_dir: str, prompt_ids: list[int]):
    from transformers import AutoModelForCausalLM

    print("\n[验证] 与 transformers 比对 logits / 贪心生成")
    hf = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", torch_dtype=torch.float16).eval()

    ids = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)

    ours = model.forward(ids, cache=None, start_pos=0)  # [B, T, vocab]
    theirs = hf(ids).logits[:, -1:, :].float()  # [B, 1, vocab]

    err = (ours - theirs).abs()
    print(f"  logits: max_err={err.max().item():.4f}  mean_err={err.mean().item():.6f}")
    print(f"  argmax: ours={ours.argmax(-1).item()}  hf={theirs.argmax(-1).item()}")

    # 带 KV Cache 的分步 vs 一次性（正确做法：前 4 个走 prefill，第 5 个增量喂）
    ours_full = model.forward(ids, cache=None, start_pos=0)  # [B, T, vocab]

    cache = NaiveKVCache(model.cfg, model.cfg.max_position_embeddings, model.device, model.dtype)
    model.forward(ids[:, :-1], cache=cache, start_pos=0) # prefill
    cache.advance(ids.shape[1] - 1)
    lg2 = model.forward(ids[:, -1:], cache=cache, start_pos=ids.shape[1] - 1)  # decode

    d = (ours_full - lg2).abs()
    print(f"  KV Cache 一致性: max_err={d.max().item():.6f}  "
          f"argmax ours={ours_full.argmax(-1).item()} kv={lg2.argmax(-1).item()}")

    del hf
    torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="本地模型目录（含 config.json / *.safetensors）")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    cfg = LlamaConfig.from_json(args.model)
    print(f"配置: L={cfg.num_hidden_layers} H={cfg.num_attention_heads} "
          f"Hkv={cfg.num_key_value_heads} D={cfg.head_dim} V={cfg.vocab_size}")

    t0 = time.perf_counter()
    weights = load_weights(args.model)
    print(f"权重: {len(weights)} 个张量，加载 {time.perf_counter()-t0:.1f}s")

    model = NaiveLlama(cfg, weights, device=args.device, dtype=dtype)
    n_param = sum(v.numel() for v in model.weights.values())
    print(f"模型: {n_param/1e9:.2f}B 参数（{n_param*2/2**30:.2f} GiB @ fp16）")

    # tokenizer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = tok(args.prompt, return_tensors=None)["input_ids"]
    print(f"\nprompt: {args.prompt!r} → {len(prompt_ids)} tokens {prompt_ids[:12]}")

    out = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_k)
    print(f"\n输出: {tok.decode(out)!r}")

    if args.verify:
        verify(model, args.model, prompt_ids)


if __name__ == "__main__":
    main()