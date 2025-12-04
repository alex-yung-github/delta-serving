# HBHE-ZSTD + Hash-Proxy Framework (Improved Accuracy + Speed)
import numpy as np
import torch
import torch.nn as nn
import zstandard as zstd
from typing import List, Tuple, Dict
from copy import deepcopy
from time import perf_counter

np.random.seed(42)
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def simhash(data: np.ndarray, bits: int = 128) -> int:
    flattened = data.flatten().astype(np.float64)
    rand_vecs = np.random.randn(len(flattened), bits)
    proj = flattened @ rand_vecs
    return int("".join("1" if p > 0 else "0" for p in proj), 2)

def divide_into_blocks_with_coords(tensor: torch.Tensor, block_size: Tuple[int, int] = (16, 16)):
    """
    Returns list of (block_numpy, row_start, col_start) for 2D or 1D->2D tensors.
    """
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError("Only 2D or 1D tensors supported")
    rows, cols = tensor.shape
    br, bc = block_size
    blocks = []
    for i in range(0, rows, br):
        for j in range(0, cols, bc):
            block = tensor[i:i+br, j:j+bc].cpu().numpy().astype(np.float32)
            blocks.append((block, i, j))
    return blocks

class TinyTransformerLayer(nn.Module):
    def __init__(self, d_model=64, nhead=4, dim_feedforward=128):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        a, _ = self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + a
        y = self.ff(self.ln2(x))
        return x + y

base_layer = TinyTransformerLayer().to(device).eval()
print("Params:", sum(p.numel() for p in base_layer.parameters()))

def simulate_finetune(module: nn.Module, frac: float = 0.03, mag: float = 0.01) -> nn.Module:
    finetuned = deepcopy(module)
    for p in finetuned.parameters():
        if p.dtype.is_floating_point:
            mask = torch.rand_like(p) < frac
            noise = mag * torch.randn_like(p)
            p.data.add_(mask.float() * noise)
    return finetuned.eval()

finetuned_layer = simulate_finetune(base_layer)
changed = sum((base_p != ft_p).sum().item()
              for base_p, ft_p in zip(base_layer.parameters(), finetuned_layer.parameters())
              if base_p.dtype.is_floating_point)
total = sum(p.numel() for p in base_layer.parameters() if p.dtype.is_floating_point)
print(f"Changed: {changed}/{total} ({100*changed/total:.2f}%)")

def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().ravel().view(np.uint8)

def compress_with_hash(base_sd: Dict, ft_sd: Dict, level: int = 10, bits: int = 128, block_size: Tuple[int,int]=(16,16)) -> Dict:
    compressor = zstd.ZstdCompressor(level=level)
    pkg = {'meta': {}, 'deltas': {}, 'hashes': {}, 'stats': {'per_param': {}}, 'block_centroids': {}}
    total_ratio = 0

    for name, base_t in base_sd.items():
        if not base_t.dtype.is_floating_point:
            continue
        ft_t = ft_sd[name]
        delta_u8 = np.bitwise_xor(tensor_to_uint8(base_t), tensor_to_uint8(ft_t))
        compressed = compressor.compress(delta_u8.tobytes())
        orig_bytes = base_t.numel() * base_t.element_size()
        ratio = len(compressed) / orig_bytes
        pkg['stats']['per_param'][name] = ratio
        total_ratio += ratio

        # Reconstruct float tensor for hashing and block-centroids
        delta_float = torch.from_numpy(delta_u8.view(np.float32).reshape(base_t.shape))
        blocks_with_coords = divide_into_blocks_with_coords(delta_float, block_size=block_size)
        hashes = [simhash(b[0], bits) for b in blocks_with_coords]

        # store block centroids (the actual block matrices) to enable better approx GEMM
        centroids = [b[0] for b in blocks_with_coords]  # each is br x bc numpy array
        coords = [(b[1], b[2]) for b in blocks_with_coords]

        pkg['meta'][name] = {
            'shape': list(base_t.shape),
            'dtype': str(base_t.dtype),
            'num_blocks': len(blocks_with_coords),
            'block_coords': coords,
            'block_size': list(block_size)
        }
        pkg['deltas'][name] = compressed
        pkg['hashes'][name] = hashes
        pkg['block_centroids'][name] = centroids

    pkg['stats']['avg_ratio'] = total_ratio / len(pkg['deltas']) if len(pkg['deltas'])>0 else 0.0
    return pkg

base_sd = base_layer.state_dict()
ft_sd = finetuned_layer.state_dict()
pkg = compress_with_hash(base_sd, ft_sd, block_size=(128,128))  # smaller blocks help accuracy
print("Avg Compression Ratio:", round(pkg['stats']['avg_ratio'], 4))

def reconstruct_exact(base_sd: Dict, pkg: Dict) -> Dict:
    dctx = zstd.ZstdDecompressor()
    rec_sd = {}
    for name, meta in pkg['meta'].items():
        raw = dctx.decompress(pkg['deltas'][name])
        delta_u8 = np.frombuffer(raw, np.uint8)
        base_u8 = tensor_to_uint8(base_sd[name])
        ft_u8 = np.bitwise_xor(base_u8, delta_u8)
        dtype = np.float32
        ft_tensor = torch.from_numpy(ft_u8.view(dtype).reshape(meta['shape']))
        rec_sd[name] = ft_tensor
    # For parameters that weren't changed (or not included), copy base
    for name, base_t in base_sd.items():
        if name not in rec_sd and base_t.dtype.is_floating_point:
            rec_sd[name] = base_t.clone()
    return rec_sd

def approx_gemm_hash_for_param(act: np.ndarray, name: str, pkg: Dict) -> np.ndarray:
    """
    Approximate multiplication of parameter matrix (shape r x c) with act (sequence x c)
    using stored block centroids:
      - For each block at coords (i,j) with block matrix B (br x bc),
        compute contribution: out_rows[i:i+br] += B @ act_block.T  (for each token)
    act: (seq_len, c)
    returns out: (seq_len, r)
    """
    meta = pkg['meta'][name]
    centroids = pkg['block_centroids'][name]
    coords = meta['block_coords']
    r, c = meta['shape']
    seq_len = act.shape[0]
    out = np.zeros((seq_len, r), dtype=np.float32)

    # vectorized-ish: iterate blocks but do matrix multiply for token batch at once
    for (B, (i, j)) in zip(centroids, coords):
        # shape of B: br x bc
        br = B.shape[0]
        bc = B.shape[1]
        # act_block: (seq_len, bc)
        act_block = act[:, j:j+bc]
        if act_block.shape[1] != bc:
            # pad/truncate if at edge
            padded = np.zeros((seq_len, bc), dtype=np.float32)
            padded[:, :act_block.shape[1]] = act_block
            act_block = padded
        # compute (seq_len, br) = act_block (seq_len x bc) @ B.T (bc x br)
        contrib = act_block @ B.T  # (seq_len, br)
        out[:, i:i+br] += contrib
    return out  # (seq_len, r)

# === Calibration ===
cal_x = torch.randn(1, 4, 64, device=device)
rec_sd = reconstruct_exact(base_sd, pkg)
cal_layer = TinyTransformerLayer().to(device)
cal_layer.load_state_dict(rec_sd)
cal_layer.eval()
with torch.no_grad():
    exact_out = cal_layer(cal_x)
cal_y_mean = exact_out.mean().item()

# Build approximate sum baseline using stored centroids
# Use numpy act representation for calibration
act_np = cal_x[0].detach().cpu().numpy().astype(np.float32)  # (seq_len, hidden)
approx_accum = np.zeros((act_np.shape[0], act_np.shape[1]), dtype=np.float32)
for name in ['self_attn.in_proj_weight', 'self_attn.out_proj.weight', 'ff.0.weight', 'ff.2.weight']:
    if name in pkg['meta']:
        part = approx_gemm_hash_for_param(act_np, name, pkg)  # (seq_len, rows)
        # if rows != hidden, try to reduce or match shape by averaging or slicing
        r, _ = pkg['meta'][name]['shape']
        if r == act_np.shape[1]:
            approx_accum += part
        elif r > act_np.shape[1]:
            approx_accum += part[:, :act_np.shape[1]]
        else:
            # r < hidden, tile/accumulate into first r dims
            approx_accum[:, :r] += part

approx_mean = approx_accum.mean()
scale_factor = cal_y_mean / (approx_mean + 1e-12)

def e2e_forward(pkg: Dict, base_layer: nn.Module, x: torch.Tensor, approx: bool = False, rec_layer_cached: nn.Module = None) -> Tuple[float, np.ndarray]:
    t0 = perf_counter()
    if not approx:
        # use cached reconstructed layer if provided (avoid repeated decompress + deepcopy)
        if rec_layer_cached is None:
            rec_sd_local = reconstruct_exact(base_sd, pkg)
            layer_local = deepcopy(base_layer).to(device)
            layer_local.load_state_dict(rec_sd_local)
            layer_local.eval()
        else:
            layer_local = rec_layer_cached
        with torch.no_grad():
            out_t = layer_local(x).detach().cpu().numpy()  # (1, seq, hidden)
        out = out_t.reshape(x.shape[0], -1)  # flatten per example
    else:
        # Approximate using stored centroids; produce (1, seq, hidden)
        act_np = x[0].detach().cpu().numpy().astype(np.float32)  # (seq_len, hidden)
        seq_len = act_np.shape[0]
        hidden = act_np.shape[1]
        y_approx = np.zeros((seq_len, hidden), dtype=np.float32)

        for name in ['self_attn.in_proj_weight', 'self_attn.out_proj.weight', 'ff.0.weight', 'ff.2.weight']:
            if name in pkg['meta']:
                part = approx_gemm_hash_for_param(act_np, name, pkg)  # (seq_len, rows)
                r, _ = pkg['meta'][name]['shape']
                # Place/align part into y_approx: try to align first dims
                if r == hidden:
                    y_approx += part
                elif r > hidden:
                    y_approx += part[:, :hidden]
                else:
                    y_approx[:, :r] += part

        y_approx = y_approx * scale_factor
        out = y_approx.reshape(1, -1)
    return perf_counter() - t0, out

def benchmark_e2e(pkg: Dict, base_layer: nn.Module, x: torch.Tensor, repeats: int = 20):
    # Reconstruct exact layer once and cache it for repeated exact calls (major speedup)
    rec_sd_cached = reconstruct_exact(base_sd, pkg)
    rec_layer = deepcopy(base_layer).to(device)
    rec_layer.load_state_dict(rec_sd_cached)
    rec_layer.eval()

    exact_times, approx_times = [], []
    exact_ys, approx_ys = [], []

    for _ in range(repeats):
        t, y = e2e_forward(pkg, base_layer, x, approx=False, rec_layer_cached=rec_layer)
        exact_times.append(t)
        exact_ys.append(y.flatten())

        t, y = e2e_forward(pkg, base_layer, x, approx=True, rec_layer_cached=rec_layer)
        approx_times.append(t)
        approx_ys.append(y.flatten())

    exact_y = np.stack(exact_ys)  # (repeats, seq*hidden)
    approx_y = np.stack(approx_ys)
    nmse = np.mean((exact_y - approx_y)**2) / (np.var(exact_y) + 1e-12)

    stats = {
        'exact_time': float(np.mean(exact_times)),
        'approx_time': float(np.mean(approx_times)),
        'speedup': float(np.mean(exact_times) / (np.mean(approx_times) + 1e-12)),
        'nmse': float(nmse)
    }
    return stats

# Run benchmark
x = torch.randn(1, 4, 64, device=device)
stats = benchmark_e2e(pkg, base_layer, x, repeats=30)
print("\nBenchmark Results:")
print(f"Exact time : {stats['exact_time']*1000:.2f} ms")
print(f"Approx time: {stats['approx_time']*1000:.2f} ms")
print(f"Speedup    : {stats['speedup']:.1f}x")
print(f"NMSE       : {stats['nmse']:.3f}")
