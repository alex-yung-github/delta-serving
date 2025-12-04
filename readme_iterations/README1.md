import numpy as np
import torch
import zstd
from typing import Dict, List, Tuple
from numpy.linalg import svd


# ============================================================
# Utility: Convert fp32 tensor → uint8 view and back
# ============================================================

def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """View a float32 tensor as uint8 (little-endian)."""
    arr = t.detach().cpu().numpy().astype(np.float32)
    return arr.view(np.uint8)


# ============================================================
# 1. SimHash Projection Initialization
# ============================================================

def init_simhash_proj(bits: int, max_block_size: int, seed: int = 42) -> np.ndarray:
    """
    Create a random projection matrix used for simhash.
    Shape: (max_block_size, bits)
    """
    rng = np.random.RandomState(seed)
    return rng.normal(size=(max_block_size, bits)).astype(np.float64)


# ============================================================
# 2. Compute SimHash Over a List of Flattened Blocks
# ============================================================

def simhash_blockset(blocks: List[np.ndarray], proj: np.ndarray, bits: int) -> np.ndarray:
    """
    Compute SimHash for each block (flatten → projected → sign bits).
    """
    hashes = []
    max_len = proj.shape[0]

    for B in blocks:
        flat = B.flatten().astype(np.float64)
        L = min(len(flat), max_len)
        subproj = proj[:L, :]
        proj_vals = flat[:L] @ subproj
        bits_arr = ['1' if v > 0 else '0' for v in proj_vals]
        hashes.append(int("".join(bits_arr), 2))

    return np.array(hashes, dtype=object)


# ============================================================
# 3. Build Low-Rank Proxies for Each Block
# ============================================================

def build_block_proxies(pkg: Dict,
                        base_sd: Dict,
                        block_size: Tuple[int,int]=(16,16),
                        proxy_rank: int = 4) -> Dict:
    """
    Reconstruct weight matrix from base_sd XOR pkg['deltas'],
    split it into blocks, and compute SVD low-rank blocks.

    Output structure:
    proxies[name] = {
        'coords': [(i,j), ...],
        'U_list': [...],
        'V_list': [...],
        'shape': (R, C),
        'block_size': [br, bc]
    }
    """
    proxies = {}
    dctx = zstd.ZstdDecompressor()

    br, bc = block_size

    for name, meta in pkg['meta'].items():
        shape = tuple(meta['shape'])
        R, C = shape

        # ---- 1. Decompress delta ----
        raw = dctx.decompress(pkg['deltas'][name])
        delta_u8 = np.frombuffer(raw, dtype=np.uint8)
        base_u8  = tensor_to_uint8(base_sd[name])
        ft_u8    = np.bitwise_xor(base_u8, delta_u8)
        full_param = ft_u8.view(np.float32).reshape(shape)

        # ---- 2. Split into blocks ----
        blocks = []
        coords = []

        for i in range(0, R, br):
            for j in range(0, C, bc):
                sub = full_param[i:i+br, j:j+bc]
                # Fix 1D or malformed shapes
                if sub.ndim != 2:
                    sub = sub.reshape(1, -1)
                blocks.append(sub.astype(np.float32))
                coords.append((i, j))

        # ---- 3. Compute SVD Proxies ----
        U_list, V_list = [], []

        for B in blocks:
            r_dim, c_dim = B.shape

            try:
                U, s, Vh = svd(B, full_matrices=False)
                r = min(proxy_rank, len(s))
                Ur = (U[:, :r] * s[:r])
                Vr = Vh[:r, :]
            except:
                # fallback rank-1
                col_mean = B.mean(axis=1, keepdims=True)
                row_mean = B.mean(axis=0, keepdims=True)
                Ur = col_mean
                Vr = row_mean

            U_list.append(Ur.astype(np.float32))
            V_list.append(Vr.astype(np.float32))

        proxies[name] = {
            'coords': coords,
            'shape': shape,
            'U_list': U_list,
            'V_list': V_list,
            'block_size': [br, bc],
        }

    return proxies


# ============================================================
# 4. Calibration Scalars (Proxy → True Output Fit)
# ============================================================

def calibrate_block_scalars(proxies: Dict,
                            base_sd: Dict,
                            pkg: Dict,
                            sample_act: np.ndarray,
                            bits: int,
                            proj: np.ndarray,
                            samples_per_block: int = 10):
    """
    Fit alpha for each block so:
        true = alpha * proxy
    """
    scalars = {}
    dctx = zstd.ZstdDecompressor()

    for name, info in proxies.items():
        coords = info['coords']
        Ulist = info['U_list']
        Vlist = info['V_list']
        br, bc = info['block_size']

        # reconstruct full param for true GEMMs
        shape = info['shape']
        R, C = shape

        raw = dctx.decompress(pkg['deltas'][name])
        delta_u8 = np.frombuffer(raw, dtype=np.uint8)
        base_u8  = tensor_to_uint8(base_sd[name])
        ft_u8    = np.bitwise_xor(base_u8, delta_u8)
        full_param = ft_u8.view(np.float32).reshape(shape)

        seq_len = sample_act.shape[0]
        pick = np.random.choice(seq_len, min(samples_per_block, seq_len), replace=False)

        alphas = []

        for (i, j), U, V in zip(coords, Ulist, Vlist):
            # pick activation block
            bc_eff = V.shape[1]
            act_blk = sample_act[pick, j:j+bc_eff]

            # proxy
            mid = act_blk @ V.T
            proxy_out = mid @ U.T

            # true
            B = full_param[i:i+U.shape[0], j:j+bc_eff]
            true_out = act_blk @ B.T

            p = proxy_out.ravel()
            t = true_out.ravel()
            denom = p @ p

            if denom < 1e-8:
                alphas.append(0.0)
            else:
                alphas.append(float((p @ t) / denom))

        scalars[name] = np.array(alphas, dtype=np.float32)

    return scalars


# ============================================================
# 5. Approximate Hashed-GEMM Forward
# ============================================================

def approx_gemm_hashed(x: torch.Tensor,
                       pkg: Dict,
                       proxies: Dict,
                       scalars: Dict,
                       proj: np.ndarray,
                       bits: int):

    assert x.ndim == 3 and x.shape[0] == 1
    act = x[0].detach().cpu().numpy().astype(np.float32)
    seq_len, hidden = act.shape

    y = np.zeros((seq_len, hidden), dtype=np.float32)

    for name, info in proxies.items():
        coords = info['coords']
        Ulist  = info['U_list']
        Vlist  = info['V_list']
        alphas = scalars[name]

        for (i, j), U, V, a in zip(coords, Ulist, Vlist, alphas):
            bc_eff = V.shape[1]

            # activation block slice
            act_blk = act[:, j:j+bc_eff]
            if act_blk.shape[1] == 0:
                continue

            # proxy
            mid = act_blk @ V.T
            out = (mid @ U.T) * float(a)

            # write back
            br_eff = U.shape[0]
            end = min(hidden, i + br_eff)
            cols = end - i
            y[:, i:end] += out[:, :cols]

    return y.reshape(1, -1)


# ============================================================
# 6. One-time Preparation
# ============================================================

def prepare_hashed_gemm(pkg: Dict,
                        base_sd: Dict,
                        bits: int = 128,
                        block_size: Tuple[int,int]=(16,16),
                        proxy_rank: int = 4,
                        calib_samples: int = 16):

    max_len = block_size[0] * block_size[1]
    proj = init_simhash_proj(bits, max_len)

    proxies = build_block_proxies(pkg, base_sd, block_size, proxy_rank)

    # random calibration activations
    hidden = list(proxies.values())[0]['shape'][1]
    sample_act = np.random.randn(calib_samples, hidden).astype(np.float32)

    scalars = calibrate_block_scalars(
        proxies, base_sd, pkg, sample_act,
        bits=bits, proj=proj,
        samples_per_block=min(8, calib_samples)
    )

    return {
        'proj': proj,
        'proxies': proxies,
        'scalars': scalars,
        'info': {'bits': bits, 'block_size': block_size}
    }


# ============================================================
# 7. Wrapper for Approx Forward
# ============================================================

def approx_forward_wrapper(pkg, base_sd, x, prep_cache):
    return approx_gemm_hashed(
        x,
        pkg=pkg,
        proxies=prep_cache['proxies'],
        scalars=prep_cache['scalars'],
        proj=prep_cache['proj'],
        bits=prep_cache['info']['bits']
    )
