import math
from dataclasses import dataclass
from typing import Tuple
import torch
import triton
import triton.language as tl


def _sigma_rho_to_cov_elements(sigx, sigy, rho):
    sxx = sigx * sigx
    syy = sigy * sigy
    sxy = rho * sigx * sigy
    return sxx, sxy, syy

def _max_sigma_principal_radius(sigx, sigy, rho):
    sxx, sxy, syy = _sigma_rho_to_cov_elements(sigx, sigy, rho)
    tr = sxx + syy
    disc = torch.clamp((sxx - syy)**2 + 4.0 * (sxy**2), min=0.0)
    lam_max = 0.5 * (tr + torch.sqrt(disc))
    return torch.sqrt(torch.clamp(lam_max, min=0.0))

@dataclass
class RenderConfig:
    tile_size: int = 16
    max_gauss_chunk: int = 32
    min_sigma_px: float = 0.1
    clamp_rho: float = 0.999


def _build_tile_gauss_csr_batched(
    cx_local, cy_local, sigx, sigy, rho,
    H, W, tile_size,
    b_of_gauss, B,
    pad_sigma=3.0
):
    device = cx_local.device

    tiles_x_per = (W + tile_size - 1) // tile_size
    tiles_y = (H + tile_size - 1) // tile_size
    tiles_x_total = tiles_x_per * B 
    T_total = tiles_x_total * tiles_y

    r = pad_sigma * _max_sigma_principal_radius(sigx, sigy, rho)  # [G]

    x0 = torch.floor((cx_local - r) / tile_size).to(torch.int64).clamp(0, tiles_x_per - 1)
    x1 = torch.floor((cx_local + r) / tile_size).to(torch.int64).clamp(0, tiles_x_per - 1)
    y0 = torch.floor((cy_local - r) / tile_size).to(torch.int64).clamp(0, tiles_y - 1)
    y1 = torch.floor((cy_local + r) / tile_size).to(torch.int64).clamp(0, tiles_y - 1)

    span_x = (x1 - x0 + 1).to(torch.int64)
    span_y = (y1 - y0 + 1).to(torch.int64)
    spans  = span_x * span_y 
    total_pairs = spans.sum()
    if total_pairs == 0:
        starts = torch.zeros(T_total + 1, dtype=torch.int32, device=device)
        indices = torch.empty(0, dtype=torch.int32, device=device)
        return starts, indices, (tiles_y, tiles_x_total)

    offsets = torch.zeros_like(spans)
    torch.cumsum(spans[:-1], dim=0, out=offsets[1:])
    positions = torch.arange(total_pairs, device=device, dtype=torch.int64)

    gauss_id = torch.searchsorted(offsets, positions, right=True) - 1
    local = positions - offsets[gauss_id]
    dx = local % span_x[gauss_id]
    dy = local // span_x[gauss_id]

    xi_local = (x0[gauss_id] + dx)                    # [0, tiles_x_per)
    yi       = (y0[gauss_id] + dy)                    # [0, tiles_y)
    b_sel    = b_of_gauss[gauss_id] 

    xi_total = xi_local + b_sel * tiles_x_per         # [0, tiles_x_total)
    tile_id_total = (yi * tiles_x_total + xi_total).to(torch.int64)

    order = torch.argsort(tile_id_total)
    tile_id_sorted = tile_id_total[order]
    indices_sorted = gauss_id[order].to(torch.int32)

    counts = torch.bincount(tile_id_sorted, minlength=T_total).to(torch.int32)
    starts = torch.empty(T_total + 1, dtype=torch.int32, device=device)
    torch.cumsum(counts, dim=0, out=starts[1:])
    starts[0] = 0
    return starts, indices_sorted, (tiles_y, tiles_x_total)


@triton.jit
def _render_tiles_forward(
    img_ptr, H, Wc,
    tile_size, tiles_x, tiles_y,
    tile_starts_ptr, tile_indices_ptr,
    cx_ptr, cy_ptr, sx_ptr, sy_ptr, rho_ptr, color_ptr,
    min_sigma, det_eps,
    BLOCK_PIX: tl.constexpr, 
    GAUSS_CHUNK: tl.constexpr,
    BLOCK_FEAT: tl.constexpr 
):
    pid_tile  = tl.program_id(0)
    pid_chunk = tl.program_id(1)

    tx = pid_tile % tiles_x
    ty = pid_tile // tiles_x
    x0 = tx * tile_size
    y0 = ty * tile_size

    xs = x0 + tl.arange(0, BLOCK_PIX)
    xs_mask = (xs < Wc)
    xs = tl.minimum(xs, Wc - 1)

    start = tl.load(tile_starts_ptr + pid_tile)
    end   = tl.load(tile_starts_ptr + pid_tile + 1)

    base  = start + pid_chunk * GAUSS_CHUNK
    g_idx = base + tl.arange(0, GAUSS_CHUNK)
    g_mask = g_idx < end
    gids = tl.load(tile_indices_ptr + g_idx, mask=g_mask, other=0)

    cx  = tl.load(cx_ptr  + gids, mask=g_mask, other=0.0)
    cy  = tl.load(cy_ptr  + gids, mask=g_mask, other=0.0)
    sx  = tl.load(sx_ptr  + gids, mask=g_mask, other=0.0)
    sy  = tl.load(sy_ptr  + gids, mask=g_mask, other=0.0)
    rho = tl.load(rho_ptr + gids, mask=g_mask, other=0.0)

    f_offs = tl.arange(0, BLOCK_FEAT)[None, :]
    gids_exp = gids[:, None]
    col_offs = gids_exp * BLOCK_FEAT + f_offs
    col = tl.load(color_ptr + col_offs, mask=g_mask[:, None], other=0.0)

    sx  = tl.maximum(sx, min_sigma)
    sy  = tl.maximum(sy, min_sigma)
    rho = tl.maximum(tl.minimum(rho, 0.999), -0.999)

    sxx = sx * sx
    syy = sy * sy
    sxy = rho * sx * sy
    det = tl.maximum(sxx * syy - sxy * sxy, det_eps)
    inv00 =  syy / det
    inv01 = -sxy / det
    inv11 =  sxx / det

    m_row = g_mask.to(tl.float32)[:, None, None] # [G, 1, 1]
    col   = col[:, None, :]                      # [G, 1, F]
    cx    = cx[:, None]
    cy    = cy[:, None]
    inv00 = inv00[:, None]
    inv01 = inv01[:, None]
    inv11 = inv11[:, None]

    for ry in range(0, BLOCK_PIX):
        iy = tl.minimum(y0 + ry, H - 1)
        
        dx = xs[None, :] - cx   # [G, P]
        dy = (iy - cy) + tl.zeros_like(dx)
        vx = inv00 * dx + inv01 * dy
        vy = inv01 * dx + inv11 * dy
        q  = 0.5 * (dx * vx + dy * vy)
        w  = tl.exp(-q)
        w  = w[:, :, None] # [G, P, 1]

        # (col * w) -> [G, 1, F] * [G, P, 1] = [G, P, F]
        # m_row -> [G, 1, 1]
        contrib = tl.sum((col * w) * m_row, axis=0)
        offs_base = (iy * Wc + xs[None, :]) * BLOCK_FEAT
        offs_feat = offs_base + tl.arange(0, BLOCK_FEAT)[:, None]
        tl.atomic_add(img_ptr + offs_feat, tl.trans(contrib), mask=xs_mask[None, :])



@triton.jit
def _render_tiles_backward(
    grad_img_ptr, H, Wc,
    tile_size, tiles_x, tiles_y,
    tile_starts_ptr, tile_indices_ptr,
    cx_ptr, cy_ptr, sx_ptr, sy_ptr, rho_ptr, color_ptr,
    g_c_ptr, g_cx_ptr, g_cy_ptr, g_sx_ptr, g_sy_ptr, g_rho_ptr,
    min_sigma, det_eps,
    BLOCK_PIX: tl.constexpr, 
    GAUSS_CHUNK: tl.constexpr,
    BLOCK_FEAT: tl.constexpr
):
    pid_tile  = tl.program_id(0)
    pid_chunk = tl.program_id(1)

    tx = pid_tile % tiles_x
    ty = pid_tile // tiles_x
    x0 = tx * tile_size
    y0 = ty * tile_size
    xs = x0 + tl.arange(0, BLOCK_PIX)
    xs_mask = (xs < Wc)
    xs = tl.minimum(xs, Wc - 1)

    start = tl.load(tile_starts_ptr + pid_tile)
    end   = tl.load(tile_starts_ptr + pid_tile + 1)
    base  = start + pid_chunk * GAUSS_CHUNK
    g_idx = base + tl.arange(0, GAUSS_CHUNK)
    g_mask = g_idx < end
    gids = tl.load(tile_indices_ptr + g_idx, mask=g_mask, other=0)

    cx  = tl.load(cx_ptr  + gids, mask=g_mask, other=0.0)
    cy  = tl.load(cy_ptr  + gids, mask=g_mask, other=0.0)
    sx  = tl.load(sx_ptr  + gids, mask=g_mask, other=0.0)
    sy  = tl.load(sy_ptr  + gids, mask=g_mask, other=0.0)
    rho = tl.load(rho_ptr + gids, mask=g_mask, other=0.0)

    f_offs = tl.arange(0, BLOCK_FEAT)[None, :]
    gids_exp = gids[:, None]
    col_offs = gids_exp * BLOCK_FEAT + f_offs
    col = tl.load(color_ptr + col_offs, mask=g_mask[:, None], other=0.0)

    sx  = tl.maximum(sx, min_sigma)
    sy  = tl.maximum(sy, min_sigma)
    rho = tl.maximum(tl.minimum(rho, 0.999), -0.999)

    sxx = sx * sx
    syy = sy * sy
    sxy = rho * sx * sy
    det = tl.maximum(sxx * syy - sxy * sxy, det_eps)
    inv00 =  syy / det
    inv01 = -sxy / det
    inv11 =  sxx / det

    S0  = tl.zeros((GAUSS_CHUNK, BLOCK_FEAT), dtype=tl.float32)
    Sx  = tl.zeros((GAUSS_CHUNK,), dtype=tl.float32)
    Sy  = tl.zeros_like(Sx)
    Sdx = tl.zeros_like(Sx)
    Sdy = tl.zeros_like(Sx)
    Sdr = tl.zeros_like(Sx)

    cx_b  = cx[:, None]
    cy_b  = cy[:, None]
    col_b = col[:, None, :]
    inv00_b = inv00[:, None]
    inv01_b = inv01[:, None]
    inv11_b = inv11[:, None]
    sxv = sx[:, None]
    syv = sy[:, None]
    rhov = rho[:, None]
    m_row = g_mask.to(tl.float32)[:, None]

    for ry in range(0, BLOCK_PIX):
        iy = tl.minimum(y0 + ry, H - 1)
        
        offs_base = (iy * Wc + xs[None, :]) * BLOCK_FEAT
        offs_feat = offs_base + tl.arange(0, BLOCK_FEAT)[:, None]
        gi_row = tl.load(grad_img_ptr + offs_feat, mask=xs_mask[None, :], other=0.0)
        gi_row_t = tl.trans(gi_row)
        gi_row_b = gi_row_t[None, :, :]

        dx = xs[None, :] - cx_b
        dy = (iy - cy_b) + tl.zeros_like(dx)
        vx = inv00_b * dx + inv01_b * dy
        vy = inv01_b * dx + inv11_b * dy
        q  = 0.5 * (dx * vx + dy * vy)
        w  = tl.exp(-q)
        mw = w * m_row

        S0 += tl.sum(gi_row_b * mw[:, :, None], axis=1)
        grad_x_color_sum = tl.sum(col_b * gi_row_b, axis=2)
        # [G, 1, F] * [1, P, F] -> [G, P, F], sum(2) -> [G, P]
        
        grad_x_color_sum_m = grad_x_color_sum * mw
        
        Sx  += tl.sum(grad_x_color_sum_m * vx, axis=1)
        Sy  += tl.sum(grad_x_color_sum_m * vy, axis=1)

        # d(q)/d(sigma)
        t_dx = 0.5 * ((2.0*sxv)*vx*vx + 2.0*(rhov*syv)*vx*vy)
        t_dy = 0.5 * (2.0*(rhov*sxv)*vx*vy + (2.0*syv)*vy*vy)
        t_dr = 0.5 * (2.0*(sxv*syv)*vx*vy)

        Sdx += tl.sum(grad_x_color_sum_m * t_dx, axis=1)
        Sdy += tl.sum(grad_x_color_sum_m * t_dy, axis=1)
        Sdr += tl.sum(grad_x_color_sum_m * t_dr, axis=1)

    tl.atomic_add(g_c_ptr + col_offs, S0, mask=g_mask[:, None])
    tl.atomic_add(g_cx_ptr  + gids, Sx, mask=g_mask)
    tl.atomic_add(g_cy_ptr  + gids, Sy, mask=g_mask)
    tl.atomic_add(g_sx_ptr  + gids, Sdx, mask=g_mask)
    tl.atomic_add(g_sy_ptr  + gids, Sdy, mask=g_mask)
    tl.atomic_add(g_rho_ptr + gids, Sdr, mask=g_mask)



class _TileGaussianRenderFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gs_params, H: int, W: int, cfg: RenderConfig):
        """
        gs_params: [B, N, 5 + D_feat] (x,y,sx,sy,rho, ...features)
        Returns: img [B, D_feat, H, W]
        """
        assert gs_params.is_cuda, "Please move gs_params to CUDA"
        torch.cuda.set_device(gs_params.device)
        B, N, D_in = gs_params.shape
        assert D_in >= 6, "gs_params must have at least 6 dimensions (x,y,sx,sy,rho,c...)"
        D_feat = D_in - 5
        
        device = gs_params.device
        dtype  = gs_params.dtype

        tile = cfg.tile_size
        Wc = B * W

        gs_geom = gs_params[..., :5].contiguous()
        x   = gs_geom[..., 0]
        y   = gs_geom[..., 1]
        sx  = gs_geom[..., 2]
        sy  = gs_geom[..., 3]
        rho = gs_geom[..., 4]
        
        c = gs_params[..., 5:].contiguous() # [B, N, D_feat]

        b_idx_f = torch.arange(B, device=device, dtype=dtype).view(B, 1)
        cx_global = (x * W + b_idx_f * W).reshape(-1).to(dtype)
        cy_global = (y * H).reshape(-1).to(dtype)
        cx_local = (x * W).reshape(-1).to(dtype)
        cy_local = (y * H).reshape(-1).to(dtype)

        sx_pix   = torch.clamp((sx * W).reshape(-1).to(dtype), min=cfg.min_sigma_px)
        sy_pix   = torch.clamp((sy * H).reshape(-1).to(dtype), min=cfg.min_sigma_px)
        rho_stab = torch.clamp(rho.reshape(-1).to(dtype), min=-cfg.clamp_rho, max=cfg.clamp_rho)
        
        c_flat   = c.reshape(-1, D_feat).contiguous() # [G, D_feat]
        G = cx_local.numel()
        assert G == c_flat.shape[0]

        b_idx_i = torch.arange(B, device=device, dtype=torch.int64).view(B, 1).expand(B, N)
        b_of_gauss = b_idx_i.reshape(-1)

        starts, indices, (tiles_y, tiles_x_total) = _build_tile_gauss_csr_batched(
            cx_local, cy_local, sx_pix, sy_pix, rho_stab,
            H, W, tile,
            b_of_gauss, B
        )

        img_flat = torch.zeros((H, Wc, D_feat), dtype=dtype, device=device)

        counts = (starts[1:] - starts[:-1]).to(torch.int32)
        max_count = int(counts.max().item()) if counts.numel() > 0 else 0
        GAUSS_CHUNK = cfg.max_gauss_chunk
        chunks_per_tile = max((max_count + GAUSS_CHUNK - 1) // GAUSS_CHUNK, 1)
        grid = (tiles_x_total * tiles_y, chunks_per_tile)
        det_eps = 1e-8

        _render_tiles_forward[grid](
            img_flat, H, Wc,
            tile, tiles_x_total, tiles_y,
            starts, indices,
            cx_global, cy_global, sx_pix, sy_pix, rho_stab, c_flat,
            float(cfg.min_sigma_px), det_eps,
            BLOCK_PIX=tile, 
            GAUSS_CHUNK=GAUSS_CHUNK,
            BLOCK_FEAT=D_feat,
            num_warps=4, num_stages=2
        )

        img = img_flat.view(H, B, W, D_feat).permute(1, 3, 0, 2).contiguous()

        ctx.cfg = cfg
        ctx.H = H
        ctx.W = W
        ctx.starts = starts
        ctx.indices = indices
        ctx.cx = cx_global
        ctx.cy = cy_global
        ctx.sx = sx_pix
        ctx.sy = sy_pix
        ctx.rho = rho_stab
        ctx.c = c_flat
        ctx.B = B
        ctx.Wc = Wc
        ctx.G = G
        ctx.D_feat = D_feat
        ctx.tiles_x = int(tiles_x_total)
        ctx.tiles_y = int(tiles_y)
        return img

    @staticmethod
    def backward(ctx, grad_img):
        """
        grad_img: [B, D_feat, H, W]
        """
        torch.cuda.set_device(grad_img.device)
        cfg  = ctx.cfg
        H    = ctx.H
        W    = ctx.W
        B    = ctx.B
        D_feat = ctx.D_feat
        tile = cfg.tile_size

        if grad_img.dim() == 4:
            grad_img_permuted = grad_img.permute(2, 0, 3, 1).contiguous()
            grad_img_2d = grad_img_permuted.view(H, ctx.Wc, D_feat)
        else:
            raise RuntimeError(f"Unexpected grad shape {grad_img.shape}, expected [B,D_feat,H,W].")

        tiles_x = ctx.tiles_x
        tiles_y = ctx.tiles_y
        T = tiles_x * tiles_y

        G = ctx.G
        device = grad_img_2d.device
        dtype  = grad_img_2d.dtype
        g_c   = torch.zeros((G, D_feat), dtype=dtype, device=device)
        g_cx  = torch.zeros(G, dtype=dtype, device=device)
        g_cy  = torch.zeros(G, dtype=dtype, device=device)
        g_sx  = torch.zeros(G, dtype=dtype, device=device)
        g_sy  = torch.zeros(G, dtype=dtype, device=device)
        g_rho = torch.zeros(G, dtype=dtype, device=device)

        counts = (ctx.starts[1:] - ctx.starts[:-1]).to(torch.int32)
        max_count = int(counts.max().item()) if T > 0 else 0
        GAUSS_CHUNK = cfg.max_gauss_chunk
        chunks_per_tile = max((max_count + GAUSS_CHUNK - 1) // GAUSS_CHUNK, 1)
        grid = (T, chunks_per_tile)

        det_eps = 1e-8
        min_sigma_val = float(cfg.min_sigma_px)

        _render_tiles_backward[grid](
            grad_img_2d, H, ctx.Wc,
            tile, tiles_x, tiles_y,
            ctx.starts, ctx.indices,
            ctx.cx, ctx.cy, ctx.sx, ctx.sy, ctx.rho, ctx.c,
            g_c, g_cx, g_cy, g_sx, g_sy, g_rho,
            min_sigma_val, det_eps,
            BLOCK_PIX=tile, 
            GAUSS_CHUNK=GAUSS_CHUNK,
            BLOCK_FEAT=D_feat,
            num_warps=4, num_stages=2
        )

        N = G // B
        gx   = (g_cx.view(B, N) * W).contiguous()
        gy   = (g_cy.view(B, N) * H).contiguous()
        gsx  = (g_sx.view(B, N) * W).contiguous()
        gsy  = (g_sy.view(B, N) * H).contiguous()
        grho =  g_rho.view(B, N).contiguous()
        gc   =  g_c.view(B, N, D_feat).contiguous()

        grad_params = torch.zeros((B, N, 5 + D_feat), dtype=dtype, device=device)
        grad_params[..., 0] = gx
        grad_params[..., 1] = gy
        grad_params[..., 2] = gsx
        grad_params[..., 3] = gsy
        grad_params[..., 4] = grho
        grad_params[..., 5:] = gc
        
        return grad_params, None, None, None
    

def render_gaussians(
    gs_params: torch.Tensor,
    resolution: Tuple[int, int],
    tile_size: int = 16,
    max_gauss_chunk: int = 32,
    min_sigma_px: float = 0.1,
    clamp_rho: float = 0.999
) -> torch.Tensor:
    H, W = resolution
    B, N, D_in = gs_params.shape
    if D_in <= 5:
        raise ValueError(f"gs_params must have > 5 channels. Got {D_in} (x,y,sx,sy,rho,...features)")
        
    cfg = RenderConfig(
        tile_size=tile_size,
        max_gauss_chunk=max_gauss_chunk,
        min_sigma_px=min_sigma_px,
        clamp_rho=clamp_rho
    )
    return _TileGaussianRenderFn.apply(gs_params, int(H), int(W), cfg)