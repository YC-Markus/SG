import triton
import triton.language as tl
import torch
import math
import torch.nn.functional as F



@triton.jit
def union4_kernel(
    labels_ptr,
    parent_ptr,
    H: tl.int32, W: tl.int32,
    stride_h: tl.int32, stride_w: tl.int32
):
    pid = tl.program_id(0)
    N   = H * W
    if pid >= N:
        return
    y = pid // W
    x = pid - y * W
    lab = tl.load(labels_ptr + y * stride_h + x * stride_w)

    if x + 1 < W:
        nlab = tl.load(labels_ptr + y*stride_h + (x+1)*stride_w)
        if nlab == lab:
            id1 = pid
            id2 = pid + 1
            done = tl.cast(0, tl.int32)
            
            while done == tl.cast(0, tl.int32):
                r1 = id1
                p1 = tl.load(parent_ptr + r1)
                while r1 != p1:
                    r1 = p1
                    p1 = tl.load(parent_ptr + r1)
                
                r2 = id2
                p2 = tl.load(parent_ptr + r2)
                while r2 != p2:
                    r2 = p2
                    p2 = tl.load(parent_ptr + r2)

                equal = r1 == r2
                done = tl.where(equal, tl.cast(1, tl.int32), done)
                
                hi = tl.where(r1 > r2, r1, r2)
                lo = tl.where(r1 > r2, r2, r1)
                
                prev = tl.atomic_cas(parent_ptr + hi, hi, lo)
                done = tl.where(prev == hi, tl.cast(1, tl.int32), done)

    if y + 1 < H:
        nlab = tl.load(labels_ptr + (y + 1) * stride_h + x * stride_w)
        if nlab == lab:
            id1 = pid
            id2 = pid + W
            done = tl.cast(0, tl.int32) 
            while done == tl.cast(0, tl.int32):
                r1 = id1
                p1 = tl.load(parent_ptr + r1)
                while r1 != p1:
                    r1 = p1
                    p1 = tl.load(parent_ptr + r1)
                
                r2 = id2
                p2 = tl.load(parent_ptr + r2)
                while r2 != p2:
                    r2 = p2
                    p2 = tl.load(parent_ptr + r2)

                equal = r1 == r2
                done = tl.where(equal, tl.cast(1, tl.int32), done)
                
                hi = tl.where(r1 > r2, r1, r2)
                lo = tl.where(r1 > r2, r2, r1)
                
                prev = tl.atomic_cas(parent_ptr + hi, hi, lo)
                done = tl.where(prev == hi, tl.cast(1, tl.int32), done)



@triton.jit
def compress_kernel(parent_ptr, N: tl.int32):
    pid = tl.program_id(0)
    if pid >= N:
        return
        
    current = pid
    parent = tl.load(parent_ptr + current)
    while current != parent:
        current = parent
        parent = tl.load(parent_ptr + current)
    root = current
    
    tl.store(parent_ptr + pid, root)



@triton.jit
def size_kernel(parent_ptr, size_ptr, N: tl.int32):
    pid = tl.program_id(0)
    if pid >= N:
        return
    root = tl.load(parent_ptr + pid)
    tl.atomic_add(size_ptr + root, 1)


@triton.jit
def merge_small_kernel(
    parent_ptr, size_ptr,
    H: tl.int32, W: tl.int32,
    min_size: tl.int32
):
    pid = tl.program_id(0)
    N   = H * W
    if pid >= N:
        return

    root = tl.load(parent_ptr + pid) 
    sz   = tl.load(size_ptr + root)
    if sz >= min_size:
        return 

    y = pid // W
    x = pid - y * W
    best_rt = root
    best_sz = 0

    if x > 0:
        nroot = tl.load(parent_ptr + pid - 1) 
        if nroot != root:
            nsz   = tl.load(size_ptr + nroot)
            cond  = nsz > best_sz
            best_sz = tl.where(cond, nsz, best_sz)
            best_rt = tl.where(cond, nroot, best_rt)
    if x + 1 < W:
        nroot = tl.load(parent_ptr + pid + 1)
        if nroot != root:
            nsz   = tl.load(size_ptr + nroot)
            cond  = nsz > best_sz
            best_sz = tl.where(cond, nsz, best_sz)
            best_rt = tl.where(cond, nroot, best_rt)
    if y > 0:
        nroot = tl.load(parent_ptr + pid - W)
        if nroot != root:
            nsz   = tl.load(size_ptr + nroot)
            cond  = nsz > best_sz
            best_sz = tl.where(cond, nsz, best_sz)
            best_rt = tl.where(cond, nroot, best_rt)

    if y + 1 < H:
        nroot = tl.load(parent_ptr + pid + W) 
        if nroot != root:
            nsz   = tl.load(size_ptr + nroot)
            cond  = nsz > best_sz
            best_sz = tl.where(cond, nsz, best_sz)
            best_rt = tl.where(cond, nroot, best_rt)
            
    if best_rt != root:
        hi = tl.where(root > best_rt, root, best_rt)
        lo = tl.where(root > best_rt, best_rt, root)
        tl.atomic_cas(parent_ptr + hi, hi, lo)



@torch.no_grad()
def _enforce_connectivity_gpu(labels, min_size: int, compress_iters: int = 5, merge_iters: int = 3):
    assert labels.is_cuda
    H, W = labels.shape
    N    = H * W
    device = labels.device

    labels_i32 = labels.to(torch.int32).contiguous()
    parent = torch.arange(N, dtype=torch.int32, device=device)
    stride_h, stride_w = labels_i32.stride()
    grid = (N,)
    union4_kernel[grid](
        labels_i32, parent,
        H, W,
        stride_h, stride_w
    )

    comp_grid = (N,)

    for merge_iter_idx in range(merge_iters):
        for _ in range(compress_iters):
            compress_kernel[comp_grid](parent, N)
        size = torch.zeros(N, dtype=torch.int32, device=device)
        size_kernel[comp_grid](parent, size, N)
        merge_small_kernel[grid](
            parent, size,
            H, W,
            min_size
        )

    for _ in range(compress_iters):
        compress_kernel[comp_grid](parent, N)

    roots = parent
    flat_labels_in = labels_i32.view(-1)
    new_labels_flat = flat_labels_in[roots]
    labels_out = new_labels_flat.view(H, W).to(labels.dtype)
    return labels_out

@torch.no_grad()
def enforce_connectivity_batch_gpu(labels_tensor: torch.Tensor,
                                   min_size: int):
    B, H, W = labels_tensor.shape
    out = torch.empty_like(labels_tensor)
    for b in range(B):
        out[b] = _enforce_connectivity_gpu(labels_tensor[b], min_size)
    return out




@torch.no_grad()
def _compute_gradient_batch(imgs):
    B, C, H, W = imgs.shape
    device = imgs.device
    dtype = imgs.dtype

    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=dtype, device=device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=dtype, device=device).view(1,1,3,3)

    sobel_x_w = sobel_x.repeat(C, 1, 1, 1) # [C, 1, 3, 3]
    sobel_y_w = sobel_y.repeat(C, 1, 1, 1) # [C, 1, 3, 3]

    gx = F.conv2d(imgs, sobel_x_w, padding=1, groups=C)  # [B, C, H, W]
    gy = F.conv2d(imgs, sobel_y_w, padding=1, groups=C)  # [B, C, H, W]
    grad = torch.sqrt(gx.pow(2) + gy.pow(2)).mean(dim=1, keepdim=False) # [B, H, W]
    return grad


@triton.jit
def init_centers_kernel(
    grads_ptr,       # [B, H, W]
    centers_in_ptr,  # [B, K, 2]
    centers_out_ptr, # [B, K, 2]
    B: tl.int32,
    H: tl.int32,
    W: tl.int32,
    K: tl.int32,
    R: tl.int32, 
    stride_grad_b, stride_grad_h, stride_grad_w,
    

    stride_ctr_in_b, stride_ctr_in_k, stride_ctr_in_yx,
    stride_ctr_out_b, stride_ctr_out_k, stride_ctr_out_yx
):
    b = tl.program_id(0)
    k = tl.program_id(1)

    ctr_in_base = centers_in_ptr + b * stride_ctr_in_b + k * stride_ctr_in_k
    y0 = tl.load(ctr_in_base + 0 * stride_ctr_in_yx).to(tl.float32)
    x0 = tl.load(ctr_in_base + 1 * stride_ctr_in_yx).to(tl.float32)

    y0_int = tl.math.floor(y0 + 0.5).to(tl.int32)
    x0_int = tl.math.floor(x0 + 0.5).to(tl.int32)

    y_min = tl.maximum(R, y0_int - R)
    y_max = tl.minimum(H - R - 1, y0_int + R)
    x_min = tl.maximum(R, x0_int - R)
    x_max = tl.minimum(W - R - 1, x0_int + R)

    grad_base = grads_ptr + b * stride_grad_b
    min_grad = 1e10
    best_y = y_min
    best_x = x_min

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            g = tl.load(grad_base + y * stride_grad_h + x * stride_grad_w)
            if g < min_grad:
                min_grad = g
                best_y = y
                best_x = x

    ctr_out_base = centers_out_ptr + b * stride_ctr_out_b + k * stride_ctr_out_k
    tl.store(ctr_out_base + 0 * stride_ctr_out_yx, best_y.to(tl.float32))
    tl.store(ctr_out_base + 1 * stride_ctr_out_yx, best_x.to(tl.float32))


@torch.no_grad()
def _move_to_low_gradient_batch(init_yx, grads, radius: int = 2):
    B, K, _ = init_yx.shape
    _, H, W = grads.shape
    centers_out = torch.empty_like(init_yx)

    assert init_yx.is_cuda and grads.is_cuda and centers_out.is_cuda
    assert grads.dtype == torch.float32 and init_yx.dtype == torch.float32

    if not centers_out.is_contiguous():
        centers_out = centers_out.contiguous()
    grid = (B, K)

    
    init_centers_kernel[grid](
        grads, init_yx, centers_out,
        B, H, W, K, radius,
        grads.stride(0), grads.stride(1), grads.stride(2),
        init_yx.stride(0), init_yx.stride(1), init_yx.stride(2),
        centers_out.stride(0), centers_out.stride(1), centers_out.stride(2)
    )    
    return centers_out


@triton.jit
def slico_m_kernel(
    imgs_ptr,          # [B, C, H, W]
    centers_yx_ptr,    # [B, K, 2]
    centers_color_ptr, # [B, K, C]
    m_squared_ptr,     # [B, K] (output)
    B: tl.int32, C: tl.constexpr, H: tl.int32, W: tl.int32, K: tl.int32,
    S_float: tl.float32,
    stride_img_b, stride_img_c, stride_img_h, stride_img_w,
    stride_ctr_b, stride_ctr_k, stride_ctr_yx,
    stride_ccol_b, stride_ccol_k, stride_ccol_c,
    stride_m_b, stride_m_k
):
    b = tl.program_id(0)
    k = tl.program_id(1)

    ctr_yx_base = centers_yx_ptr + b * stride_ctr_b + k * stride_ctr_k
    cy = tl.load(ctr_yx_base + 0 * stride_ctr_yx).to(tl.float32)
    cx = tl.load(ctr_yx_base + 1 * stride_ctr_yx).to(tl.float32)
    ccol_base = centers_color_ptr + b * stride_ccol_b + k * stride_ccol_k
    
    offs_c_init = tl.arange(0, C)
    ccol = tl.load(ccol_base + offs_c_init * stride_ccol_c).to(tl.float32) 

    win_r = 2 * S_float
    y_min = tl.maximum(0, tl.math.floor(cy - win_r)).to(tl.int32)
    y_max = tl.minimum(H, tl.math.ceil(cy + win_r)).to(tl.int32)
    x_min = tl.maximum(0, tl.math.floor(cx - win_r)).to(tl.int32)
    x_max = tl.minimum(W, tl.math.ceil(cx + win_r)).to(tl.int32)

    img_base = imgs_ptr + b * stride_img_b
    max_dc2 = 0.0

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            offs_c = tl.arange(0, C)
            p_ptr = img_base + (offs_c * stride_img_c) + (y * stride_img_h) + (x * stride_img_w)
            mask_c = offs_c < C
            p = tl.load(p_ptr, mask=mask_c, other=0.0)
            diff = p - ccol
            dc2 = tl.sum(diff * diff)             
            max_dc2 = tl.maximum(max_dc2, dc2)

    tl.store(m_squared_ptr + b * stride_m_b + k * stride_m_k, max_dc2 + 1e-10)

@triton.jit
def assignment_kernel(
    imgs_ptr,          # [B, C, H, W]
    centers_yx_ptr,    # [B, K, 2]
    centers_color_ptr,  # [B, K, C]
    m_squared_ptr,
    labels_out_ptr,
    dists_out_ptr,
    B: tl.int32, C: tl.constexpr, H: tl.int32, W: tl.int32, K: tl.int32,
    S_float: tl.float32, 
    S_squared: tl.float32,
    m_over_S_squared: tl.float32,
    slico: tl.int32,    # 1 for True, 0 for False
    stride_img_b, stride_img_c, stride_img_h, stride_img_w,
    stride_ctr_b, stride_ctr_k, stride_ctr_yx,
    stride_ccol_b, stride_ccol_k, stride_ccol_c,
    stride_m_b, stride_m_k,
    stride_lab_b, stride_lab_h, stride_lab_w,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    px = offs_w[None, :] # [1, BW]
    py = offs_h[:, None] # [BH, 1]

    mask_h = py < H
    mask_w = px < W
    px_mask = mask_h & mask_w # [BH, BW]

    img_base = imgs_ptr + b * stride_img_b
    pcol = tl.load(
        img_base + (tl.arange(0, C)[:, None, None] * stride_img_c) \
                + (py[None, :, :] * stride_img_h) \
                + (px[None, :, :] * stride_img_w),
        mask = (px_mask[None, :, :]) & (tl.arange(0, C)[:, None, None] < C),
        other = 0.0
    ).to(tl.float32) # [C, BH, BW]

    min_dist = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_W), 1e10, dtype=tl.float32)
    min_label = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_W), -1, dtype=tl.int32)
    win_r_sq = (2 * S_float) * (2 * S_float) # (2S)^2

    ctr_yx_base = centers_yx_ptr + b * stride_ctr_b
    ccol_base = centers_color_ptr + b * stride_ccol_b
    m_base = m_squared_ptr + b * stride_m_b

    for k in range(K):
        cy = tl.load(ctr_yx_base + k * stride_ctr_k + 0 * stride_ctr_yx).to(tl.float32)
        cx = tl.load(ctr_yx_base + k * stride_ctr_k + 1 * stride_ctr_yx).to(tl.float32)
        ccol_k_ptr = ccol_base + k * stride_ccol_k
        ccol_k = tl.load(ccol_k_ptr + tl.arange(0, C) * stride_ccol_c).to(tl.float32)

        diff_y = py.to(tl.float32) - cy
        diff_x = px.to(tl.float32) - cx
        ds2 = (diff_y * diff_y) + (diff_x * diff_x)

        mask_ds = ds2 < win_r_sq
        diff = pcol - ccol_k[:, None, None]
        dc2 = tl.sum(diff * diff, axis=0) 

        D = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_W), 0.0, dtype=tl.float32)
        if slico:
            m_k_sq = tl.load(m_base + k * stride_m_k)
            D = (dc2 / m_k_sq) + (ds2 / S_squared)
        else:
            D = dc2 + m_over_S_squared * ds2

        mask_update = (D < min_dist) & mask_ds
        min_dist = tl.where(mask_update, D, min_dist)
        min_label = tl.where(mask_update, k, min_label)

    lab_base = labels_out_ptr + b * stride_lab_b
    dist_base = dists_out_ptr + b * stride_lab_b

    lab_ptr = lab_base + py * stride_lab_h + px * stride_lab_w
    dist_ptr = dist_base + py * stride_lab_h + px * stride_lab_w
    
    tl.store(lab_ptr, min_label.to(tl.int64), mask=px_mask)
    tl.store(dist_ptr, min_dist, mask=px_mask)






@torch.no_grad() 
def slic_batch_parallel(
    imgs_tensor: torch.Tensor,
    n_segments: int = 100,
    compactness: float = 10.0,
    n_iters: int = 10,
    enforce_conn: bool = True,
    slico: bool = True,
    hex_grid: bool = True,
    move_grad_radius: int = 2,
    conn_min_size_ratio: float = 0.25,
    return_numpy: bool = False,
    tile: int = 16,
):

    B, C, H, W = imgs_tensor.shape
    N = H * W
    device = imgs_tensor.device
    dtype = imgs_tensor.dtype

    S_float = float(torch.sqrt(torch.tensor(N / n_segments, dtype=torch.float32)))
    S_float = max(S_float, 1.0)

    centers_list = []
    if hex_grid:
        hex_spacing_factor = math.sqrt(2.0 / math.sqrt(3.0))
        def _calculate_K_for_S(S_spacing, H, W, hex_factor):
            if S_spacing <= 1e-6: return H * W
            S_hex_base = S_spacing * hex_factor
            Sv = S_hex_base * math.sqrt(3) / 2
            Sh = S_hex_base
            if Sv <= 1e-6 or Sh <= 1e-6: return H * W
            
            K_total = 0
            start_y = Sv / 2
            if start_y >= H: return 0
            Ky = math.floor((H - start_y - 1e-6) / Sv) + 1

            for row_i in range(int(Ky)):
                offset = 0.0 if (row_i % 2 == 0) else Sh / 2
                start_x = offset + Sh / 2
                if start_x >= W: continue
                
                Kx = math.floor((W - start_x - 1e-6) / Sh) + 1
                K_total += Kx
            return K_total
        K_init = _calculate_K_for_S(S_float, H, W, hex_spacing_factor)
        S_adj = S_float 

        if K_init > n_segments:        
            S_low = S_float
            S_high = float(max(H, W))
            
            for _ in range(20):
                S_mid = (S_low + S_high) / 2.0
                K_mid = _calculate_K_for_S(S_mid, H, W, hex_spacing_factor)
                
                if K_mid <= n_segments:
                    S_adj = S_mid
                    S_high = S_mid
                else:
                    S_low = S_mid

        S_hex_base = S_adj * hex_spacing_factor
        Sv = S_hex_base * math.sqrt(3) / 2
        Sh = S_hex_base 
        
        grid_y = torch.arange(Sv/2, H, Sv, device=device, dtype=dtype)
        for row_i, gy in enumerate(grid_y):
            offset = 0.0 if (row_i % 2 == 0) else Sh / 2
            grid_x = torch.arange(offset + Sh/2, W, Sh, device=device, dtype=dtype)
            if grid_x.numel() == 0: continue
            gy_vec = gy.expand_as(grid_x)
            yx = torch.stack([gy_vec, grid_x], dim=1) # [K_row, 2]
            centers_list.append(yx)
    else:
        grid_y = torch.arange(S_float/2, H, S_float, device=device, dtype=dtype)
        grid_x = torch.arange(S_float/2, W, S_float, device=device, dtype=dtype)
        yx = torch.cartesian_prod(grid_y, grid_x) # [K_y * K_x, 2]
        centers_list.append(yx)

    init_centers = torch.cat(centers_list, dim=0)  
    K = init_centers.shape[0]
    assert K > 0
    
    init_centers_batch = init_centers.unsqueeze(0).expand(B, K, 2)
    grads = _compute_gradient_batch(imgs_tensor) # [B, H, W]
    centers_yx = _move_to_low_gradient_batch(init_centers_batch, grads, move_grad_radius) # [B, K, 2]
    
    y_norm = (centers_yx[..., 0] / (H - 1)) * 2 - 1 # [B, K]
    x_norm = (centers_yx[..., 1] / (W - 1)) * 2 - 1 # [B, K]
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1) # [B, 1, K, 2]
    
    centers_color = F.grid_sample(
        imgs_tensor,  
        grid,  
        mode='bilinear',  
        padding_mode='reflection',  
        align_corners=True
    ).squeeze(2).permute(0, 2, 1) # [B, K, C]

    labels = torch.full((B, H, W), -1, dtype=torch.int64, device=device)
    dists = torch.full((B, H, W), float('inf'), dtype=dtype, device=device)

    ys = torch.arange(H, device=device, dtype=dtype).view(1, H, 1).expand(B, H, W)
    xs = torch.arange(W, device=device, dtype=dtype).view(1, 1, W).expand(B, H, W)

    m_over_S_squared = (compactness / S_float)**2
    S_squared = S_float * S_float
    BLOCK_SIZE_H, BLOCK_SIZE_W = tile, tile

    for i in range(n_iters):
        m_squared = torch.tensor([1e-10], device=device, dtype=dtype) 
        if slico:
            m_squared = torch.empty(B, K, device=device, dtype=dtype)
            grid_m = (B, K)
            slico_m_kernel[grid_m](
                imgs_tensor, centers_yx, centers_color, m_squared,
                B, C, H, W, K, S_float,
                imgs_tensor.stride(0), imgs_tensor.stride(1), imgs_tensor.stride(2), imgs_tensor.stride(3),
                centers_yx.stride(0), centers_yx.stride(1), centers_yx.stride(2),
                centers_color.stride(0), centers_color.stride(1), centers_color.stride(2),
                m_squared.stride(0), m_squared.stride(1)
            )

        dists.fill_(float('inf'))
        labels.fill_(-1)
        
        grid_assign = (B, triton.cdiv(H, BLOCK_SIZE_H), triton.cdiv(W, BLOCK_SIZE_W))
        assignment_kernel[grid_assign](
            imgs_tensor, centers_yx, centers_color, m_squared,
            labels, dists,
            B, C, H, W, K,
            S_float, S_squared, m_over_S_squared,
            1 if slico else 0,
            imgs_tensor.stride(0), imgs_tensor.stride(1), imgs_tensor.stride(2), imgs_tensor.stride(3),
            centers_yx.stride(0), centers_yx.stride(1), centers_yx.stride(2),
            centers_color.stride(0), centers_color.stride(1), centers_color.stride(2),
            m_squared.stride(0), m_squared.stride(1) if slico else 0,
            labels.stride(0), labels.stride(1), labels.stride(2),
            BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W
        )
        
        flat_labels = labels.view(B, -1)                # [B, N]
        flat_pixels = imgs_tensor.reshape(B, C, -1).permute(0, 2, 1) # [B, N, C]
        flat_y = ys.reshape(B, -1)                      # [B, N]
        flat_x = xs.reshape(B, -1)                      # [B, N]
        
        valid_mask = flat_labels >= 0                   # [B, N]
        valid_mask_1d = valid_mask.view(-1)             # [B*N]
        N_valid = valid_mask_1d.sum()
        if N_valid == 0:
            break
        
        batch_offset = torch.arange(B, device=device, dtype=torch.int64).view(B, 1) * K
        global_labels = flat_labels + batch_offset      # [B, N]
        
        labels_idx = global_labels[valid_mask]          # [N_valid]
        
        pixels_src = flat_pixels[valid_mask]            # [N_valid, C]
        y_src = flat_y[valid_mask]                      # [N_valid]
        x_src = flat_x[valid_mask]                      # [N_valid]
        yx_src = torch.stack([y_src, x_src], dim=1)     # [N_valid, 2]
        ones_src = torch.ones_like(y_src)               # [N_valid]

        sums_color = torch.zeros((B * K, C), dtype=dtype, device=device)
        sums_yx = torch.zeros((B * K, 2), dtype=dtype, device=device)
        _counts = torch.zeros(B * K, dtype=dtype, device=device)
        
        sums_color.index_add_(0, labels_idx, pixels_src)
        sums_yx.index_add_(0, labels_idx, yx_src)
        _counts.index_add_(0, labels_idx, ones_src)
        counts = _counts.clamp(min=1.0) 
        
        new_centers_color_1d = sums_color / counts.unsqueeze(-1) 
        new_centers_yx_1d = sums_yx / counts.unsqueeze(-1) 
        
        nz_mask_1d = _counts > 0
        nz_mask_color = nz_mask_1d.view(B, K, 1).expand(B, K, C)
        nz_mask_yx = nz_mask_1d.view(B, K, 1).expand(B, K, 2)
        
        centers_color = torch.where(
            nz_mask_color,
            new_centers_color_1d.view(B, K, C),
            centers_color
        )
        centers_yx = torch.where(
            nz_mask_yx,
            new_centers_yx_1d.view(B, K, 2),
            centers_yx
        )

    if enforce_conn:
        min_size = max(1, int((N / max(K,1)) * conn_min_size_ratio))
        labels = enforce_connectivity_batch_gpu(labels, min_size)


    flat_labels = labels.view(B, -1)              # [B, N]
    valid_mask = flat_labels >= 0                 # [B, N]
    
    batch_offset = torch.arange(B, device=device, dtype=torch.int64).view(B, 1) * K
    global_labels = flat_labels + batch_offset    # [B, N]
    labels_idx = global_labels[valid_mask]        # [N_valid]

    flat_pixels = imgs_tensor.reshape(B, C, -1).permute(0, 2, 1) 
    
    ys_norm = ys / (H - 1) # [B, H, W]
    xs_norm = xs / (W - 1) # [B, H, W]
    
    flat_y_norm = ys_norm.reshape(B, -1)           # [B, N]
    flat_x_norm = xs_norm.reshape(B, -1)           # [B, N]
    
    # [B, N, 2] (x, y order)
    flat_xy_norm = torch.stack([flat_x_norm, flat_y_norm], dim=-1) 

    pixels_src = flat_pixels[valid_mask]         # [N_valid, C]
    xy_norm_src = flat_xy_norm[valid_mask]       # [N_valid, 2]
    ones_src = torch.ones(xy_norm_src.shape[0], dtype=dtype, device=device) # [N_valid]

    sums_color = torch.zeros((B * K, C), dtype=dtype, device=device)
    sums_xy_norm = torch.zeros((B * K, 2), dtype=dtype, device=device)
    _counts = torch.zeros(B * K, dtype=dtype, device=device)
    
    sums_color.index_add_(0, labels_idx, pixels_src)
    sums_xy_norm.index_add_(0, labels_idx, xy_norm_src)
    _counts.index_add_(0, labels_idx, ones_src)
    
    counts = _counts.clamp(min=1.0) # [B*K]
    counts_bkc = counts.view(B, K, 1) # [B, K, 1]

    final_centers_color = (sums_color / counts.unsqueeze(-1)).view(B, K, C)
    final_centers_xy_norm = (sums_xy_norm / counts.unsqueeze(-1)).view(B, K, 2)

    x_norm_src = xy_norm_src[:, 0]
    y_norm_src = xy_norm_src[:, 1]
    
    xx_src = x_norm_src * x_norm_src
    yy_src = y_norm_src * y_norm_src
    xy_src = x_norm_src * y_norm_src
    
    sums_xx = torch.zeros(B * K, dtype=dtype, device=device)
    sums_yy = torch.zeros(B * K, dtype=dtype, device=device)
    sums_xy = torch.zeros(B * K, dtype=dtype, device=device)
    
    sums_xx.index_add_(0, labels_idx, xx_src)
    sums_yy.index_add_(0, labels_idx, yy_src)
    sums_xy.index_add_(0, labels_idx, xy_src)
    
    E_xx = (sums_xx / counts).view(B, K)
    E_yy = (sums_yy / counts).view(B, K)
    E_xy = (sums_xy / counts).view(B, K)

    E_x = final_centers_xy_norm[..., 0]
    E_y = final_centers_xy_norm[..., 1]

    var_x = E_xx - E_x.pow(2)
    var_y = E_yy - E_y.pow(2)
    cov_xy = E_xy - E_x * E_y

    final_cov_matrix = torch.stack([
        torch.stack([var_x, cov_xy], dim=-1),
        torch.stack([cov_xy, var_y], dim=-1)
    ], dim=-2)
    
    empty_mask_1d = _counts == 0 
    empty_mask_feat = empty_mask_1d.view(B, K, 1)
    empty_mask_cov = empty_mask_1d.view(B, K, 1, 1)
    
    final_centers_color = final_centers_color.masked_fill(
        empty_mask_feat.expand_as(final_centers_color), 0.0)
    final_centers_xy_norm = final_centers_xy_norm.masked_fill(
        empty_mask_feat.expand_as(final_centers_xy_norm), 0.0)
    
    final_cov_matrix = final_cov_matrix.masked_fill(
        empty_mask_cov.expand_as(final_cov_matrix), 0.0)
    return labels, final_centers_color, final_centers_xy_norm, final_cov_matrix, empty_mask_feat