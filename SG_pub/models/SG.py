import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import xformers.ops as xops
from models.render import render_gaussians
from models.slico import slic_batch_parallel
import numpy as np
from models.unet import UNet


class CA(nn.Module):
    def __init__(self, dim, squeeze_factor=8):
        super(CA, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(dim, dim // squeeze_factor, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // squeeze_factor, dim, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_tmp = self.conv1(x)
        avg_out = self.shared_MLP(self.avg_pool(x_tmp))
        max_out = self.shared_MLP(self.max_pool(x_tmp))
        out = self.sigmoid(avg_out + max_out)
        return out, x_tmp



class ModalityFusion(nn.Module):
    def __init__(self, in_out_channels):
        super(ModalityFusion, self).__init__()
        self.fuse1 = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_out_channels, in_out_channels*2, kernel_size=1)
        )
        self.ca = CA(dim=in_out_channels)

    def forward(self, x_tar, x_aux, miu=1):
        params = self.fuse1(x_aux)
        gamma, beta = torch.chunk(params, 2, dim=1)
        x = x_tar * (1 + gamma) + beta
        c_weight, x_tmp = self.ca(x)
        x = x + miu * c_weight * x_tmp
        return x, c_weight





class MidFusion(nn.Module):
    def __init__(self, in_out_channels):
        super(MidFusion, self).__init__()
        self.fuse_conv1 = nn.Conv2d(in_out_channels, in_out_channels, kernel_size=3, padding=1, bias=True)
        self.act = nn.GELU()
        self.fuse_conv2 = nn.Conv2d(in_out_channels, in_out_channels, kernel_size=3, padding=1, bias=True)
        self.ca = CA(dim=in_out_channels)

    def forward(self, x, miu=1):
        x = self.act(self.fuse_conv1(x))
        x = self.fuse_conv2(x)
        c_weight, x_tmp = self.ca(x)
        x = x + miu * c_weight * x_tmp
        return x, c_weight



class FeatUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        scale_factor = 2
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=kernel_size, padding=padding)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x



def top_k_channel(x, weights, k_channels):
    b, c, h, w = x.shape
    weights_squeezed = weights.squeeze(-1).squeeze(-1) # (B, C)
    _, topk_indices = torch.topk(weights_squeezed, k_channels, dim=1, largest=True, sorted=False)
    # (B, K) -> (B, K, 1, 1)
    topk_indices_expanded = topk_indices.view(b, k_channels, 1, 1)
    # (B, K, 1, 1) -> (B, K, H, W)
    topk_indices_broadcasted = topk_indices_expanded.expand(-1, -1, h, w)
    x_reduced = torch.gather(x, dim=1, index=topk_indices_broadcasted)
    return x_reduced




def get_superpixel_centers(x, labels, empty_mask):
    B, C, H, W = x.shape
    K = empty_mask.shape[1]
    
    # x: [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
    x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
    # labels: [B, H, W] -> [B, H*W]
    labels_flat = labels.reshape(B, H * W)
    # [B, H*W] -> [B, H*W, K]
    labels_one_hot = torch.nn.functional.one_hot(labels_flat, num_classes=K).float()
    # [B, H*W, K] -> [B, K, 1]
    pixel_counts = labels_one_hot.sum(dim=1)
    pixel_counts_unsqueezed = pixel_counts.unsqueeze(-1)

    labels_one_hot_t = labels_one_hot.permute(0, 2, 1)
    sum_features = torch.bmm(labels_one_hot_t, x_flat)
    centers_color = sum_features / (pixel_counts_unsqueezed + 1e-6)
    centers_color = centers_color.masked_fill(empty_mask, 0.0)
    return centers_color



class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


class FFN2d(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, norm_layer=LayerNorm2d):
        super(FFN2d, self).__init__()
        self.norm = norm_layer(dim)
        self.project_in = nn.Conv2d(dim, hidden_dim*2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, stride=1, padding=1, groups=hidden_dim*2)
        self.project_out = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class MaskAttention(nn.Module):
    def __init__(self, dim, heads, qk_dim):
        super().__init__()
        self.heads = heads
        self.head_dim_q = qk_dim // heads
        self.head_dim_v = dim // heads
        self.scale = self.head_dim_q ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x, attn_bias=None): 
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        # Reshape for xformers: [B, N, H, D_head]
        q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.heads) 
        out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=self.scale)
        out = rearrange(out, 'b n h d -> b n (h d)')
        return out


from timm.models.layers import trunc_normal_
class LocalTokenAttention(nn.Module):
    def __init__(self, dim, heads, qk_dim, local_size, overlap=1, mlp_ratio=2):
        super().__init__()
        self.local_size = local_size
        self.dh, self.dw = local_size[0], local_size[1]
        self.heads = heads 
        
        self.stride_h, self.stride_w = self.dh - overlap, self.dw - overlap
        self.norm = nn.LayerNorm(dim)
        self.attn = MaskAttention(dim, heads, qk_dim) 
        self.ffn_2d = FFN2d(dim, dim*mlp_ratio, dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.dh - 1) * (2 * self.dw - 1), heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=.02)


        coords_h = torch.arange(self.dh)
        coords_w = torch.arange(self.dw)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) 
        coords_flat = torch.flatten(coords, 1) # (2, dh*dw)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :] # (2, dh*dw, dh*dw)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # (dh*dw, dh*dw, 2)

        relative_coords[:, :, 0] += self.dh - 1 
        relative_coords[:, :, 1] += self.dw - 1 

        relative_coords[:, :, 0] *= (2 * self.dw - 1)
        relative_position_index = relative_coords.sum(-1) # (dh*dw, dh*dw)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        B, C, H, W = x.shape
        
        patches = F.unfold(
            x, 
            kernel_size=(self.dh, self.dw), 
            stride=(self.stride_h, self.stride_w)
        )
        patches = patches.view(B, C, self.dh, self.dw, -1).permute(0, 4, 1, 2, 3).contiguous()
        # (B, L, C, dh, dw) -> (B*L, C, dh*dw)
        patches = patches.view(-1, C, self.dh * self.dw)
        # (B*L, C, dh*dw) -> (B*L, dh*dw, C)
        patches = patches.permute(0, 2, 1).contiguous()
        B_L = patches.shape[0]
        N_window = self.dh * self.dw
        # (N_window * N_window, )
        flat_index = self.relative_position_index.view(-1)
        # (N_window * N_window, heads)
        bias = self.relative_position_bias_table[flat_index]
        # (N_window, N_window, heads)
        bias = bias.view(N_window, N_window, self.heads)
        # (heads, N_window, N_window)
        bias = bias.permute(2, 0, 1).contiguous()
        # (1, heads, N_window, N_window)
        attn_bias = bias.unsqueeze(0)
        attn_bias = attn_bias.expand(B_L, -1, -1, -1)
        # x: (B*L, N_window, C)
        # attn_bias: (1, heads, N_window, N_window)
        attn_patches = self.attn(self.norm(patches), attn_bias=attn_bias)
        # (B*L, N_window, C) -> (B*L, C, N_window)
        attn_patches = attn_patches.permute(0, 2, 1).contiguous()
        # (B*L, C, N_window) -> (B, L, C, dh, dw)
        attn_patches = attn_patches.view(B, -1, C, self.dh, self.dw)
        # (B, L, C, dh, dw) -> (B, C, dh, dw, L)
        attn_patches = attn_patches.permute(0, 2, 3, 4, 1).contiguous()
        # (B, C, dh, dw, L) -> (B, C*dh*dw, L)
        attn_patches = attn_patches.view(B, C * self.dh * self.dw, -1)

        attn_out = F.fold(
            attn_patches,
            output_size=(H, W),
            kernel_size=(self.dh, self.dw),
            stride=(self.stride_h, self.stride_w)
        )
        
        ones = torch.ones_like(x)
        count_patches = F.unfold(
            ones,
            kernel_size=(self.dh, self.dw),
            stride=(self.stride_h, self.stride_w)
        )
        count = F.fold(
            count_patches,
            output_size=(H, W),
            kernel_size=(self.dh, self.dw),
            stride=(self.stride_h, self.stride_w)
        )
        
        attn_out = attn_out / (count + 1e-6)
        x = attn_out + x
        x_ffn = self.ffn_2d(x)
        x = x_ffn + x
        return x



class SuperpixelAttentionAggregator(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        self.norm_gs = nn.LayerNorm(dim)
        self.norm_pixels = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.pixel_pos_embed_layer = nn.Sequential(
            nn.Linear(2, dim), 
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.register_buffer('pixel_grid_norm', None, persistent=False)

    def _create_pixel_grid(self, B, H, W, device):
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij'
            )
            xx_norm = (xx + 0.5) / max(W, 1)
            yy_norm = (yy + 0.5) / max(H, 1)
            grid_norm = torch.stack([xx_norm, yy_norm], dim=-1)
            grid_norm_flat = grid_norm.reshape(1, H * W, 2).expand(B, -1, -1)
            return grid_norm_flat

    def forward(self, pixel_features, superpixel_labels, initial_gs_features, initial_gs_geom):
        B, C, H_img, W_img = pixel_features.shape
        K = initial_gs_features.shape[1]
        N = H_img * W_img
        device = pixel_features.device

        q_in = self.norm_gs(initial_gs_features)
        pixels_flat = pixel_features.flatten(2).transpose(1, 2) # [B, N, C]
        gs_centers_norm = initial_gs_geom[:, :, :2].clone() 
        K_safe = gs_centers_norm.shape[1] # K_safe == K

        if self.pixel_grid_norm is None or self.pixel_grid_norm.shape[1] != N or self.pixel_grid_norm.device != device or self.pixel_grid_norm.shape[0] != B:
            self.pixel_grid_norm = self._create_pixel_grid(B, H_img, W_img, device)
        pixel_coords_norm = self.pixel_grid_norm

        labels_flat = superpixel_labels.flatten(1) # [B, N]
        labels_flat_expanded = labels_flat.unsqueeze(-1).expand(-1, -1, 2)
        
        # centers_for_pixels[b, i, :] = gs_centers_norm[b, labels_flat[b, i], :]
        centers_for_pixels_norm = torch.gather(gs_centers_norm, 1, labels_flat_expanded) # [B, N, 2]
        rel_coords = pixel_coords_norm - centers_for_pixels_norm # [B, N, 2]
        pos_encoding = self.pixel_pos_embed_layer(rel_coords) # [B, N, C]

        pixels_flat_norm = self.norm_pixels(pixels_flat)
        k_in = pixels_flat_norm + pos_encoding 
        v_in = pixels_flat_norm

        labels_k = labels_flat.unsqueeze(1) # [B, 1, N]
        labels_q = torch.arange(K, device=q_in.device, dtype=labels_k.dtype).view(1, K, 1)
        labels_q = labels_q.expand(B, -1, -1) # [B, K, 1]
        
        valid_pixel_mask = (labels_k >= 0) & (labels_k < K_safe) # [B, 1, N]
        attn_mask_bool = (labels_q != labels_k) | (~valid_pixel_mask) # [B, K, N]
        
        attn_bias = torch.zeros(B, self.heads, K, N, device=q_in.device, dtype=q_in.dtype)
        attn_bias.masked_fill_(attn_mask_bool.unsqueeze(1), -torch.inf)
        
        # q: [B, K, C], k: [B, N, C], v: [B, N, C]
        q = self.to_q(q_in)
        k = self.to_k(k_in)
        v = self.to_v(v_in)

        # Reshape: [B, Seq, H, D_head]
        q = rearrange(q, 'b k (h d) -> b k h d', h=self.heads) # [B, K, H, D_head]
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.heads) # [B, N, H, D_head]
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.heads) # [B, N, H, D_head]

        # q: [B, K, H, D], k: [B, N, H, D], v: [B, N, H, D]
        # attn_bias: [B, 1, K, N]
        # out: [B, K, H, D]
        out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=self.scale)
        # Reshape back: [B, K, C]
        aggregated_features = rearrange(out, 'b k h d -> b k (h d)')

        aggregated_features = torch.nan_to_num(
            aggregated_features, nan=0.0, posinf=0.0, neginf=0.0
        )
        
        out = initial_gs_features + aggregated_features
        out = out + self.ffn(out)
        return out
    


class InterGaussianAttention(nn.Module):
    def __init__(self, dim, heads, qk_dim, mlp_ratio=4):
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.gs_embed_layer = nn.Sequential(nn.Linear(5, self.dim), nn.LayerNorm(self.dim), nn.GELU(), nn.Linear(self.dim, self.dim))

        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = MaskAttention(dim=self.dim, heads=self.heads, qk_dim=qk_dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.ffn1d = nn.Sequential(nn.Linear(self.dim, self.dim * mlp_ratio), nn.GELU(), nn.Linear(self.dim * mlp_ratio, self.dim))

        # to gs parameters
        self.to_gs = nn.ModuleList([
            nn.Linear(dim, 5), # [x,y,sigma_x,sigma_y,rho]
            nn.Linear(dim, dim), # features
            ])
        self.to_gs[0].weight.data.fill_(0.0)
        self.to_gs[0].bias.data.fill_(0.0)

        self.feat_aggregator = SuperpixelAttentionAggregator(dim=dim, heads=4)
        self.norm_feat = nn.LayerNorm(self.dim)
       
        
        
    def forward(self, gs_params, empty_mask, pixel_features, superpixel_labels):
        B, K, _ = gs_params.shape
        
        struct_gs = gs_params[:,:,:5]
        init_gs_codes = self.gs_embed_layer(struct_gs) #[x,y,sigma_x,sigma_y,rho]
        gs_embed = init_gs_codes # [B, K, C]

        feat_embed = gs_params[:,:,5:] # [B, K, C]
        assert feat_embed.shape[-1] == self.dim
        x = gs_embed + self.norm_feat(feat_embed)

        x_feat = self.feat_aggregator(pixel_features=pixel_features, superpixel_labels=superpixel_labels, initial_gs_features=x, initial_gs_geom=struct_gs)
        x = x_feat + gs_embed

        mask = empty_mask.squeeze(-1) # [B, K]
        H = self.heads
        
        attn_bias = torch.zeros(B, H, K, K, device=x.device, dtype=x.dtype)
        mask_to_fill = mask.unsqueeze(1).unsqueeze(1)
        attn_bias.masked_fill_(mask_to_fill, -torch.inf)

        x_attn = self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + x_attn
        x_ffn = self.ffn1d(self.norm2(x))
        x = x + x_ffn        

        gs_delta = self.to_gs[0](x)
        contents = self.to_gs[-1](x)
        gs_delta[:,:,:2] = F.tanh(gs_delta[:,:,:2]) * (1/np.sqrt(K))
        gs_delta[:,:,2:4] = torch.clamp(F.tanh(gs_delta[:,:,2:4]), min= -0.5) * gs_params[:,:,2:4]
        gs_delta[:,:,4:5] = F.tanh(gs_delta[:,:,4:5]) * gs_params[:,:,4:5]
        upd_gs_params_1 = gs_delta[:,:,:5] + gs_params[:,:,:5]
        upd_gs_params = torch.cat([upd_gs_params_1, contents], dim=-1)
        return upd_gs_params



class IntraAttention(nn.Module):
    def __init__(self, dim, heads, qk_dim, num_segments, mlp_ratio=2):
        super().__init__()
        self.heads = heads
        self.qk_dim = qk_dim
        self.head_dim_q = qk_dim // heads
        self.head_dim_v = dim // heads
        self.scale = self.head_dim_q ** -0.5
        self.num_segments = num_segments # K

        self.norm = LayerNorm2d(dim)
        self.to_q = nn.Conv2d(dim, qk_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, qk_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias=False) 

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, qk_dim),
            nn.LayerNorm(qk_dim),
            nn.GELU(),
            nn.Linear(qk_dim, qk_dim)
        )
        self.register_buffer('pixel_grid_norm', None, persistent=False)

        self.ffn_2d = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim * mlp_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim * mlp_ratio, dim, 1)
        )

    def _create_pixel_grid(self, B, H, W, device):
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        xx_norm = (xx + 0.5) / max(W, 1)
        yy_norm = (yy + 0.5) / max(H, 1)
        grid_norm = torch.stack([xx_norm, yy_norm], dim=-1) # [H, W, 2]
        return grid_norm.unsqueeze(0).expand(B, -1, -1, -1) # [B, H, W, 2]

    def forward(self, x, labels, superpixel_centers):
        B, C, H, W = x.shape
        N = H * W
        K = self.num_segments
        device = x.device

        if self.pixel_grid_norm is None or self.pixel_grid_norm.shape[0:3] != (B, H, W):
            self.pixel_grid_norm = self._create_pixel_grid(B, H, W, device)
        pixel_coords = self.pixel_grid_norm # [B, H, W, 2]

        labels_flat = labels.view(-1) # (B*N)
        batch_offsets = torch.arange(B, device=device).repeat_interleave(N) * K
        global_indices = labels_flat + batch_offsets # (B*N)
        centers_flat = superpixel_centers.view(-1, 2) # (B*K, 2)
        
        pixel_centers = centers_flat[global_indices].view(B, H, W, 2) # [B, H, W, 2]

        rel_coords = pixel_coords - pixel_centers # [B, H, W, 2]
        
        pos_embed = self.pos_mlp(rel_coords) 
        pos_embed = rearrange(pos_embed, 'b h w c -> b c h w') # [B, qk_dim, H, W]

        x_norm = self.norm(x)
        q = self.to_q(x_norm) # [B, qk_dim, H, W]
        k = self.to_k(x_norm) # [B, qk_dim, H, W]
        v = self.to_v(x_norm) # [B, dim, H, W]

        q = q + pos_embed
        k = k + pos_embed

        q, k, v = [rearrange(t, 'b c h w -> (b h w) c') for t in (q, k, v)]
        sort_indices = torch.argsort(global_indices)
        
        q_sorted = q[sort_indices]
        k_sorted = k[sort_indices]
        v_sorted = v[sort_indices]

        counts = torch.bincount(global_indices, minlength=B * K)
        seqlens_list = counts.cpu().tolist()
        
        attn_bias = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(seqlens_list)

        # Reshape for xformers: [1, Total_Pixels, Heads, Head_Dim]
        q_sorted = rearrange(q_sorted, 'n (h d) -> 1 n h d', h=self.heads)
        k_sorted = rearrange(k_sorted, 'n (h d) -> 1 n h d', h=self.heads)
        v_sorted = rearrange(v_sorted, 'n (h d) -> 1 n h d', h=self.heads, d=self.head_dim_v)

        out_sorted = xops.memory_efficient_attention(
            q_sorted, k_sorted, v_sorted, attn_bias=attn_bias, scale=self.scale
        )
        out_sorted = rearrange(out_sorted, '1 n h d -> n (h d)')

        inv_sort_indices = torch.empty_like(sort_indices)
        inv_sort_indices[sort_indices] = torch.arange(B * N, device=device)
        
        attn_out = out_sorted[inv_sort_indices]
        attn_out = rearrange(attn_out, '(b h w) c -> b c h w', b=B, h=H, w=W) 
        
        x = attn_out + x
        x_ffn = self.ffn_2d(x)
        x = x_ffn + x
        return x


class SGBlock(nn.Module):
    def __init__(self, dim, n_segments, local_size, seq, heads, qk_dim, mlp_ratio, **kwargs):
        super(SGBlock, self).__init__()
        self.n_segments = n_segments
        inter_mlp_ratio, intra_mlp_ratio = 4, 2  
        self.slic_n_iters = 10

        self.inter_attn = InterGaussianAttention(dim, heads, qk_dim, mlp_ratio=inter_mlp_ratio)
        
        self.intra_attn = IntraAttention(dim, heads, qk_dim, n_segments, mlp_ratio=intra_mlp_ratio)
        
        self.ffn_fusion = FFN2d(dim, dim*mlp_ratio, dim)

        self.local_layer = LocalTokenAttention(dim, heads, qk_dim, local_size, mlp_ratio=mlp_ratio)

    def forward(self, x, c_weight):
        B, C, H, W = x.shape
        with torch.no_grad():
            x_reduced = top_k_channel(x, c_weight, k_channels=8)
            labels, centers_color_fake, centers_xy, cov_matrix, empty_mask = slic_batch_parallel(
                x_reduced,
                n_segments=self.n_segments,
                n_iters=self.slic_n_iters,
                slico = True, hex_grid=True, tile=16
            )
            centers_color = get_superpixel_centers(x, labels, empty_mask)
            

            var_x = cov_matrix[:, :, 0, 0]  # σ_x^2
            var_y = cov_matrix[:, :, 1, 1]  # σ_y^2
            cov_xy = cov_matrix[:, :, 0, 1]  # cov(x, y)
            sigma_x = torch.sqrt(var_x.clamp(min=0.0) + 1e-6)
            sigma_y = torch.sqrt(var_y.clamp(min=0.0) + 1e-6)
            rho = cov_xy / (sigma_x * sigma_y + 1e-6)  # 避免除以0
            rho = torch.clamp(rho, min=-1.0, max=1.0)
            
            gs_params = torch.cat([
                centers_xy, 
                sigma_x.unsqueeze(-1), 
                sigma_y.unsqueeze(-1), 
                rho.unsqueeze(-1), 
                centers_color
            ], dim=2)

            B, labels_num, all_dim = gs_params.shape
            if labels_num < self.n_segments:
                diff = self.n_segments - labels_num
                pad_gs = torch.zeros(B, diff, all_dim, device=gs_params.device, dtype=gs_params.dtype)
                pad_mask = torch.ones(B, diff, 1, device=empty_mask.device, dtype=empty_mask.dtype)
                gs_params = torch.cat([gs_params, pad_gs], dim=1)
                empty_mask = torch.cat([empty_mask, pad_mask], dim=1)


        upd_gs_params = self.inter_attn(gs_params, empty_mask, pixel_features=x, superpixel_labels=labels)
        upd_gs_params[:,:,5:] = upd_gs_params[:,:,5:].masked_fill(empty_mask, 0.0) # shape [8, 256, 1] 会自动广播到 [8, 256, 128]
        if upd_gs_params.shape[-1] != 101: 
            x_inter = render_gaussians(upd_gs_params, resolution=(H, W), tile_size=2, max_gauss_chunk=8)
        else: 
            struct_channels = upd_gs_params[:, :, :5]
            feat_channels_1 = upd_gs_params[:, :, 5:5+64]
            feat_channels_2 = upd_gs_params[:, :, 101-32:]
            x_inter_1 = render_gaussians(torch.cat((struct_channels, feat_channels_1), dim=-1), resolution=(H, W), tile_size=2, max_gauss_chunk=8)
            x_inter_2 = render_gaussians(torch.cat((struct_channels, feat_channels_2), dim=-1), resolution=(H, W), tile_size=2, max_gauss_chunk=8)
            x_inter = torch.cat((x_inter_1, x_inter_2), dim=1)

        x_intra = self.intra_attn(x, labels, gs_params[:,:,:2])

        x_fused = x_inter + x_intra 
        x = x + x_fused 
        x = x + self.ffn_fusion(x) # FFN
    
        x = self.local_layer(x)   
        return x, labels, gs_params, upd_gs_params



class SG(nn.Module):
    def __init__(self, colors=1, dim=[128,96,64], block_num=[4,2,1], heads=[4,3,2], qk_dim=[128,96,64], mlp_ratio=[2,2,2], upscale=4, flow_dim=[64,48,32],
                 local_size=[[6,8,8,6],[6,6],[6]], n_segments=[[256,256,256,256],[512,512],[1024]]):
        super(SG, self).__init__()
        self.block_num = block_num
        self.upscale = upscale

        aux_dim, main_dim = [x*1 for x in dim], [x*1 for x in dim]
        aux_dim, main_dim = [int(x) for x in aux_dim], [int(x) for x in main_dim]

        self.downsampler = nn.ModuleList()
        self.downsampler.append(nn.Sequential(nn.Conv2d(colors, 64, kernel_size=3, stride=1, padding=1), nn.GELU(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)))
        self.downsampler.append(nn.Sequential(nn.Conv2d(64, 96//4, kernel_size=3, stride=1, padding=1), nn.PixelUnshuffle(2), nn.GELU(), nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)))
        self.downsampler.append(nn.Sequential(nn.Conv2d(96, 128//4, kernel_size=3, stride=1, padding=1), nn.PixelUnshuffle(2), nn.GELU(), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)))


        self.flow_cnn = nn.ModuleList()
        for i in range(0, len(block_num)): 
            if i==0:
                num_channels=[flow_dim[i], flow_dim[i]]
            elif i==1:
                num_channels=[flow_dim[i], flow_dim[i]]
            elif i==2:
                num_channels=[flow_dim[i], flow_dim[i]]
            if i==0:
                self.flow_cnn.append(UNet(
                    in_channels=dim[i]*2,
                    out_channels=2,
                    num_groups=16,
                    channel_dims=num_channels
                ))
            else:
                self.flow_cnn.append(UNet(
                    in_channels=dim[i]*2+2,
                    out_channels=2,
                    num_groups=16,
                    channel_dims=num_channels
                ))       

        self.first_conv = nn.Sequential(nn.Conv2d(colors, main_dim[0], 3, 1, 1), nn.GELU(), nn.Conv2d(main_dim[0], main_dim[0], 3, 1, 1))
        
        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        self.early_fusion = nn.ModuleList()
        self.feat_up = nn.ModuleList()
        self.last_conv = nn.ModuleList()
        for i in range(0, len(block_num)): 
            self.blocks.append(nn.ModuleList())
            self.mid_convs.append(nn.ModuleList())
            self.early_fusion.append(ModalityFusion(in_out_channels=dim[i]))
            for j in range(block_num[i]):
                self.blocks[i].append(SGBlock(
                    dim=dim[i], 
                    n_segments=n_segments[i][j],
                    local_size=[local_size[i][j], local_size[i][j]],
                    heads=heads[i], 
                    qk_dim=qk_dim[i], 
                    mlp_ratio=mlp_ratio[i],
                    seq=j
                ))
                self.mid_convs[i].append(MidFusion(in_out_channels=dim[i]))
            if i < len(block_num)-1: self.feat_up.append(FeatUp(in_channels=dim[i], out_channels=dim[i+1]))
        self.last_conv.append(nn.Conv2d(dim[i], colors, 3, 1, 1))

        if upscale == 4:
            self.upconv = nn.ModuleList()
            if len(block_num) == 1: self.upconv.append(nn.Conv2d(dim[0], dim[0] * 4, 3, 1, 1, bias=True))
            if len(block_num) == 1: self.upconv.append(nn.Conv2d(dim[0], dim[0] * 4, 3, 1, 1, bias=True))
            if len(block_num) == 2: self.upconv.append(nn.Conv2d(dim[1], dim[1] * 4, 3, 1, 1, bias=True))
            self.pixel_shuffle = nn.PixelShuffle(2)
        else:
            raise NotImplementedError('Upscale factor is expected to be one of (4), but got {}'.format(upscale))
        
        self.upsample_act = nn.GELU() 
        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        print('[SG Model] #Params : {:<.4f} [K]'.format(num_parameters / 10 ** 3))


    def forward(self, x, aux):
        self.B, _, self.H_lr, self.W_lr = x.shape
        self.device = x.device
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        x = self.first_conv(x)

        AUX_FEAT, X_FEAT = [], []
        for i in range(len(self.downsampler)):
            aux = self.downsampler[i](aux)
            AUX_FEAT.append(aux)
        AUX_FEAT.reverse()

        for i in range(len(self.block_num)):
            if i == 0: previous_field = None
            AUX_FEAT, previous_field = self.flow_correct(i, AUX_FEAT, x, previous_field)
            x, c_weight = self.early_fusion[i](x, AUX_FEAT[i])
            
            for j in range(self.block_num[i]):
                residual, labels, gs_params, upd_gs_params = self.blocks[i][j](x, c_weight)
                residual, c_weight = self.mid_convs[i][j](residual)
                x = x + residual
            
            X_FEAT.append(x)
            if i < len(self.block_num)-1: x = self.feat_up[i](x)

        if self.upscale == 4:
            OUT = []
            if len(self.block_num) == 1:
                out = self.pixel_shuffle(self.upconv[0](X_FEAT[0])) 
                out = self.pixel_shuffle(self.upconv[1](out))
                out1 = base + self.last_conv[0](out)
                OUT.append(out1)

            if len(self.block_num) == 2:
                out = self.pixel_shuffle(self.upconv[0](X_FEAT[1]))
                out2 = base + self.last_conv[0](out)
                OUT.append(out2)

            if len(self.block_num) == 3:
                out = X_FEAT[2]
                out3 = out2 + self.last_conv[2](out)
                OUT.append(out3)
        else:
            raise NotImplementedError('Upscale factor is expected to be one of (4), but got {}'.format(self.upscale))
        return OUT
    



    def flow_correct(self, stage, AUX_FEAT, x_feat, previous_field):
        if stage == 0:
            flow_cnn_input = torch.cat([x_feat, AUX_FEAT[stage]], dim=1)
            displacement_field = self.flow_cnn[stage](flow_cnn_input) 
        else:
            flow_cnn_input = torch.cat([x_feat, AUX_FEAT[stage], previous_field], dim=1)
            displacement_field = self.flow_cnn[stage](flow_cnn_input) + previous_field


        previous_field = F.interpolate(displacement_field, scale_factor=2, mode='bilinear')
        _, _, H_lr, W_lr = displacement_field.shape
        scale_factor_lr = torch.tensor([2.0 / W_lr, 2.0 / H_lr], device=self.device).view(1, 2, 1, 1)

        normalized_displacement_field = displacement_field * scale_factor_lr
        theta = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.B, 1, 1)
        _, _, H, W = AUX_FEAT[stage].shape
        # [B, 2, H_lr, W_lr] -> [B, 2, H, W]
        assert normalized_displacement_field.shape[-1] == W

        normalized_displacement_field = normalized_displacement_field.permute(0, 2, 3, 1)
        base_grid = F.affine_grid(theta, (self.B, 1, H, W), align_corners=False)
        new_grid = base_grid + normalized_displacement_field
        aligned_aux_feat = F.grid_sample(
            AUX_FEAT[stage], 
            new_grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=False
        )
        AUX_FEAT[stage] = aligned_aux_feat
        return AUX_FEAT, previous_field