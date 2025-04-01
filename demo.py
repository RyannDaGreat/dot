import torch
from torch import nn
import os.path as osp
from tqdm import tqdm
from matplotlib import colormaps
import numpy as np
import scipy
import math

from dot.utils.options.demo_options import DemoOptions
from dot.models import create_model
from dot.utils.io import create_folder, write_video, read_video, read_frame
from dot.utils.torch import to_device, get_grid


class Visualizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.save_mode = args.save_mode
        self.result_path = args.result_path
        self.overlay_factor = args.overlay_factor
        self.spaghetti_radius = args.spaghetti_radius
        self.spaghetti_length = args.spaghetti_length
        self.spaghetti_grid = args.spaghetti_grid
        self.spaghetti_scale = args.spaghetti_scale
        self.spaghetti_every = args.spaghetti_every
        self.spaghetti_dropout = args.spaghetti_dropout

    def forward(self, data, mode):
        if "overlay" in mode:
            video = self.plot_overlay(data, mode)
        elif "spaghetti" in mode:
            video = self.plot_spaghetti(data, mode)
        elif "warp" in mode:
            video = self.plot_warp(data, mode)
        else:
            raise ValueError(f"Unknown mode {mode}")
        save_path = osp.join(self.result_path, mode) + ".mp4" if self.save_mode == "video" else ""
        write_video(video, save_path)

    def plot_overlay(self, data, mode):
        T, C, H, W = data["video"].shape
        mask = data["mask"] if "mask" in mode else torch.ones_like(data["mask"])
        tracks = data["tracks"]

        if tracks.ndim == 4:
            col = get_rainbow_colors(int(mask.sum())).cuda()
        else:
            col = get_rainbow_colors(tracks.size(1)).cuda()

        video = []
        for tgt_step in tqdm(range(T), leave=False, desc="Plot target frame"):
            tgt_frame = data["video"][tgt_step]
            tgt_frame = tgt_frame.permute(1, 2, 0)

            # Plot rainbow points
            tgt_pos = tracks[tgt_step, ..., :2]
            tgt_vis = tracks[tgt_step, ..., 2]
            if tracks.ndim == 4:
                tgt_pos = tgt_pos[mask]
                tgt_vis = tgt_vis[mask]
            rainbow, alpha = draw(tgt_pos, tgt_vis, col, H, W)

            # Plot rainbow points with white stripes in occluded regions
            if "stripes" in mode:
                rainbow_occ, alpha_occ = draw(tgt_pos, 1 - tgt_vis, col, H, W)
                stripes = torch.arange(H).view(-1, 1) + torch.arange(W).view(1, -1)
                stripes = stripes % 9 < 3
                rainbow_occ[stripes] = 1.
                rainbow = alpha * rainbow + (1 - alpha) * rainbow_occ
                alpha = alpha + (1 - alpha) * alpha_occ

            # Overlay rainbow points over target frame
            tgt_frame = self.overlay_factor * alpha * rainbow + (1 - self.overlay_factor * alpha) * tgt_frame

            # Convert from H W C to C H W
            tgt_frame = tgt_frame.permute(2, 0, 1)
            video.append(tgt_frame)
        video = torch.stack(video)
        return video

    def plot_warp(self, data, mode):
        """Warp the first frame to match subsequent frames using tracking data."""
        T, C, H, W = data["video"].shape
        tracks = data["tracks"]
        
        # Get the first frame that we'll warp
        first_frame = data["video"][0].permute(1, 2, 0)  # H, W, C
        
        # Create a regular grid for sampling
        grid = get_grid(H, W).cuda()
        
        video = []
        for tgt_step in tqdm(range(T), leave=False, desc="Warping first frame"):
            # Skip first frame (just display it as is)
            if tgt_step == 0:
                video.append(data["video"][0])
                continue
                
            # Get tracks for current frame
            tgt_tracks = tracks[tgt_step]
            
            if tracks.ndim == 4:  # Full dense tracks for every pixel (T, H, W, 3)
                # Create backward flow grid: where does each pixel in current frame come from in first frame?
                flow_grid = torch.zeros((H, W, 2), device=tracks.device)
                
                # Extract positions and visibility
                src_points = tracks[0, ..., :2]    # positions in first frame
                tgt_points = tgt_tracks[..., :2]   # positions in current frame
                tgt_vis = tgt_tracks[..., 2] > 0.5 # visibility in current frame
                
                # Create a grid that maps from target back to source
                # For each position in the target frame, find where it comes from in the source frame
                for h in range(H):
                    for w in range(W):
                        if tgt_vis[h, w]:
                            # Normalize coordinates to [-1, 1] for grid_sample
                            src_x = 2.0 * src_points[h, w, 0] - 1.0
                            src_y = 2.0 * src_points[h, w, 1] - 1.0
                            flow_grid[h, w, 0] = src_x
                            flow_grid[h, w, 1] = src_y
                
                # Create a mask for valid pixels
                valid_mask = tgt_vis.float()[..., None].expand(-1, -1, 2)
                
                # Use nearest neighbor filling for invisible pixels
                # This is a simple approach - could be improved with more sophisticated inpainting
                from scipy.ndimage import distance_transform_edt
                
                # Convert to numpy for processing
                valid_mask_np = valid_mask[..., 0].cpu().numpy()
                flow_grid_np = flow_grid.cpu().numpy()
                
                # Find distance to nearest valid pixel
                dist_transform = distance_transform_edt(1 - valid_mask_np)
                
                # For each invalid pixel, find nearest valid pixel
                for h in range(H):
                    for w in range(W):
                        if not valid_mask_np[h, w]:
                            # Find nearest valid pixel
                            y, x = np.unravel_index(np.argmin(dist_transform + valid_mask_np * H * W), (H, W))
                            flow_grid_np[h, w] = flow_grid_np[y, x]
                
                # Convert back to tensor
                flow_grid = torch.from_numpy(flow_grid_np).to(tracks.device)
                
            else:  # Sparse tracks (T, N, 3)
                # For sparse tracks, we'll interpolate to get dense flow
                src_points = tracks[0, :, :2]  # N, 2
                tgt_points = tgt_tracks[:, :2]  # N, 2
                tgt_vis = tgt_tracks[:, 2] > 0.5  # N
                
                # Only use visible points
                src_points = src_points[tgt_vis]
                tgt_points = tgt_points[tgt_vis]
                
                # Convert to pixel coordinates
                src_points_px = src_points.clone()
                src_points_px[:, 0] = src_points_px[:, 0] * (W - 1)
                src_points_px[:, 1] = src_points_px[:, 1] * (H - 1)
                
                tgt_points_px = tgt_points.clone()
                tgt_points_px[:, 0] = tgt_points_px[:, 0] * (W - 1)
                tgt_points_px[:, 1] = tgt_points_px[:, 1] * (H - 1)
                
                # Create a regular grid
                y_coords, x_coords = torch.meshgrid(torch.arange(H, device=tracks.device), 
                                                   torch.arange(W, device=tracks.device))
                tgt_coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1).float()  # HW, 2
                
                # For each target position, find K nearest source points
                K = min(5, len(src_points_px))
                flow_grid = torch.zeros((H*W, 2), device=tracks.device)
                
                for i in range(H*W):
                    # Calculate distances to all tracked points
                    dists = torch.sum((tgt_coords[i].unsqueeze(0) - tgt_points_px) ** 2, dim=1)
                    
                    # Get K nearest neighbors
                    k_indices = torch.topk(dists, k=K, largest=False).indices
                    
                    # Get weights based on distance (inverse distance weighting)
                    weights = 1.0 / (dists[k_indices] + 1e-6)
                    weights = weights / weights.sum()
                    
                    # Weighted average of source positions
                    src_pos = torch.sum(weights.unsqueeze(1) * src_points[k_indices], dim=0)
                    
                    # Convert to grid coordinates [-1, 1]
                    flow_grid[i, 0] = 2.0 * src_pos[0] - 1.0
                    flow_grid[i, 1] = 2.0 * src_pos[1] - 1.0
                
                flow_grid = flow_grid.reshape(H, W, 2)
            
            # Perform the warping using grid_sample
            first_frame_tensor = first_frame.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
            warped_frame = torch.nn.functional.grid_sample(
                first_frame_tensor, 
                flow_grid.unsqueeze(0),  # Add batch dimension
                mode='bilinear',
                align_corners=True
            )
            
            # Get the current frame to blend with for occlusions
            current_frame = data["video"][tgt_step].unsqueeze(0)
            
            # For areas where the track is not visible (occluded), use the current frame
            if tracks.ndim == 4:
                blend_mask = tgt_tracks[..., 2].unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                warped_frame = blend_mask * warped_frame + (1 - blend_mask) * current_frame
            
            video.append(warped_frame[0])  # Remove batch dimension
        
        video = torch.stack(video)
        return video

    def plot_spaghetti(self, data, mode):
        bg_color = 1.
        T, C, H, W = data["video"].shape
        G, S, R, L = self.spaghetti_grid, self.spaghetti_scale, self.spaghetti_radius, self.spaghetti_length
        D = self.spaghetti_dropout

        # Extract a grid of tracks
        mask = data["mask"] if "mask" in mode else torch.ones_like(data["mask"])
        mask = mask[G // 2:-G // 2 + 1:G, G // 2:-G // 2 + 1:G]
        tracks = data["tracks"]
        if tracks.ndim == 4:
            tracks = tracks[:, G // 2:-G // 2 + 1:G, G // 2:-G // 2 + 1:G]
            tracks = tracks[:, mask]
        elif D > 0:
            N = tracks.size(1)
            assert D < 1
            samples = np.sort(np.random.choice(N, int((1 - D) * N), replace=False))
            tracks = tracks[:, samples]
        col = get_rainbow_colors(tracks.size(1)).cuda()

        # Densify tracks over temporal axis
        tracks = spline_interpolation(tracks, length=L)

        video = []
        cur_frame = None
        cur_alpha = None
        grid = get_grid(H, W).cuda()
        grid[..., 0] *= (W - 1)
        grid[..., 1] *= (H - 1)
        for tgt_step in tqdm(range(T), leave=False, desc="Plot target frame"):
            for delta in range(L):
                # Plot rainbow points
                tgt_pos = tracks[tgt_step * L + delta, :, :2]
                tgt_vis = torch.ones_like(tgt_pos[..., 0])
                tgt_pos = project(tgt_pos, tgt_step * L + delta, T * L, H, W)
                tgt_col = col.clone()
                rainbow, alpha = draw(S * tgt_pos, tgt_vis, tgt_col, int(S * H), int(S * W), radius=R)
                rainbow, alpha = rainbow.cpu(), alpha.cpu()

                # Overlay rainbow points over previous points / frames
                if cur_frame is None:
                    cur_frame = rainbow
                    cur_alpha = alpha
                else:
                    cur_frame = alpha * rainbow + (1 - alpha) * cur_frame
                    cur_alpha = 1 - (1 - cur_alpha) * (1 - alpha)

                plot_first = "first" in mode and tgt_step == 0 and delta == 0
                plot_last = "last" in mode and delta == 0
                plot_every = "every" in mode and delta == 0 and tgt_step % self.spaghetti_every == 0
                if delta == 0:
                    if plot_first or plot_last or plot_every:
                        # Plot target frame
                        tgt_col = data["video"][tgt_step].permute(1, 2, 0).reshape(-1, 3)
                        tgt_pos = grid.view(-1, 2)
                        tgt_vis = torch.ones_like(tgt_pos[..., 0])
                        tgt_pos = project(tgt_pos, tgt_step * L + delta, T * L, H, W)
                        tgt_frame, alpha_frame = draw(S * tgt_pos, tgt_vis, tgt_col, int(S * H), int(S * W))
                        tgt_frame, alpha_frame = tgt_frame.cpu(), alpha_frame.cpu()

                        # Overlay target frame over previous points / frames
                        tgt_frame = alpha_frame * tgt_frame + (1 - alpha_frame) * cur_frame
                        alpha_frame = 1 - (1 - cur_alpha) * (1 - alpha_frame)

                        # Add last points on top
                        tgt_frame = alpha * rainbow + (1 - alpha) * tgt_frame
                        alpha_frame = 1 - (1 - alpha_frame) * (1 - alpha)

                        # Set background color
                        tgt_frame = alpha_frame * tgt_frame + (1 - alpha_frame) * torch.ones_like(tgt_frame) * bg_color

                        if plot_first or plot_every:
                            cur_frame = tgt_frame
                            cur_alpha = alpha_frame
                    else:
                        tgt_frame = cur_alpha * cur_frame + (1 - cur_alpha) * torch.ones_like(cur_frame) * bg_color

                    # Convert from H W C to C H W
                    tgt_frame = tgt_frame.permute(2, 0, 1)

                    # Translate everything to make the target frame look static
                    if "static" in mode:
                        end_pos = project(torch.tensor([[0, 0]]), T * L, T * L, H, W)[0]
                        cur_pos = project(torch.tensor([[0, 0]]), tgt_step * L + delta, T * L, H, W)[0]
                        delta_pos = S * (end_pos - cur_pos)
                        tgt_frame = translation(tgt_frame, delta_pos[0], delta_pos[1], bg_color)
                    video.append(tgt_frame)
        video = torch.stack(video)
        return video


def translation(frame, dx, dy, pad_value):
    C, H, W = frame.shape
    grid = get_grid(H, W, device=frame.device)
    grid[..., 0] = grid[..., 0] - (dx / (W - 1))
    grid[..., 1] = grid[..., 1] - (dy / (H - 1))
    frame = frame - pad_value
    frame = torch.nn.functional.grid_sample(frame[None], grid[None] * 2 - 1, mode='bilinear', align_corners=True)[0]
    frame = frame + pad_value
    return frame


def spline_interpolation(x, length=10):
    if length != 1:
        T, N, C = x.shape
        x = x.view(T, -1).cpu().numpy()
        original_time = np.arange(T)
        cs = scipy.interpolate.CubicSpline(original_time, x)
        new_time = np.linspace(original_time[0], original_time[-1], T * length)
        x = torch.from_numpy(cs(new_time)).view(-1, N, C).float().cuda()
    return x


def get_rainbow_colors(size):
    col_map = colormaps["jet"]
    col_range = np.array(range(size)) / (size - 1)
    col = torch.from_numpy(col_map(col_range)[..., :3]).float()
    col = col.view(-1, 3)
    return col


def draw(pos, vis, col, height, width, radius=1):
    H, W = height, width
    frame = torch.zeros(H * W, 4, device=pos.device)
    pos = pos[vis.bool()]
    col = col[vis.bool()]
    if radius > 1:
        pos, col = get_radius_neighbors(pos, col, radius)
    else:
        pos, col = get_cardinal_neighbors(pos, col)
    inbound = (pos[:, 0] >= 0) & (pos[:, 0] <= W - 1) & (pos[:, 1] >= 0) & (pos[:, 1] <= H - 1)
    pos = pos[inbound]
    col = col[inbound]
    pos = pos.round().long()
    idx = pos[:, 1] * W + pos[:, 0]
    idx = idx.view(-1, 1).expand(-1, 4)
    frame.scatter_add_(0, idx, col)
    frame = frame.view(H, W, 4)
    frame, alpha = frame[..., :3], frame[..., 3]
    nonzero = alpha > 0
    frame[nonzero] /= alpha[nonzero][..., None]
    alpha = nonzero[..., None].float()
    return frame, alpha


def get_cardinal_neighbors(pos, col, eps=0.01):
    pos_nw = torch.stack([pos[:, 0].floor(), pos[:, 1].floor()], dim=-1)
    pos_sw = torch.stack([pos[:, 0].floor(), pos[:, 1].floor() + 1], dim=-1)
    pos_ne = torch.stack([pos[:, 0].floor() + 1, pos[:, 1].floor()], dim=-1)
    pos_se = torch.stack([pos[:, 0].floor() + 1, pos[:, 1].floor() + 1], dim=-1)
    w_n = pos[:, 1].floor() + 1 - pos[:, 1] + eps
    w_s = pos[:, 1] - pos[:, 1].floor() + eps
    w_w = pos[:, 0].floor() + 1 - pos[:, 0] + eps
    w_e = pos[:, 0] - pos[:, 0].floor() + eps
    w_nw = (w_n * w_w)[:, None]
    w_sw = (w_s * w_w)[:, None]
    w_ne = (w_n * w_e)[:, None]
    w_se = (w_s * w_e)[:, None]
    col_nw = torch.cat([w_nw * col, w_nw], dim=-1)
    col_sw = torch.cat([w_sw * col, w_sw], dim=-1)
    col_ne = torch.cat([w_ne * col, w_ne], dim=-1)
    col_se = torch.cat([w_se * col, w_se], dim=-1)
    pos = torch.cat([pos_nw, pos_sw, pos_ne, pos_se], dim=0)
    col = torch.cat([col_nw, col_sw, col_ne, col_se], dim=0)
    return pos, col


def get_radius_neighbors(pos, col, radius):
    R = math.ceil(radius)
    center = torch.stack([pos[:, 0].round(), pos[:, 1].round()], dim=-1)
    nn = torch.arange(-R, R + 1)
    nn = torch.stack([nn[None, :].expand(2 * R + 1, -1), nn[:, None].expand(-1, 2 * R + 1)], dim=-1)
    nn = nn.view(-1, 2).cuda()
    in_radius = nn[:, 0] ** 2 + nn[:, 1] ** 2 <= radius ** 2
    nn = nn[in_radius]
    w = 1 - nn.pow(2).sum(-1).sqrt() / radius + 0.01
    w = w[None].expand(pos.size(0), -1).reshape(-1)
    pos = (center.view(-1, 1, 2) + nn.view(1, -1, 2)).view(-1, 2)
    col = col.view(-1, 1, 3).repeat(1, nn.size(0), 1)
    col = col.view(-1, 3)
    col = torch.cat([col * w[:, None], w[:, None]], dim=-1)
    return pos, col


def project(pos, t, time_steps, heigh, width):
    T, H, W = time_steps, heigh, width
    pos = torch.stack([pos[..., 0] / (W - 1), pos[..., 1] / (H - 1)], dim=-1)
    pos = pos - 0.5
    pos = pos * 0.25
    t = 1 - torch.ones_like(pos[..., :1]) * t / (T - 1)
    pos = torch.cat([pos, t], dim=-1)
    M = torch.tensor([
        [0.8, 0, 0.5],
        [-0.2, 1.0, 0.1],
        [0.0, 0.0, 0.0]
    ])
    pos = pos @ M.t().to(pos.device)
    pos = pos[..., :2]
    pos[..., 0] += 0.25
    pos[..., 1] += 0.45
    pos[..., 0] *= (W - 1)
    pos[..., 1] *= (H - 1)
    return pos


def main(args):
    model = create_model(args).cuda()
    visualizer = Visualizer(args).cuda()
    resolution = (args.height, args.width)

    tracks_path = osp.join(args.result_path, "tracks.pth")
    create_folder(args.result_path)

    video = read_video(osp.join(args.data_root, args.video_path), resolution=resolution).cuda() # , time_steps=20

    if not osp.exists(tracks_path) or args.recompute_tracks:
        with torch.no_grad():
            pred = model({"video": video[None]}, mode=args.inference_mode, **vars(args))
        tracks = pred["tracks"][0]
        if args.save_tracks:
            torch.save(tracks.cpu(), tracks_path)
    else:
        tracks = torch.load(tracks_path)

    mask_path = osp.join(args.data_root, args.mask_path)
    if any(["mask" in mode] for mode in args.visualization_modes) and osp.exists(mask_path):
        mask = read_frame(mask_path, resolution=resolution)[0] > 0.5
    else:
        mask = torch.ones(args.height, args.width).bool()

    data = {
        "video": video,
        "tracks": tracks,
        "mask": mask
    }

    data = to_device(data, "cuda")

    if data["tracks"].ndim == 4 and args.rainbow_mode == "left_right":
        data["mask"] = data["mask"].permute(1, 0)
        data["tracks"] = data["tracks"].permute(0, 2, 1, 3)
    elif data["tracks"].ndim == 3:
        points = data["tracks"][0]
        x, y = points[..., 0].long(), points[..., 1].long()
        x, y = x - x.min(), y - y.min()
        if args.rainbow_mode == "left_right":
            idx = y + x * y.max()
        else:
            idx = x + y * x.max()
        order = idx.argsort(dim=0)
        data["tracks"] = data["tracks"][:, order]

    for mode in args.visualization_modes:
        visualizer(data, mode=mode)


if __name__ == "__main__":
    args = DemoOptions().parse_args()
    main(args)
    print("Done.")
