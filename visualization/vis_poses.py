
from itertools import combinations, product
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2, torch
import numpy as np
from kornia import create_meshgrid
import plotly.graph_objects as go
import json


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def get_ray_directions_blender(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]+0.5
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal, -(j - cent[1]) / focal, -torch.ones_like(i)],
                             -1)  # (H, W, 3)

    return directions

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:,:,0] += np.arange(w)
    flow_new[:,:,1] += np.arange(h)[:,np.newaxis]

    res = cv2.remap(img, flow_new, None,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)
    return res

def get_camera_mesh(pose,depth=1):
    vertices = torch.tensor([[-0.5,-0.5,-1],
                            [0.5,-0.5,-1],
                            [0.5,0.5,-1],
                            [-0.5,0.5,-1],
                            [0,0,0]])*depth
    faces = torch.tensor([[0,1,2],
                        [0,2,3],
                        [0,1,4],
                        [1,2,4],
                        [2,3,4],
                        [3,0,4]])
    # vertices = cam2world(vertices[None],pose)
    vertices = vertices @ pose[:, :3, :3].transpose(-1, -2)
    vertices += pose[:, None, :3, 3]
    wireframe = vertices[:,[0,1,2,3,0,4,1,2,4,3]]
    return vertices,faces,wireframe

def merge_wireframes(wireframe):
    wireframe_merged = [[],[],[]]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:,0]]
        wireframe_merged[1] += [float(n) for n in w[:,1]]
        wireframe_merged[2] += [float(n) for n in w[:,2]]
    return wireframe_merged

def sobel_by_quantile(img_points: np.ndarray, q: float):
    """Return a boundary mask where 255 indicates boundaries (where gradient is
    bigger than quantile).
    """
    dx0 = np.linalg.norm(
        img_points[1:-1, 1:-1] - img_points[1:-1, :-2], axis=-1
    )
    dx1 = np.linalg.norm(
        img_points[1:-1, 1:-1] - img_points[1:-1, 2:], axis=-1
    )
    dy0 = np.linalg.norm(
        img_points[1:-1, 1:-1] - img_points[:-2, 1:-1], axis=-1
    )
    dy1 = np.linalg.norm(
        img_points[1:-1, 1:-1] - img_points[2:, 1:-1], axis=-1
    )
    dx01 = (dx0 + dx1) / 2
    dy01 = (dy0 + dy1) / 2
    dxy01 = np.linalg.norm(np.stack([dx01, dy01], axis=-1), axis=-1)

    # (H, W, 1) uint8
    boundary_mask = (dxy01 > np.quantile(dxy01, q)).astype(np.float32)
    boundary_mask = (
        np.pad(boundary_mask, ((1, 1), (1, 1)), constant_values=False)[
            ..., None
        ].astype(np.uint8)
        * 255
    )
    return  boundary_mask

def draw_line(fig, p0, p1, color, name):
    pts = np.stack([p0, p1], axis=-1)
    fig.add_trace(
        go.Scatter3d(
            x=pts[0],
            y=pts[1],
            z=pts[2],
            mode="lines",
            line={"color": color},
            name=name,
            legendgroup=name,
            showlegend=False,
        )
    )

def visualize_bbox(bbox_min, bbox_max, fig, shift):
    fig = fig or go.Figure()
    r0 = [bbox_min[0] + shift[0], bbox_max[0] + shift[0]]
    r1 = [bbox_min[1] + shift[1], bbox_max[1] + shift[1]]
    r2 = [bbox_min[2] + shift[2], bbox_max[2] + shift[2]]
    for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
        if (np.abs(s - e) != 0).sum() == 1:
            draw_line(fig, s, e, color="blue", name=f"bbox_{bbox_min}_{bbox_max}")
    return fig


folder = "data/uw2"

with open(f"{folder}/transforms_noprog.json", 'r') as f:
    transforms = json.loads(f.read())
poses = [frame["transform_matrix"] for frame in transforms["frames"]]
# poses = poses[:200]
scale = 5e1
skip = 1
num_frames = len(poses)
center_idx = num_frames//2
# center_idx = 75

layout = go.Layout(
    # xaxis={'range': [-2, 2], 'fixedrange': True, 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': -2, 'dtick': 1, 'automargin': False},
    # yaxis={'range': [-2, 2], 'fixedrange': True, 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': -2, 'dtick': 1, 'automargin': False},
    hovermode="closest",
    updatemenus=[{
        'buttons': [{'args': [None, {"frame": {"duration": 50, 
                                                                        "redraw": True},
                                                              "fromcurrent": True, 
                                                              "transition": {"duration": 0}}], 'label':'Play', 'method':'animate'}],
        "x": 0.1,
        'xanchor':'right',
        "y": 0,
        'yanchor':'top',
        'pad':{'l': 30},
        'showactive': False,
        'type': 'buttons'}],
    scene={
        'xaxis': {'range': [-10, 10], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': -2, 'dtick': 1},
        'yaxis': {'range': [-10, 10], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': -2, 'dtick': 1},
        'zaxis': {'range': [-10, 10], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': -2, 'dtick': 1},
        # 'xaxis': {},
        # 'yaxis': {},
        # 'zaxis': {},
        'aspectratio': {
            'x': 0,
            'y': 0,
            'z': 0,
        },
        'aspectmode': 'cube',},
    margin={'autoexpand': False},
    autosize=False,
    width=1000,
    height=1000)

fig = go.Figure(layout=layout)
# camera poses
allposes = torch.from_numpy(np.array(poses, dtype=np.float32)[:, :3, :]) # torch.from_numpy(np.load('poses.npy'))
allposes[..., 3] *= scale
vertices, faces, wireframe = get_camera_mesh(allposes, 0.4)
center_gt = vertices[:,-1]
wireframe_merged = merge_wireframes(wireframe)
for c in range(10, num_frames, int(4)):
    fig.add_trace(go.Scatter3d(
        x=wireframe_merged[0][c*10:(c+1)*10], 
        y=wireframe_merged[1][c*10:(c+1)*10], 
        z=wireframe_merged[2][c*10:(c+1)*10],
        mode='lines',
        line=dict(width=3, color='black'))
    )

# with open(f"{folder}/transforms_rf.json", 'r') as f:
#     transforms = json.loads(f.read())
# rf_poses = np.array([frame["transform_matrix"] for frame in transforms["frames"]])

# for rf_pose in rf_poses:
#     visualize_bbox([-1, -1, -1], [1, 1, 1], fig, rf_pose[:3, 3])




# # center camera
# allposes = torch.from_numpy(np.array(poses, dtype=np.float32)[:, :3, :]) # torch.from_numpy(np.load('poses.npy'))
# # allposes = torch.from_numpy(np.load('poses.npy'))
# vertices, faces, wireframe = get_camera_mesh(allposes, 0.4)
# center_gt = vertices[center_idx:center_idx+1,-1]
# fig.add_trace(
#     go.Scatter3d(x=center_gt[..., 0].flatten(), y=center_gt[..., 1].flatten(), z=center_gt[..., 2].flatten(),
#                                    mode='markers', marker=dict(color=[f'rgb({0}, {0}, {0})' for i in list(range(center_gt.shape[0]))],
#                        size=3))
# )
# wireframe_merged = merge_wireframes(wireframe)
# for c in [center_idx]:
#     fig.add_trace(go.Scatter3d(
#         x=wireframe_merged[0][c*10:(c+1)*10], 
#         y=wireframe_merged[1][c*10:(c+1)*10], 
#         z=wireframe_merged[2][c*10:(c+1)*10],
#         mode='lines',
#         line=dict(width=2, color='red'))
#     )

# # for c in range(center_gt.shape[0]):
# #     ax.plot(wireframe_merged[0][c*10:(c+1)*10], wireframe_merged[1][c*10:(c+1)*10], wireframe_merged[2][c*10:(c+1)*10], color='C0')


# fig = go.Figure(data=[go.Scatter3d(x=pts_3d[:, :, 0].flatten()[::10], y=pts_3d[:, :, 1].flatten()[::10], z=pts_3d[:, :, 2].flatten()[::10],
#                                    mode='markers', marker=dict(color=[f'rgb({image[i, 2]}, {image[i, 1]}, {image[i, 0]})' for i in list(range(image.shape[0]))[::10]],
#                        size=2))])
fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
fig.show()

