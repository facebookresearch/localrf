import numpy as np
import torch
from easydict import EasyDict as edict

def inverse_pose(pose):
    pose_inv = torch.zeros_like(pose)
    pose_inv[:, :3, :3] = torch.transpose(pose[:, :3, :3], 1, 2)
    pose_inv[:, :3, 3] = -torch.bmm(pose_inv[:, :3, :3].clone(), pose[:, :3, 3:])[..., 0]
    return pose_inv

def get_cam2cams(cam2worlds, indices, offset):
    idx = torch.clamp(indices + offset, 0, len(cam2worlds) - 1)
    world2cam = inverse_pose(cam2worlds[idx])
    cam2cams = torch.zeros_like(world2cam)
    cam2cams[:, :3, :3] = torch.bmm(world2cam[:, :3, :3], cam2worlds[indices, :3, :3])
    cam2cams[:, :3, 3] = torch.bmm(world2cam[:, :3, :3], cam2worlds[indices, :3, 3:])[..., 0]
    cam2cams[:, :3, 3] += world2cam[:, :3, 3]
    return cam2cams

def get_fwd_bwd_cam2cams(cam2worlds, indices):
    fwd_cam2cams = get_cam2cams(cam2worlds, indices, 1)
    bwd_cam2cams = get_cam2cams(cam2worlds, indices, -1)
    return fwd_cam2cams, bwd_cam2cams

def pts2px(pts, f, center):
    pts[..., 1] = -pts[..., 1]
    pts[..., 2] = -pts[..., 2]
    pts[..., 2] = torch.clip(pts[..., 2].clone(), min=1e-6)
    return torch.stack(
        [pts[..., 0] / pts[..., 2] * f[0] + center[0], pts[..., 1] / pts[..., 2] * f[1] + center[1]], 
        dim=-1)

# def get_cam2cams(cam2worlds):
#     world2cam = inverse_pose(cam2worlds)
#     cam2cams = torch.zeros_like(cam2worlds[:-1])
#     cam2cams[:, :3, :3] = torch.bmm(world2cam[:-1, :3, :3], cam2worlds[1:, :3, :3])
#     cam2cams[:, :3, 3] = torch.bmm(world2cam[:-1, :3, :3], cam2worlds[1:, :3, 3:])[..., 0]
#     return cam2cams

# def get_fwd_bwd_cam2cams(cam2workds):
#     fwd_cam2cams = get_cam2cams(cam2workds)
#     fwd_cam2cams = torch.cat([fwd_cam2cams, torch.zeros_like(fwd_cam2cams[0][None])], dim=0)
#     bwd_cam2cams = get_cam2cams(cam2workds[::-1])
#     bwd_cam2cams = torch.cat([torch.zeros_like(fwd_cam2cams[0][None]), bwd_cam2cams], dim=0)
#     return fwd_cam2cams, bwd_cam2cams


def sixD_to_mtx(pose, t_scale=0.3):
    b1 = pose[..., 0]
    b1 = b1 / torch.norm(b1, dim=-1)[:, None]
    b2 = pose[..., 1] - torch.sum(b1 * pose[..., 1], dim=-1)[:, None] * b1
    b2 = b2 / torch.norm(b2, dim=-1)[:, None]
    b3 = torch.cross(b1, b2)

    return torch.stack([b1, b2, b3, t_scale * pose[..., 2]], dim=-1)


def mtx_to_sixD(pose):
    return torch.stack([pose[..., 0], pose[..., 1], pose[..., 3]], dim=-1)


class Pose:
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4])
    each [3,4] camera pose takes the form of [R|t]
    """

    def __call__(self, R=None, t=None):
        # construct a camera pose from the given R and/or t
        assert R is not None or t is not None
        if R is None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            R = torch.eye(3, device=t.device).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1], device=R.device)
        else:
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R)
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
        assert R.shape[:-1] == t.shape and R.shape[-2:] == (3, 3)
        R = R.float()
        t = t.float()
        pose = torch.cat([R, t[..., None]], dim=-1)  # [...,3,4]
        assert pose.shape[-2:] == (3, 4)
        return pose

    def invert(self, pose, use_inverse=False):
        # invert a camera pose
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv @ t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        # compose a sequence of poses together
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b @ R_a
        t_new = (R_b @ t_a + t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new


class Lie:
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self, w):  # [...,3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I + A * wx + B * wx @ wx
        return R

    def SO3_to_so3(self, R, eps=1e-7):  # [...,3,3]
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()[
            ..., None, None
        ] % np.pi  # ln(R) will explode if theta==pi
        lnR = (
            1 / (2 * self.taylor_A(theta) + 1e-8) * (R - R.transpose(-2, -1))
        )  # FIXME: wei-chiu finds it weird
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    def se3_to_SE3(self, wu):  # [...,3]
        w, u = wu.split([3, 3], dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I + A * wx + B * wx @ wx
        V = I + B * wx + C * wx @ wx
        Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
        return Rt

    def SE3_to_se3(self, Rt, eps=1e-8):  # [...,3,4]
        R, t = Rt.split([3, 1], dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I - 0.5 * wx + (1 - A / (2 * B)) / (theta**2 + eps) * wx @ wx
        u = (invV @ t)[..., 0]
        wu = torch.cat([w, u], dim=-1)
        return wu

    def skew_symmetric(self, w):
        w0, w1, w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack(
            [
                torch.stack([O, -w2, w1], dim=-1),
                torch.stack([w2, O, -w0], dim=-1),
                torch.stack([-w1, w0, O], dim=-1),
            ],
            dim=-2,
        )
        return wx

    def taylor_A(self, x, nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            if i > 0:
                denom *= (2 * i) * (2 * i + 1)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_B(self, x, nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            denom *= (2 * i + 1) * (2 * i + 2)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_C(self, x, nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            denom *= (2 * i + 2) * (2 * i + 3)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans


class Quaternion:
    def q_to_R(self, q):
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        qa, qb, qc, qd = q.unbind(dim=-1)
        R = torch.stack(
            [
                torch.stack(
                    [
                        1 - 2 * (qc**2 + qd**2),
                        2 * (qb * qc - qa * qd),
                        2 * (qa * qc + qb * qd),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        2 * (qb * qc + qa * qd),
                        1 - 2 * (qb**2 + qd**2),
                        2 * (qc * qd - qa * qb),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        2 * (qb * qd - qa * qc),
                        2 * (qa * qb + qc * qd),
                        1 - 2 * (qb**2 + qc**2),
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )
        return R

    def R_to_q(self, R, eps=1e-8):  # [B,3,3]
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # FIXME: this function seems a bit problematic, need to double-check
        row0, row1, row2 = R.unbind(dim=-2)
        R00, R01, R02 = row0.unbind(dim=-1)
        R10, R11, R12 = row1.unbind(dim=-1)
        R20, R21, R22 = row2.unbind(dim=-1)
        t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        r = (1 + t + eps).sqrt()
        qa = 0.5 * r
        qb = (R21 - R12).sign() * 0.5 * (1 + R00 - R11 - R22 + eps).sqrt()
        qc = (R02 - R20).sign() * 0.5 * (1 - R00 + R11 - R22 + eps).sqrt()
        qd = (R10 - R01).sign() * 0.5 * (1 - R00 - R11 + R22 + eps).sqrt()
        q = torch.stack([qa, qb, qc, qd], dim=-1)
        for i, qi in enumerate(q):
            if torch.isnan(qi).any():
                K = (
                    torch.stack(
                        [
                            torch.stack(
                                [R00 - R11 - R22, R10 + R01, R20 + R02, R12 - R21],
                                dim=-1,
                            ),
                            torch.stack(
                                [R10 + R01, R11 - R00 - R22, R21 + R12, R20 - R02],
                                dim=-1,
                            ),
                            torch.stack(
                                [R20 + R02, R21 + R12, R22 - R00 - R11, R01 - R10],
                                dim=-1,
                            ),
                            torch.stack(
                                [R12 - R21, R20 - R02, R01 - R10, R00 + R11 + R22],
                                dim=-1,
                            ),
                        ],
                        dim=-2,
                    )
                    / 3.0
                )
                K = K[i]
                eigval, eigvec = torch.linalg.eigh(K)
                V = eigvec[:, eigval.argmax()]
                q[i] = torch.stack([V[3], V[0], V[1], V[2]])
        return q

    def invert(self, q):
        qa, qb, qc, qd = q.unbind(dim=-1)
        norm = q.norm(dim=-1, keepdim=True)
        q_inv = torch.stack([qa, -qb, -qc, -qd], dim=-1) / norm**2
        return q_inv

    def product(self, q1, q2):  # [B,4]
        q1a, q1b, q1c, q1d = q1.unbind(dim=-1)
        q2a, q2b, q2c, q2d = q2.unbind(dim=-1)
        hamil_prod = torch.stack(
            [
                q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d,
                q1a * q2b + q1b * q2a + q1c * q2d - q1d * q2c,
                q1a * q2c - q1b * q2d + q1c * q2a + q1d * q2b,
                q1a * q2d + q1b * q2c - q1c * q2b + q1d * q2a,
            ],
            dim=-1,
        )
        return hamil_prod


pose = Pose()
lie = Lie()
quaternion = Quaternion()


def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom


# basic operations of transforming 3D points between world/camera/image coordinates
def world2cam(X, pose):  # [B,N,3]
    X_hom = to_hom(X)
    return X_hom @ pose.transpose(-1, -2)


def cam2img(X, cam_intr):
    return X @ cam_intr.transpose(-1, -2)


def img2cam(X, cam_intr):
    return X @ cam_intr.inverse().transpose(-1, -2)


def cam2world(X, pose):
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    return X_hom @ pose_inv.transpose(-1, -2)


def angle_to_rotation_matrix(a, axis):
    # get the rotation matrix from Euler angle around specific axis
    roll = dict(X=1, Y=2, Z=0)[axis]
    O = torch.zeros_like(a)
    I = torch.ones_like(a)
    M = torch.stack(
        [
            torch.stack([a.cos(), -a.sin(), O], dim=-1),
            torch.stack([a.sin(), a.cos(), O], dim=-1),
            torch.stack([O, O, I], dim=-1),
        ],
        dim=-2,
    )
    M = M.roll((roll, roll), dims=(-2, -1))
    return M


def get_center_and_ray(opt, pose, intr=None):  # [HW,2]
    # given the intrinsic/extrinsic matrices, get the camera center and ray directions]
    assert opt.camera.model == "perspective"
    with torch.no_grad():
        # compute image coordinate grid
        y_range = torch.arange(opt.H, dtype=torch.float32, device=opt.device).add_(0.5)
        x_range = torch.arange(opt.W, dtype=torch.float32, device=opt.device).add_(0.5)
        Y, X = torch.meshgrid(y_range, x_range)  # [H,W]
        xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]
    # compute center and ray
    batch_size = len(pose)
    xy_grid = xy_grid.repeat(batch_size, 1, 1)  # [B,HW,2]
    grid_3D = img2cam(to_hom(xy_grid), intr)  # [B,HW,3]
    center_3D = torch.zeros_like(grid_3D)  # [B,HW,3]
    # transform from camera to world coordinates
    grid_3D = cam2world(grid_3D, pose)  # [B,HW,3]
    center_3D = cam2world(center_3D, pose)  # [B,HW,3]
    ray = grid_3D - center_3D  # [B,HW,3]
    return center_3D, ray


def get_3D_points_from_depth(opt, center, ray, depth, multi_samples=False):
    if multi_samples:
        center, ray = center[:, :, None], ray[:, :, None]
    # x = c+dv
    points_3D = center + ray * depth  # [B,HW,3]/[B,HW,N,3]/[N,3]
    return points_3D


def convert_NDC(opt, center, ray, intr, near=1):
    # shift camera center (ray origins) to near plane (z=1)
    # (unlike conventional NDC, we assume the cameras are facing towards the +z direction)
    center = center + (near - center[..., 2:]) / ray[..., 2:] * ray
    # projection
    cx, cy, cz = center.unbind(dim=-1)  # [B,HW]
    rx, ry, rz = ray.unbind(dim=-1)  # [B,HW]
    scale_x = intr[:, 0, 0] / intr[:, 0, 2]  # [B]
    scale_y = intr[:, 1, 1] / intr[:, 1, 2]  # [B]
    cnx = scale_x[:, None] * (cx / cz)
    cny = scale_y[:, None] * (cy / cz)
    cnz = 1 - 2 * near / cz
    rnx = scale_x[:, None] * (rx / rz - cx / cz)
    rny = scale_y[:, None] * (ry / rz - cy / cz)
    rnz = 2 * near / cz
    center_ndc = torch.stack([cnx, cny, cnz], dim=-1)  # [B,HW,3]
    ray_ndc = torch.stack([rnx, rny, rnz], dim=-1)  # [B,HW,3]
    return center_ndc, ray_ndc


def rotation_distance(R1, R2, eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1 @ R2.transpose(-2, -1)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = (
        ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()
    )  # numerical stability near -1/+1
    return angle


def procrustes_analysis(X0, X1):  # [N,3]
    # translation
    t0 = X0.mean(dim=0, keepdim=True)
    t1 = X1.mean(dim=0, keepdim=True)
    X0c = X0 - t0
    X1c = X1 - t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c / s0
    X1cs = X1c / s1
    # rotation (use double for SVD, float loses precision)
    U, S, V = (X0cs.t() @ X1cs).double().svd(some=True)
    R = (U @ V.t()).float()
    if R.det() < 0:
        R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0], t1=t1[0], s0=s0, s1=s1, R=R)
    return sim3


def get_novel_view_poses(pose_anchor, N=60, scale=1):
    # create circular viewpoints (small oscillations)
    theta = torch.arange(N) / N * 2 * np.pi
    R_x = angle_to_rotation_matrix((-theta.sin() * 0.05 / 3).asin(), "X")
    R_y = angle_to_rotation_matrix((-theta.cos() * 0.05).asin(), "Y")
    pose_rot = pose(R=R_y @ R_x)
    pose_shift = pose(t=[0, 0, 4.0 * scale])
    pose_shift2 = pose(t=[0, 0, -4.0 * scale])
    pose_oscil = pose.compose([pose_shift, pose_rot, pose_shift2])
    pose_novel = pose.compose([pose_oscil, pose_anchor.cpu()[None]])
    return pose_novel

def rodrigues_rot(P, n0, n1):
    """
    Rotate given points based on a starting and ending vector

    Axis k and angle of rotation theta given by vectors n0, n1
        P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
    """
    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[np.newaxis, :]

    # Get vector of rotation k and angle theta
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    k = np.cross(n0, n1)
    k = k / np.linalg.norm(k)
    theta = np.arccos(np.dot(n0, n1))

    # Compute rotated points
    P_rot = np.zeros((len(P), 3))
    for i in range(len(P)):
        P_rot[i] = (
            P[i] * np.cos(theta)
            + np.cross(k, P[i]) * np.sin(theta)
            + k * np.dot(k, P[i]) * (1 - np.cos(theta))
        )

    return P_rot

def closest_point_2_lines(oa, da, ob, db):
    """
    Find a central point they are all looking at.

    Returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    """
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom

def fit_circle_2d(x: np.ndarray, y: np.ndarray, w: np.ndarray = None):
    """
    Find center [xc, yc] and radius r of circle fitting to set of 2D points.
    Optionally specify weights for points.

    Implicit circle function:
        (x-xc)^2 + (y-yc)^2 = r^2
        (2*xc)*x + (2*yc)*y + (r^2-xc^2-yc^2) = x^2+y^2
        c[0]*x + c[1]*y + c[2] = x^2+y^2

    Solution by method of least squares:
        A*c = b, c' = argmin(||A*c - b||^2)
        A = [x y 1], b = [x^2+y^2]
    """
    w = w or []
    A = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2

    # Modify A, b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W, A)
        b = np.dot(W, b)

    # Solve by method of least squares
    c = np.linalg.lstsq(A, b, rcond=None)[0]

    # Get circle parameters from solution c
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r

def angle_between(u, v, n=None):
    """
    Get angle between vectors u,v with sign based on plane with unit normal n
    """
    if n is None:
        return np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))
    else:
        return np.arctan2(np.dot(n, np.cross(u, v)), np.dot(u, v))

def generate_circle_by_vectors(t, C, r, n, u):
    """
    Generate points on circle
        P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C
    """
    n = n / np.linalg.norm(n)
    u = u / np.linalg.norm(u)
    P_circle = (
        r * np.cos(t)[:, np.newaxis] * u
        + r * np.sin(t)[:, np.newaxis] * np.cross(n, u)
        + C
    )
    return P_circle

def compute_render_path(poses: np.ndarray, N: int = 100) -> np.ndarray:
    # Fitting plane by SVD for the mean-centered data
    # Eq. of plane is <p,n> + d = 0, where p is a point on plane and n is normal vector
    P = poses[:, :3, 3]
    P_mean = P.mean(axis=0)
    P_centered = P - P_mean
    U, s, V = np.linalg.svd(P_centered)

    # Normal vector of fitting plane is given by 3rd column in V
    # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
    normal = V[2, :]

    # take the normal direction that matches the majority of camera y directions.
    # SVD might indeed arbitrarily flip the direction
    fraction_valid = ((poses[:, :3, 1] * normal).sum(-1) > 0).mean()
    if fraction_valid < 0.5:
        normal = -normal

    # Project points to coords X-Y in 2D plane
    P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])

    # Fit circle in new 2D coords
    xc, yc, r = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])

    # Generate circle points in 2D
    t = np.linspace(0, 2 * np.pi, N)

    # Transform circle center back to 3D coords
    C = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], normal) + P_mean
    C = C.flatten()

    # Generate points for fitting circle
    t = np.linspace(0, 2 * np.pi, N)
    u = P[0] - C
    P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)

    # Generate points for fitting arc
    u = P[0] - C
    v = P[-1] - C
    theta = angle_between(u, v, normal)

    t = np.linspace(0, theta, N)

    # Compute center of attention
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for mf in poses:
        for mg in poses:
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    attention_center = totp

    render_views_xs, render_views_ys, render_views_zs = [], [], []
    render_poses = []
    for camorigin in P_fitcircle:
        vec0 = normalize(np.cross(normalize(camorigin - C), normal))
        vec2 = normalize(camorigin - attention_center)
        vec1 = normalize(np.cross(vec2, vec0))

        p = np.stack([vec0, vec1, vec2, camorigin], 1)

        render_poses.append(camorigin)
        render_views_zs.append(vec2)
        render_views_xs.append(vec0)
        render_views_ys.append(vec1)

    render_c2ws = np.stack(np.eye(4) for _ in range(len(render_poses)))
    render_c2ws[:, :3, 2] = np.stack(render_views_zs, 0)
    render_c2ws[:, :3, 0] = -np.stack(render_views_xs, 0)
    render_c2ws[:, :3, 1] = -np.stack(render_views_ys, 0)
    render_c2ws[:, :3, 3] = np.stack(render_poses, 0)

    return render_c2ws[:, :3, :]

def get_lerp_poses(poses, N=120):
    pose_time = torch.arange(N).to(poses) / (N - 1)
    t = (
        poses[-1, :, 3][None] * pose_time[:, None]
        + (1 - pose_time[:, None]) * poses[0, :, 3][None]
    )
    z = (
        poses[-1, :, 2][None] * pose_time[:, None]
        + (1 - pose_time[:, None]) * poses[0, :, 2][None]
    )
    z /= torch.norm(z, dim=-1)[:, None]
    y_ = (
        poses[-1, :, 1][None] * pose_time[:, None]
        + (1 - pose_time[:, None]) * poses[0, :, 1][None]
    )
    x = torch.cross(z, y_)
    x /= torch.norm(x, dim=-1)[:, None]
    y = torch.cross(x, z)
    pose_interp = torch.stack([x, y, z, t], -1)
    return pose_interp


def pose_weighted_sum(poses, weights):
    poses[:, 0] = -poses[:, 0]
    t = torch.sum(poses[:, :, 3] * weights[:, None], dim=0, keepdim=True)
    z = torch.sum(poses[:, :, 2] * weights[:, None], dim=0, keepdim=True)
    z /= torch.norm(z, dim=-1)[:, None]
    y_ = torch.sum(poses[:, :, 1] * weights[:, None], dim=0, keepdim=True)
    x = torch.cross(z, y_)
    x /= torch.norm(x, dim=-1)[:, None]
    y = torch.cross(x, z)

    pose = torch.stack([x, y, z, t], -1)[0]
    poses[:, 0] = -poses[:, 0]
    pose[0] = -pose[0]

    return pose

def smooth_vec(vec, time, s):
    smoothed = np.zeros_like(vec)
    for i in range(vec.shape[1]):
        spl = UnivariateSpline(time, vec[..., i])
        spl.set_smoothing_factor(s)
        smoothed[..., i] = spl(time)
    return smoothed

def smooth_poses_spline(poses, st=1, sr=1):
    poses[:, 0] = -poses[:, 0]
    posesnp = poses.cpu().numpy()
    scale = 2e-2 / np.median(np.linalg.norm(posesnp[1:, :3, 3] - posesnp[:-1, :3, 3], axis=-1))
    posesnp[:, :3, 3] *= scale
    time = np.linspace(0, 1, len(posesnp))
    
    t = smooth_vec(posesnp[..., 3], time, st)
    z = smooth_vec(posesnp[..., 2], time, sr)
    z /= np.linalg.norm(z, axis=-1)[:, None]
    y_ = smooth_vec(posesnp[..., 1], time, sr)
    x = np.cross(z, y_)
    x /= np.linalg.norm(x, axis=-1)[:, None]
    y = np.cross(x, z)

    smooth_posesnp = np.stack([x, y, z, t], -1)
    poses[:, 0] = -poses[:, 0]
    smooth_posesnp[:, 0] = -smooth_posesnp[:, 0]
    smooth_posesnp[:, :3, 3] /= scale
    return torch.tensor(smooth_posesnp.astype(np.float32)).to(poses)

def upsample_poses_2x(poses):
    upsampled_poses = [poses[0]]
    for index in range(len(poses) - 1):
        weights = torch.tensor([0.5, 0.5]).to(poses)
        upsampled_poses.append(pose_weighted_sum(poses[index : index + 2], weights))
        upsampled_poses.append(poses[index + 1])
    return torch.stack(upsampled_poses, dim=0)


def smooth_poses(poses, k_size=30, sigma=10):
    ker = [
        np.exp(-0.5 * (k_size - i) ** 2 / (sigma**2)) for i in range(2 * k_size + 1)
    ]
    ker = torch.Tensor(ker).to(poses.device)
    smoothed_poses = []
    n_poses = poses.shape[0]
    for index in range(n_poses):
        eff_k_sizem = min(index, k_size)
        eff_k_sizep = min(n_poses - index - 1, k_size)
        eff_ker = ker[k_size - eff_k_sizem : k_size + eff_k_sizep + 1].clone()
        eff_ker /= torch.sum(eff_ker)
        support_poses = poses[index - eff_k_sizem : index + eff_k_sizep + 1].clone()
        smoothed_pose = pose_weighted_sum(support_poses, eff_ker)
        smoothed_poses.append(smoothed_pose)
    return torch.stack(smoothed_poses, dim=0)

from scipy.interpolate import UnivariateSpline
def smooth_poses6D_spline(poses, sp=1, sr=1):
    posesnp = poses.cpu().numpy()
    t = np.linspace(0, 1, len(posesnp))
    smooth_posesnp = np.zeros_like(posesnp)
    for i, j in [(i,j) for i in range(3) for j in range(3)]:
        spl = UnivariateSpline(t, posesnp[:, i, j])
        s = sp if j == 2 else sr
        spl.set_smoothing_factor(s)
        smooth_posesnp[:, i, j] = spl(t)

    return sixD_to_mtx(torch.tensor(smooth_posesnp.astype(np.float32)).to(poses))

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg
