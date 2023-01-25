import numpy as np

import torch

# from lib_cuda import segment_cumsum_cuda

# from torch.utils.cpp_extension import load
# parent_dir = os.path.dirname(os.path.abspath(__file__))
# sources = [
#         os.path.join(parent_dir, path)
#         for path in ['cuda/segment_cumsum.cpp', 'cuda/segment_cumsum_kernel.cu']]
# __CUDA_FIRSTTIME__ = True


def eff_distloss_native(w, m, interval):
    """
    Efficient O(N) realization of distortion loss.
    There are B rays each with N sampled points.
    w:        Float tensor in shape [B,N]. Volume rendering weights of each point.
    m:        Float tensor in shape [B,N]. Midpoint distance to camera of each point.
    interval: Scalar or float tensor in shape [B,N]. The query interval of each point.
    """
    loss_uni = (1 / 3) * (interval * w.pow(2)).sum(dim=-1).mean()
    wm = w * m
    w_cumsum = w.cumsum(dim=-1)
    wm_cumsum = wm.cumsum(dim=-1)
    loss_bi_0 = wm[..., 1:] * w_cumsum[..., :-1]
    loss_bi_1 = w[..., 1:] * wm_cumsum[..., :-1]
    loss_bi = 2 * (loss_bi_0 - loss_bi_1).sum(dim=-1).mean()
    return loss_bi + loss_uni


# class EffDistLoss(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, w, m, interval):
#         """
#         Efficient O(N) realization of distortion loss.
#         There are B rays each with N sampled points.
#         w:        Float tensor in shape [B,N]. Volume rendering weights of each point.
#         m:        Float tensor in shape [B,N]. Midpoint distance to camera of each point.
#         interval: Scalar or float tensor in shape [B,N]. The query interval of each point.
#         """
#         n_rays = np.prod(w.shape[:-1])
#         wm = w * m
#         w_cumsum = w.cumsum(dim=-1)
#         wm_cumsum = wm.cumsum(dim=-1)

#         w_total = w_cumsum[..., [-1]]
#         wm_total = wm_cumsum[..., [-1]]
#         w_prefix = torch.cat([torch.zeros_like(w_total), w_cumsum[..., :-1]], dim=-1)
#         wm_prefix = torch.cat([torch.zeros_like(wm_total), wm_cumsum[..., :-1]], dim=-1)
#         loss_uni = (1 / 3) * interval * w.pow(2)
#         loss_bi = 2 * w * (m * w_prefix - wm_prefix)
#         if torch.is_tensor(interval):
#             ctx.save_for_backward(
#                 w, m, wm, w_prefix, w_total, wm_prefix, wm_total, interval
#             )
#             ctx.interval = None
#         else:
#             ctx.save_for_backward(w, m, wm, w_prefix, w_total, wm_prefix, wm_total)
#             ctx.interval = interval
#         ctx.n_rays = n_rays
#         return (loss_bi.sum() + loss_uni.sum()) / n_rays

#     @staticmethod
#     @torch.autograd.function.once_differentiable
#     def backward(ctx, grad_back):
#         interval = ctx.interval
#         n_rays = ctx.n_rays
#         if interval is None:
#             (
#                 w,
#                 m,
#                 wm,
#                 w_prefix,
#                 w_total,
#                 wm_prefix,
#                 wm_total,
#                 interval,
#             ) = ctx.saved_tensors
#         else:
#             w, m, wm, w_prefix, w_total, wm_prefix, wm_total = ctx.saved_tensors
#         grad_uni = (1 / 3) * interval * 2 * w
#         w_suffix = w_total - (w_prefix + w)
#         wm_suffix = wm_total - (wm_prefix + wm)
#         grad_bi = 2 * (m * (w_prefix - w_suffix) + (wm_suffix - wm_prefix))
#         grad = grad_back * (grad_bi + grad_uni) / n_rays
#         return grad, None, None, None


# eff_distloss = EffDistLoss.apply


# class FlattenEffDistLoss(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, w, m, interval, ray_id):
#         """
#         w:        Float tensor in shape [N]. Volume rendering weights of each point.
#         m:        Float tensor in shape [N]. Midpoint distance to camera of each point.
#         interval: Scalar or float tensor in shape [N]. The query interval of each point.
#         ray_id:   Long tensor in shape [N]. The ray index of each point.
#         """
#         # global __CUDA_FIRSTTIME__
#         # segment_cumsum_cuda = load(
#         #         name='segment_cumsum_cuda',
#         #         sources=sources,
#         #         verbose=__CUDA_FIRSTTIME__)
#         # __CUDA_FIRSTTIME__ = False

#         n_rays = ray_id.max() + 1
#         w_prefix, w_total, wm_prefix, wm_total = segment_cumsum_cuda.segment_cumsum(
#             w, m, ray_id
#         )
#         loss_uni = (1 / 3) * interval * w.pow(2)
#         loss_bi = 2 * w * (m * w_prefix - wm_prefix)
#         if torch.is_tensor(interval):
#             ctx.save_for_backward(
#                 w, m, w_prefix, w_total, wm_prefix, wm_total, ray_id, interval
#             )
#             ctx.interval = None
#         else:
#             ctx.save_for_backward(w, m, w_prefix, w_total, wm_prefix, wm_total, ray_id)
#             ctx.interval = interval
#         ctx.n_rays = n_rays
#         return (loss_bi.sum() + loss_uni.sum()) / n_rays

#     @staticmethod
#     @torch.autograd.function.once_differentiable
#     def backward(ctx, grad_back):
#         interval = ctx.interval
#         n_rays = ctx.n_rays
#         if interval is None:
#             (
#                 w,
#                 m,
#                 w_prefix,
#                 w_total,
#                 wm_prefix,
#                 wm_total,
#                 ray_id,
#                 interval,
#             ) = ctx.saved_tensors
#         else:
#             w, m, w_prefix, w_total, wm_prefix, wm_total, ray_id = ctx.saved_tensors
#         grad_uni = (1 / 3) * interval * 2 * w
#         w_suffix = w_total[ray_id] - (w_prefix + w)
#         wm_suffix = wm_total[ray_id] - (wm_prefix + w * m)
#         grad_bi = 2 * (m * (w_prefix - w_suffix) + (wm_suffix - wm_prefix))
#         grad = grad_back * (grad_bi + grad_uni) / n_rays
#         return grad, None, None, None


# flatten_eff_distloss = FlattenEffDistLoss.apply
