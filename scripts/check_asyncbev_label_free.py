"""Exercise the actual patched warping function on CPU without importing the detector.

Checks that evaluation is independent of GT flow and that training still consumes
supervision. This is a regression test, not a complete model accuracy benchmark.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data/external/AsyncBEV/projects/CMT/mmdet3d_plugin/models/dense_heads/cmt_head.py"
tree = ast.parse(path.read_text())
head = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CmtHead")
method = next(node for node in head.body if isinstance(node, ast.FunctionDef) and node.name == "temporal_bev_warping")
namespace = dict(torch=torch, Quaternion=Quaternion, transform_matrix=transform_matrix)
exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)


def run(training, with_labels, label_value=1):
    calls = []
    def flow_embedder(points, flows):
        calls.append(True)
        return [], torch.ones(1, 3, 2, 2)
    self = SimpleNamespace(bev_warping="lidar", delta_t_noise=False, bev_h=2, bev_w=2,
        pc_range=[-1, -1, -1, 1, 1, 1], real_w=2, real_h=2, hidden_dim=1,
        learned_temporal_bev_warping="vel_with_loss", flow_modal="same",
        bev_embed_proj=lambda value: value, use_emc_pe_encoding=False, use_time_encoding=False,
        learned_vel_estimator=lambda value: torch.ones(1, 4, 2),
        pe_voxelizer=lambda value: [dict(voxel_coords=torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
                                        point_idxes=torch.arange(4))],
        training=training, flow_embedder=flow_embedder, vis_output=False)
    pose = dict(ego2global_translation=[0, 0, 0], ego2global_rotation=[1, 0, 0, 0])
    meta = dict(timestamp=1., async_metas=dict(timestamp=0.8, **pose),
                lidar2ego_translation=[0, 0, 0], lidar2ego_rotation=[1, 0, 0, 0], **pose)
    if with_labels:
        meta.update(gt_flow_async2sync=np.full((4, 3), label_value, np.float32),
                    pts_in_boxes_async=np.zeros((4, 3), np.float32))
    result, _ = namespace["temporal_bev_warping"](self, None, torch.zeros(1, 1, 2, 2),
                                               torch.zeros(1, 4, 3), [meta])
    return result, calls


without_labels, eval_calls = run(False, False)
with_labels, _ = run(False, True, label_value=999)
torch.testing.assert_allclose(without_labels, with_labels)
torch.testing.assert_allclose(without_labels[0, :, :2], torch.full((4, 2), 0.2))
assert not eval_calls
_, train_calls = run(True, True)
assert train_calls
print("PASS: actual warping runs without labels; predictions ignore GT flow; training supervision remains active")
