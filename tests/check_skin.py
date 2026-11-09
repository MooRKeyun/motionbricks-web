"""Skin the SOMA mesh in numpy for a reference pose and a clip pose, and report bounds."""

import numpy as np
import warp as wp

from motionbricks.io.bvh import load_bvh
from motionbricks.io.usd import load_skeleton_from_usd
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file, get_soma_human_model

wp.init()
skeleton = load_skeleton_from_usd(get_asset_file("human", "soma_base_skel_minimal_no_root.usd"), SOMA_SKELETON_PRIM_PATH)
mesh = get_soma_human_model()
assert mesh.skeleton.joint_names == skeleton.joint_names

joints = skeleton.num_joints
parents = [skeleton.joint_parent(i) for i in range(joints)]
ibm = np.asarray(mesh.inverse_bind_transforms.numpy(), dtype=np.float64)
part = mesh.skinned_meshes[0]
points = np.asarray(part.points.numpy(), dtype=np.float64)
indices = np.asarray(part.joint_indices.numpy(), dtype=np.int64).reshape(points.shape[0], -1)
weights = np.asarray(part.joint_weights.numpy(), dtype=np.float64).reshape(points.shape[0], -1)


def q2m(q):
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def to_mat(t):
    m = np.eye(4)
    m[:3, :3] = q2m(t[3:7])
    m[:3, 3] = t[:3]
    return m


def skin(local):
    world = [None] * joints

    def resolve(i):
        if world[i] is not None:
            return
        parent = parents[i]
        if parent < 0:
            world[i] = to_mat(local[i])
        else:
            resolve(parent)
            world[i] = world[parent] @ to_mat(local[i])

    for i in range(joints):
        resolve(i)
    mats = np.stack([world[i] @ to_mat(ibm[i]) for i in range(joints)])  # [J,4,4]
    homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)  # [N,4]
    out = np.zeros((points.shape[0], 4))
    for k in range(indices.shape[1]):
        m = mats[indices[:, k]]  # [N,4,4]
        out += np.einsum("nij,nj->ni", m, homo) * weights[:, k:k + 1]
    return out[:, :3]


reference = np.asarray(skeleton.reference_local_transforms.numpy(), dtype=np.float64)
bbox = skin(reference)
print("reference skinned bbox:", bbox.min(axis=0).round(3), bbox.max(axis=0).round(3))

clip = load_bvh(str(get_asset_file("..")), skeleton) if False else None
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "motionbricks" / "assets" / "motions" / "soma" / "locomotion" / "idle" / "idle_loop_001__A268.bvh"
clip = load_bvh(str(path), skeleton)
pose = wp.zeros(joints, dtype=wp.transform)
clip.sample(1.0, pose)
idle = np.asarray(pose.numpy(), dtype=np.float64).reshape(joints, 7)
bbox2 = skin(idle)
print("idle skinned bbox:   ", bbox2.min(axis=0).round(3), bbox2.max(axis=0).round(3))
print("influences per vertex:", indices.shape[1])
print("weight sums:", weights.sum(axis=1).min().round(3), weights.sum(axis=1).max().round(3))
