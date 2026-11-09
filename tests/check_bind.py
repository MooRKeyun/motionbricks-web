"""Verify that the exported bind pose matches the skeleton's reference pose."""

import numpy as np
import warp as wp

from motionbricks.io.utils import get_soma_human_model

wp.init()
mesh = get_soma_human_model()
skeleton = mesh.skeleton

reference = np.asarray(skeleton.reference_local_transforms.numpy(), dtype=np.float64)
ibm = np.asarray(mesh.inverse_bind_transforms.numpy(), dtype=np.float64)
joints = skeleton.num_joints
parents = [skeleton.joint_parent(i) for i in range(joints)]


def quat_to_mat(q):
    x, y, z, w = q
    n = np.linalg.norm(q)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def to_mat(t):
    m = np.eye(4)
    m[:3, :3] = quat_to_mat(t[3:7])
    m[:3, 3] = t[:3]
    return m


world = [None] * joints


def resolve(i):
    if world[i] is not None:
        return
    local = to_mat(reference[i])
    parent = parents[i]
    if parent < 0:
        world[i] = local
    else:
        resolve(parent)
        world[i] = world[parent] @ local


for i in range(joints):
    resolve(i)

errors = []
for i in range(joints):
    m = world[i] @ to_mat(ibm[i])
    errors.append(np.abs(m - np.eye(4)).max())

print("joints:", joints)
print("max |world*IBM - I| over all joints:", round(float(np.max(errors)), 4))
print("median:", round(float(np.median(errors)), 4))
print("root world pos:", world[0][:3, 3].round(3), "IBM pos:", ibm[0][:3].round(3))
print("mesh parts:", len(mesh.skinned_meshes), "influences:", mesh.skinned_meshes[0].num_influences)
pts = mesh.skinned_meshes[0].points.numpy()
print("points bbox:", pts.min(axis=0).round(3), pts.max(axis=0).round(3))
print("reference root:", reference[0].round(3))
