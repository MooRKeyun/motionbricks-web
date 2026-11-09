"""Compare the skeleton reference pose against the mesh bind pose, joint by joint."""

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


def q2m(q):
    x, y, z, w = q / np.linalg.norm(q)
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


world = [None] * joints


def resolve(i):
    if world[i] is not None:
        return
    parent = parents[i]
    if parent < 0:
        world[i] = to_mat(reference[i])
    else:
        resolve(parent)
        world[i] = world[parent] @ to_mat(reference[i])


for i in range(joints):
    resolve(i)

bind_world = [np.linalg.inv(to_mat(ibm[i])) for i in range(joints)]
for name in ("Hips", "Spine2", "Head", "LeftFoot", "RightHand"):
    i = skeleton.joint_index(name)
    m = world[i] @ to_mat(ibm[i])
    print(f"{name:10s} refWorldPos={world[i][:3, 3].round(3)} bindWorldPos={bind_world[i][:3, 3].round(3)}")
    print("           mismatch=\n", np.round(m, 3))

# Are all mismatches the same matrix?
base = world[0] @ to_mat(ibm[0])
deltas = [np.abs(world[i] @ to_mat(ibm[i]) - base).max() for i in range(joints)]
print("max deviation from the root mismatch:", round(float(np.max(deltas)), 4))
print("root mismatch:\n", np.round(base, 3))
