"""Report the measured facing of every preset pose, to validate the heading convention."""

import math

import numpy as np
import warp as wp

from motionbricks.io.usd import load_skeleton_from_usd
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file
from motionbricks.model.contract import ModelRig
from webapp.motion import _model_to_runtime
from webapp.poses import POSE_SPECS, PoseLibrary

wp.init()
skeleton = load_skeleton_from_usd(get_asset_file("human", "soma_base_skel_minimal_no_root.usd"), SOMA_SKELETON_PRIM_PATH)
library = PoseLibrary(skeleton, _model_to_runtime(ModelRig.SOMA))

def angle(vec):
    return math.degrees(math.atan2(vec[1], vec[0]))

for spec in POSE_SPECS:
    pose = library.canonical(spec.key)
    world = library.world_positions(pose)
    idx = {name: i for i, name in enumerate(skeleton.joint_names)}
    hips = world[idx["Hips"]]
    left_leg = world[idx["LeftLeg"]]
    right_leg = world[idx["RightLeg"]]
    left_toe = world[idx["LeftToeBase"]]
    right_toe = world[idx["RightToeBase"]]
    left_dir = left_leg - right_leg
    left_dir[2] = 0
    forward_leg = np.cross(left_dir / np.linalg.norm(left_dir), [0, 0, 1])
    toe_dir = (left_toe + right_toe) / 2 - hips
    toe_dir[2] = 0
    root_yaw = math.degrees(library.forward_yaw(spec.key))
    print(
        f"{spec.key:9s} rootYaw(deg)={root_yaw:8.2f}  legForward={angle(forward_leg):8.2f}  "
        f"toeForward={angle(toe_dir):8.2f}  hipsZ={hips[2]:.3f}  feetZ={left_toe[2]:.3f}"
    )
