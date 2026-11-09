# SPDX-License-Identifier: Apache-2.0
"""Export the bundled SOMA character (USD) to a GLB with skinning data for the browser.

The MotionBricks SOMA character ships as an OpenUSD stage with a joint hierarchy and
skinned meshes. Browsers cannot read USD, so this script converts the skeleton, the
skinned geometry and the inverse bind transforms into a single binary glTF 2.0 file.

The export keeps the MotionBricks runtime convention: Z up, model space equal to the
skeleton space, identity scene root. The web viewer therefore uses a Z-up camera and
can apply generated local joint transforms to the bones without extra conversion.

Run it once; the server starts it automatically when the GLB is missing:

    python -m webapp.glb_export --out webapp/static/soma.glb
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import struct
from pathlib import Path

import numpy as np

from motionbricks.animation.mesh import SkeletalMesh
from motionbricks.io.utils import get_soma_human_model

# SOMA's USD stage is Y up. MotionBricks runs in Z up, so the export applies the same
# conversion the runtime uses: MUJOCO facing, a +90 degree turn about X.
MODEL_TO_RUNTIME = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64,
)
MODEL_TO_RUNTIME_QUAT = np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float64)

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

# glTF component types.
FLOAT = 5126
UNSIGNED_SHORT = 5123
UNSIGNED_INT = 5125

# Attribute types.
SCALAR = "SCALAR"
VEC2 = "VEC2"
VEC3 = "VEC3"
VEC4 = "VEC4"
MAT4 = "MAT4"


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix for an XYZW quaternion."""
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two XYZW quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def rotate_point(points: np.ndarray) -> np.ndarray:
    """Rotate XYZ points or normals from SOMA model space to MotionBricks runtime space."""
    out = np.empty_like(np.asarray(points, dtype=np.float64))
    out[..., 0] = points[..., 0]
    out[..., 1] = -points[..., 2]
    out[..., 2] = points[..., 1]
    return out


def transform_to_matrix(p: np.ndarray, q: np.ndarray, *, inverse_bind: bool = False) -> np.ndarray:
    """Return a 4x4 column-major matrix, as glTF stores it.

    An inverse bind matrix is post-multiplied by the inverse of the model-to-runtime
    rotation, so that it matches the rotated vertices and joint world transforms.
    """
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = quaternion_to_matrix(q)
    m[:3, 3] = p
    if inverse_bind:
        # Vertices are rotated by MODEL_TO_RUNTIME and the joint world transforms rotate
        # with the root, so the bind matrix becomes IBM @ R^-1: only its rotation changes.
        m[:3, :3] = m[:3, :3] @ MODEL_TO_RUNTIME.T
    return m.T.astype(np.float32).reshape(-1)


def vertex_normals(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Area-weighted smooth vertex normals for a triangle mesh."""
    normals = np.zeros_like(points, dtype=np.float64)
    tris = points[indices.reshape(-1, 3)].astype(np.float64)
    face = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    for corner in range(3):
        np.add.at(normals, indices.reshape(-1, 3)[:, corner], face)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(length, 1e-12)).astype(np.float32)


def top_four_influences(
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
    vertex_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the four strongest influences per vertex and renormalise the weights."""
    influences = joint_indices.reshape(vertex_count, -1)
    weights = joint_weights.reshape(vertex_count, -1)

    order = np.argsort(-weights, axis=1)
    rows = np.arange(vertex_count)[:, None]
    influences = influences[rows, order][:, :4]
    weights = weights[rows, order][:, :4].astype(np.float32)

    total = weights.sum(axis=1, keepdims=True)
    total[total <= 1e-12] = 1.0
    weights = weights / total

    if influences.shape[1] < 4:
        pad = 4 - influences.shape[1]
        influences = np.pad(influences, ((0, 0), (0, pad)))
        weights = np.pad(weights, ((0, 0), (0, pad)))
    return influences.astype(np.uint16), weights.astype(np.float32)


class GlbBuilder:
    """Accumulate binary data and glTF JSON for one GLB file."""

    def __init__(self) -> None:
        self.binary = bytearray()
        self.buffer_views: list[dict] = []
        self.accessors: list[dict] = []

    def _align(self) -> None:
        while len(self.binary) % 4:
            self.binary.append(0)

    def add_accessor(
        self,
        array: np.ndarray,
        component_type: int,
        type_name: str,
        *,
        target: int | None = None,
        min_max: bool = False,
    ) -> int:
        """Copy a numpy array into the binary chunk and register an accessor."""
        self._align()
        offset = len(self.binary)
        data = np.ascontiguousarray(array)
        self.binary.extend(data.tobytes())

        view: dict = {"buffer": 0, "byteOffset": offset, "byteLength": data.nbytes}
        if target is not None:
            view["target"] = target
        self.buffer_views.append(view)

        # A SCALAR accessor counts elements; every other type counts vectors/matrices.
        count = data.size if type_name == SCALAR else data.shape[0]
        accessor: dict = {
            "bufferView": len(self.buffer_views) - 1,
            "componentType": component_type,
            "count": count,
            "type": type_name,
        }
        if min_max:
            flat = data.reshape(count, -1)
            accessor["min"] = [float(v) for v in flat.min(axis=0)]
            accessor["max"] = [float(v) for v in flat.max(axis=0)]
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def to_glb(self, gltf: dict) -> bytes:
        """Serialise the JSON and the binary chunk into a GLB container."""
        self._align()
        gltf = dict(gltf)
        gltf["bufferViews"] = self.buffer_views
        gltf["accessors"] = self.accessors
        gltf["buffers"] = [{"byteLength": len(self.binary)}]

        json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        while len(json_bytes) % 4:
            json_bytes += b" "

        total = 12 + 8 + len(json_bytes) + 8 + len(self.binary)
        out = bytearray()
        out += struct.pack("<III", GLB_MAGIC, 2, total)
        out += struct.pack("<II", len(json_bytes), CHUNK_JSON)
        out += json_bytes
        out += struct.pack("<II", len(self.binary), CHUNK_BIN)
        out += self.binary
        return bytes(out)


def build_glb(mesh: SkeletalMesh) -> bytes:
    """Convert a SkeletalMesh into GLB bytes."""
    skeleton = mesh.skeleton
    joint_count = skeleton.num_joints
    names = list(skeleton.joint_names)
    parents = [skeleton.joint_parent(i) for i in range(joint_count)]
    reference = np.asarray(skeleton.reference_local_transforms.numpy(), dtype=np.float32)
    inverse_bind = np.asarray(mesh.inverse_bind_transforms.numpy(), dtype=np.float32)

    builder = GlbBuilder()

    nodes: list[dict] = []
    for index, name in enumerate(names):
        node: dict = {"name": name}
        p = reference[index, :3].astype(np.float64)
        q = reference[index, 3:7].astype(np.float64)
        if parents[index] < 0:
            # The root carries the model-to-runtime rotation for the whole hierarchy.
            p = rotate_point(p)
            q = quaternion_multiply(MODEL_TO_RUNTIME_QUAT, q)
        node["translation"] = [float(v) for v in p]
        node["rotation"] = [float(v) for v in q]
        nodes.append(node)
    for index, parent in enumerate(parents):
        if parent >= 0:
            nodes[parent].setdefault("children", []).append(index)

    ibm_accessor = builder.add_accessor(
        np.array(
            [
                transform_to_matrix(inverse_bind[i, :3], inverse_bind[i, 3:7], inverse_bind=True)
                for i in range(joint_count)
            ],
            dtype=np.float32,
        ),
        FLOAT,
        MAT4,
    )

    primitives = []
    materials = [{
        "name": "SOMA",
        "doubleSided": False,
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.78, 0.79, 0.82, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.72,
        },
    }]

    total_vertices = 0
    for part in mesh.skinned_meshes:
        points = np.asarray(part.points.numpy(), dtype=np.float32)
        indices = np.asarray(part.indices.numpy(), dtype=np.uint32).reshape(-1)
        if points.size == 0 or indices.size == 0:
            continue
        points = rotate_point(points).astype(np.float32)
        normals = rotate_point(vertex_normals(points, indices)).astype(np.float32)
        influences = np.asarray(part.joint_indices.numpy())
        weights = np.asarray(part.joint_weights.numpy(), dtype=np.float32)
        joints4, weights4 = top_four_influences(influences, weights, points.shape[0])

        position_accessor = builder.add_accessor(points, FLOAT, VEC3, target=34962, min_max=True)
        normal_accessor = builder.add_accessor(normals, FLOAT, VEC3, target=34962)
        joints_accessor = builder.add_accessor(joints4, UNSIGNED_SHORT, VEC4, target=34962)
        weights_accessor = builder.add_accessor(weights4, FLOAT, VEC4, target=34962)
        index_accessor = builder.add_accessor(indices, UNSIGNED_INT, SCALAR, target=34963)

        primitives.append({
            "attributes": {
                "POSITION": position_accessor,
                "NORMAL": normal_accessor,
                "JOINTS_0": joints_accessor,
                "WEIGHTS_0": weights_accessor,
            },
            "indices": index_accessor,
            "material": 0,
            "mode": 4,
        })
        total_vertices += points.shape[0]

    mesh_node_index = len(nodes)
    nodes.append({"name": "SOMA_Mesh", "mesh": 0, "skin": 0})
    root_index = 0
    skin = {
        "name": "SOMA_Skin",
        "joints": list(range(joint_count)),
        "inverseBindMatrices": ibm_accessor,
        "skeleton": root_index,
    }

    gltf = {
        "asset": {"version": "2.0", "generator": "motionbricks-web"},
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": [root_index, mesh_node_index]}],
        "nodes": nodes,
        "meshes": [{"name": "SOMA", "primitives": primitives}],
        "materials": materials,
        "skins": [skin],
        "extras": {
            "jointNames": names,
            "jointCount": joint_count,
            "vertexCount": int(total_vertices),
            "upAxis": "Z",
        },
    }
    return builder.to_glb(gltf)


def export_glb(out_path: Path) -> Path:
    """Load the bundled SOMA character and write it as a GLB file."""
    mesh = get_soma_human_model()
    data = build_glb(mesh)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "static" / "soma.glb")
    args = parser.parse_args()
    path = export_glb(args.out)
    print(f"wrote {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
