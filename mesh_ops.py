import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class FaceData:
    verts_idx: torch.Tensor


class IcoSphere:
    def __init__(self, verts: torch.Tensor, faces: torch.Tensor):
        self._verts = verts
        self._faces = faces

    def verts_packed(self) -> torch.Tensor:
        return self._verts

    def faces_packed(self) -> torch.Tensor:
        return self._faces


def save_obj(path: str, verts: torch.Tensor, faces: torch.Tensor) -> None:
    verts_cpu = verts.detach().cpu()
    faces_cpu = faces.detach().cpu().to(torch.long)
    with open(path, "w", encoding="utf-8") as f:
        for v in verts_cpu:
            f.write(f"v {v[0].item()} {v[1].item()} {v[2].item()}\n")
        for tri in faces_cpu:
            # OBJ uses 1-based indexing.
            i, j, k = (tri + 1).tolist()
            f.write(f"f {i} {j} {k}\n")


def load_obj(
    path: str,
    load_textures: bool = False,  # kept for API compatibility
    device: str = "cpu",
):
    del load_textures
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                _, x, y, z = line.split(maxsplit=3)
                verts.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                poly: List[int] = []
                for tok in tokens:
                    # Supports "f v", "f v/vt", "f v//vn", "f v/vt/vn".
                    v_idx = tok.split("/")[0]
                    poly.append(int(v_idx) - 1)
                if len(poly) >= 3:
                    # Fan triangulation for polygons.
                    for t in range(1, len(poly) - 1):
                        faces.append([poly[0], poly[t], poly[t + 1]])

    verts_t = torch.tensor(verts, dtype=torch.float32, device=device)
    faces_t = torch.tensor(faces, dtype=torch.long, device=device)
    return verts_t, FaceData(verts_idx=faces_t), None


def _normalize_rows(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm(dim=1, keepdim=True).clamp_min(eps))


def _create_icosahedron(device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = torch.tensor(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=torch.long,
        device=device,
    )
    return _normalize_rows(verts), faces


def _midpoint_vertex(
    i: int,
    j: int,
    verts_list: List[List[float]],
    midpoint_cache: Dict[Tuple[int, int], int],
) -> int:
    key = (i, j) if i < j else (j, i)
    if key in midpoint_cache:
        return midpoint_cache[key]
    vi = verts_list[i]
    vj = verts_list[j]
    mid = [(vi[0] + vj[0]) * 0.5, (vi[1] + vj[1]) * 0.5, (vi[2] + vj[2]) * 0.5]
    norm = math.sqrt(mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2])
    mid = [mid[0] / norm, mid[1] / norm, mid[2] / norm]
    verts_list.append(mid)
    idx = len(verts_list) - 1
    midpoint_cache[key] = idx
    return idx


def ico_sphere(level: int, device: str = "cpu") -> IcoSphere:
    verts_t, faces_t = _create_icosahedron(device="cpu")
    verts_list = verts_t.tolist()
    faces_list = faces_t.tolist()

    for _ in range(level):
        midpoint_cache: Dict[Tuple[int, int], int] = {}
        new_faces: List[List[int]] = []
        for tri in faces_list:
            v0, v1, v2 = tri
            a = _midpoint_vertex(v0, v1, verts_list, midpoint_cache)
            b = _midpoint_vertex(v1, v2, verts_list, midpoint_cache)
            c = _midpoint_vertex(v2, v0, verts_list, midpoint_cache)
            new_faces.append([v0, a, c])
            new_faces.append([v1, b, a])
            new_faces.append([v2, c, b])
            new_faces.append([a, b, c])
        faces_list = new_faces

    verts = torch.tensor(verts_list, dtype=torch.float32, device=device)
    faces = torch.tensor(faces_list, dtype=torch.long, device=device)
    return IcoSphere(verts=verts, faces=faces)


def _vertex_neighbor_mean(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    # Build undirected edges from triangles, then aggregate neighbor means.
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = torch.cat([e01, e12, e20], dim=0)
    rev_edges = edges[:, [1, 0]]
    directed = torch.cat([edges, rev_edges], dim=0)

    src = directed[:, 0]
    dst = directed[:, 1]
    accum = torch.zeros_like(verts)
    accum.index_add_(0, src, verts[dst])
    deg = torch.bincount(src, minlength=verts.shape[0]).to(verts.dtype).unsqueeze(1)
    return accum / deg.clamp_min(1.0)


def generate_embedding_from_normalized(
    normalized_path: str,
    embedding_path: str,
    smooth_iters: int = 30,
    lambda_step: float = 0.5,
    mu_step: float = -0.53,
    sphere_mix: float = 0.75,
) -> None:
    """
    Generate an embedding mesh that preserves vertex/face indexing.

    The output uses exactly the same topology as input_normalized.obj, so
    barycentric correspondence by face index remains valid.
    """
    verts, face_data, _ = load_obj(normalized_path, load_textures=False, device="cpu")
    faces = face_data.verts_idx
    if verts.numel() == 0 or faces.numel() == 0:
        raise ValueError(f"Invalid mesh in {normalized_path}")

    v = verts.clone()
    center = v.mean(dim=0, keepdim=True)
    v = v - center
    scale = v.norm(dim=1).max().clamp_min(1e-8)
    v = v / scale

    # Taubin smoothing: lambda/mu pair reduces high-frequency detail
    # while limiting global shrinkage.
    for _ in range(max(0, smooth_iters)):
        nbr_mean = _vertex_neighbor_mean(v, faces)
        v = v + lambda_step * (nbr_mean - v)
        nbr_mean = _vertex_neighbor_mean(v, faces)
        v = v + mu_step * (nbr_mean - v)

    sphere = v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
    embedding_verts = sphere_mix * sphere + (1.0 - sphere_mix) * v
    embedding_verts = embedding_verts / embedding_verts.norm(dim=1, keepdim=True).clamp_min(1e-8)

    save_obj(embedding_path, embedding_verts, faces)
