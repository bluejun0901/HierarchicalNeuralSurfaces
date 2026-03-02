import torch
import torch.nn.functional as F

def get_face_areas(v,f):
    f_norms = torch.cross(v[f[:,0]]-v[f[:,1]], v[f[:,0]]-v[f[:,2]], dim=1)
    f_areas = torch.sum(f_norms**2, dim=1) **0.5 * 0.5
    return f_areas


def sample_points_on_mesh(v,f,num_pts, prior = None):
    if prior is None:
        f_areas = get_face_areas(v,f)
        probs = F.normalize(f_areas, dim=0, p=1)
    else:
        probs = F.normalize(prior, dim=0, p=1)
    num_pts = int(num_pts)
    sampled_f_idxs = torch.multinomial(probs, num_pts, replacement=True)
    sampled_f = f[sampled_f_idxs]
    alpha = torch.rand(num_pts,1).to(v.device)
    beta = torch.rand(num_pts,1).to(v.device)
    k = beta**0.5
    a = 1 - k
    b = (1-alpha) * k
    c = alpha * k
    sampled_pts = v[sampled_f[:,0],:]*a + v[sampled_f[:,1],:]*b + v[sampled_f[:,2],:]*c
    return sampled_pts, sampled_f, (a,b,c)


def _segment_point_dist_sq(
    p: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    # p: [P, T, 3], x/y: [1, T, 3] or [P, T, 3]
    v = y - x
    w = p - x
    vv = (v * v).sum(dim=-1).clamp_min(eps)
    t = (w * v).sum(dim=-1) / vv
    t = t.clamp(0.0, 1.0)
    proj = x + t[..., None] * v
    d = p - proj
    return (d * d).sum(dim=-1)


def _point_triangle_dist_sq(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    # points: [P, 3], triangles: [T, 3, 3]
    p = points[:, None, :]  # [P, 1, 3]
    a = triangles[None, :, 0, :]  # [1, T, 3]
    b = triangles[None, :, 1, :]
    c = triangles[None, :, 2, :]

    ab = b - a
    bc = c - b
    ca = a - c
    ap = p - a
    bp = p - b
    cp = p - c
    n = torch.cross(ab, c - a, dim=-1)
    n2 = (n * n).sum(dim=-1).clamp_min(1e-12)

    # Inside/outside test in triangle plane (orientation-independent).
    c0 = (torch.cross(ab, ap, dim=-1) * n).sum(dim=-1)
    c1 = (torch.cross(bc, bp, dim=-1) * n).sum(dim=-1)
    c2 = (torch.cross(ca, cp, dim=-1) * n).sum(dim=-1)
    inside = ((c0 >= 0) & (c1 >= 0) & (c2 >= 0)) | ((c0 <= 0) & (c1 <= 0) & (c2 <= 0))

    plane_dist_sq = ((ap * n).sum(dim=-1) ** 2) / n2
    edge_dist_sq = torch.minimum(
        torch.minimum(_segment_point_dist_sq(p, a, b), _segment_point_dist_sq(p, b, c)),
        _segment_point_dist_sq(p, c, a),
    )
    return torch.where(inside, plane_dist_sq, edge_dist_sq)


def _closest_face_distances(
    points: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
    point_chunk: int = 1024,
    face_chunk: int = 1024,
):
    tris = verts[faces]
    all_min_d2 = []
    all_min_idx = []
    for p0 in range(0, points.shape[0], point_chunk):
        p_chunk = points[p0 : p0 + point_chunk]
        min_d2 = torch.full((p_chunk.shape[0],), float("inf"), device=points.device)
        min_idx = torch.zeros((p_chunk.shape[0],), dtype=torch.long, device=points.device)
        for f0 in range(0, tris.shape[0], face_chunk):
            tri_chunk = tris[f0 : f0 + face_chunk]
            d2 = _point_triangle_dist_sq(p_chunk, tri_chunk)
            chunk_min_d2, chunk_idx = d2.min(dim=1)
            update = chunk_min_d2 < min_d2
            min_d2 = torch.where(update, chunk_min_d2, min_d2)
            min_idx = torch.where(update, chunk_idx + f0, min_idx)
        all_min_d2.append(min_d2)
        all_min_idx.append(min_idx)
    return torch.cat(all_min_d2, dim=0), torch.cat(all_min_idx, dim=0)


def point2mesh_error(dv, ov, of, scale = 1.0):
    ov = ov.to(dv.device)
    of = of.to(dv.device)
    dpcl = dv * scale
    ov_scaled = ov * scale
    d2, _ = _closest_face_distances(dpcl, ov_scaled, of)
    errors = torch.sqrt(d2.clamp_min(0.0))
    return errors.mean() / scale


def find_closest_points(ov, of, pts):
    ov_scaled = ov * 1e4
    pts_scaled = pts * 1e4
    _, idxs = _closest_face_distances(pts_scaled, ov_scaled, of)
    closest_faces = of[idxs]
    closest_fnorms = torch.cross(
        ov_scaled[closest_faces[:, 0]] - ov_scaled[closest_faces[:, 1]],
        ov_scaled[closest_faces[:, 0]] - ov_scaled[closest_faces[:, 2]],
        dim=1,
    )
    norm_consts = torch.sum(closest_fnorms**2, dim=1) **0.5
    closest_fnorms /= norm_consts[:,None].clamp_min(1e-12)
    vec_to_closest = ov_scaled[closest_faces[:,0]] - pts_scaled
    distances = (vec_to_closest*closest_fnorms).sum(dim=1, keepdim=True)
    displacements = (distances*closest_fnorms)/1e4
    return displacements + pts


