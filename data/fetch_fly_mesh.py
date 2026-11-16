#!/usr/bin/env python3
"""Fetch the NeuroMechFly body mesh and reduce it to something a web page can carry.

NeLy-EPFL, Apache-2.0 (Lobato-Rios et al. 2022, Nature Methods). 65 parts, 334,525 triangles
reduced to 81,866 by vertex clustering, 0.98 MB.

Three assumptions that have to be right or the fly renders as scattered specks, or as nothing at
all (D-068): the mesh `<scale>` in the SDF is 1000 and omitting it shrinks every part to a
ten-thousandth of its spacing; `<link><pose>` in SDF 1.6 is already relative to the model frame,
so composing the joint chain on top of it doubles the offsets; and the parts are not all in one
frame to begin with, so drawing them untransformed piles the abdomen and every leg onto one point.

    python data/fetch_fly_mesh.py
"""
import base64, json, re, struct, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "report"
RAW = "https://raw.githubusercontent.com/NeLy-EPFL/NeuroMechFly/main/"
SDF = RAW + "data/design/sdf/neuromechfly_noLimits.sdf"


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp,   cp*sr,            cp*cr]])


def pose_to_T(txt):
    v = [float(x) for x in txt.split()]
    T = np.eye(4)
    T[:3, :3] = rpy(*v[3:6]) if len(v) >= 6 else np.eye(3)
    T[:3, 3] = v[:3]
    return T


root = ET.fromstring(urllib.request.urlopen(SDF, timeout=90).read())
model = root.find(".//model")
links, meshes = {}, {}
for l in model.findall("link"):
    n = l.get("name")
    p = l.find("pose")
    links[n] = pose_to_T(p.text) if p is not None and p.text else np.eye(4)
    u = l.find(".//visual/geometry/mesh/uri")
    if u is not None and u.text:
        # the joint poses and the STL vertices are in different units -- the SDF carries the
        # conversion in <scale>, and ignoring it left every part 1/10,000 of its spacing, so
        # 91 % of triangles collapsed to zero area under quantisation and nothing rendered
        sc = l.find(".//visual/geometry/mesh/scale")
        sv = [float(x) for x in sc.text.split()] if sc is not None and sc.text else [1, 1, 1]
        vp = l.find(".//visual/pose")
        meshes[n] = (u.text.strip().split("/")[-1], np.array(sv),
                     pose_to_T(vp.text) if vp is not None and vp.text else np.eye(4))
parent = {}
for j in model.findall("joint"):
    c = j.find("child"); pa = j.find("parent")
    jp = j.find("pose")
    if c is not None and pa is not None:
        parent[c.text.strip()] = (pa.text.strip(),
                                  pose_to_T(jp.text) if jp is not None and jp.text else np.eye(4))
print(f"{len(links)} links, {len(meshes)} with a mesh, {len(parent)} joints")
_s = {tuple(np.round(v[1], 6)) for v in meshes.values()}
print(f"mesh <scale> values present: {sorted(_s)[:4]}")

def world(n, seen=None):
    seen = seen or set()
    if n in seen or n not in parent:
        return links.get(n, np.eye(4))
    seen.add(n)
    pa, jT = parent[n]
    return world(pa, seen) @ jT @ links.get(n, np.eye(4))


def read_stl(b):
    if b[:5].lower() == b"solid" and b"facet" in b[:512]:
        v = [[float(x) for x in t.split()[1:4]]
             for t in b.decode("ascii", "ignore").splitlines() if t.split()[:1] == ["vertex"]]
        return np.array(v, np.float32).reshape(-1, 3, 3)
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b, dtype=np.uint8, count=n * 50, offset=84).reshape(n, 50)
    return a[:, 12:48].copy().view(np.float32).reshape(n, 3, 3)


def decimate(tris, grid):
    v = tris.reshape(-1, 3)
    lo, hi = v.min(0), v.max(0)
    span = (hi - lo).max() or 1.0
    q = np.floor((v - lo) / span * grid).astype(np.int64)
    key = q[:, 0] * 4194304 + q[:, 1] * 2048 + q[:, 2]   # noqa: bodyid — grid hash, not a cell
    uniq, inv = np.unique(key, return_inverse=True)
    pos = np.zeros((len(uniq), 3)); cnt = np.zeros(len(uniq))
    np.add.at(pos, inv, v.astype(np.float64)); np.add.at(cnt, inv, 1)
    pos /= cnt[:, None]
    idx = inv.reshape(-1, 3)
    ok = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
    return pos.astype(np.float32), idx[ok].astype(np.uint32)


parts, tin, tout = {}, 0, 0
for link, (fn, mscale, vT) in meshes.items():
    try:
        b = urllib.request.urlopen(RAW + "data/design/meshes/stl/" + fn, timeout=60).read()
    except Exception as e:
        print(f"  {link}: {e}"); continue
    tris = read_stl(b)
    fine = any(k in link for k in ("Head", "Eye", "Thorax"))
    pos, idx = decimate(tris, 40 if fine else 22)
    pos = pos * mscale                              # mesh units -> model units
    # SDF 1.6: <link><pose> is already relative to the MODEL frame, so composing the joint chain
    # on top of it double-counts and blows the body apart -- link-pose spread is [2.30, 0, 0.28],
    # which matches a 3 mm fly, while the composed chain gives [5.30, 0, 3.92]
    T = links.get(link, np.eye(4)) @ vT
    pos = (pos @ T[:3, :3].T) + T[:3, 3]           # into the body frame
    parts[link] = (pos, idx)
    tin += len(tris); tout += len(idx)
print(f"\n{tin:,} -> {tout:,} triangles across {len(parts)} parts")

allv = np.vstack([p for p, _ in parts.values()])
lo, hi = allv.min(0), allv.max(0)
span = float((hi - lo).max())
cents = np.array([p.mean(0) for p, _ in parts.values()])
sizes = np.array([(p.max(0) - p.min(0)).max() for p, _ in parts.values()])
print(f"spread of part centres: {np.round((cents.max(0)-cents.min(0))/span, 3)}")
print(f"part size / spread: median {np.median(sizes)/span:.3f}, max {sizes.max()/span:.3f}"
      f"  (a body's parts should be a sizeable fraction of the whole)")
doc = {"source": "NeuroMechFly, NeLy-EPFL, Apache-2.0 (Lobato-Rios et al. 2022, Nat Methods)",
       "sdf": SDF, "decimation": "vertex clustering, 22-40 cell grid",
       "tris_before": int(tin), "tris_after": int(tout),
       "lo": lo.tolist(), "span": span, "parts": {}}
for k, (p, f) in parts.items():
    q = ((p - lo) / span * 65535).clip(0, 65535).astype(np.uint16)
    u32 = f.max() >= 65536
    doc["parts"][k] = {"v": base64.b64encode(q.tobytes()).decode(),
                       "f": base64.b64encode((f if u32 else f.astype(np.uint16)).tobytes()).decode(),
                       "u32": bool(u32), "n": int(len(p))}
out = OUT / "fly_parts.json"
out.write_text(json.dumps(doc, separators=(",", ":")))
print(f"wrote {out.name} ({out.stat().st_size/1e6:.2f} MB)")
