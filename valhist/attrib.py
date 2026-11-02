"""Attribution of the cluster-PE residual between fast and Geant4 steps to the step classes that make the clusters.

A step table has three classes: the primary muon itself (P), the muon's descendants (D) and everything from non-muon
primaries (S). An arm takes each class from one source and runs the same fast response and reco on the union:

    g  Geant4 steps            f  fast steps (stage C + tier 1b)       -  class left out
    s  (class P only) Geant4 primary steps with dE/dx exchanged at random between crossings of one tier-1b bin
    r  (class D only) tier 1b drawn on the arm's primary crossings (P = g or s: Geant4, P = f: fast)
    k  (class D only) as r, each crossing's donor taken among the 4 library crossings of its draw cell nearest in the
       muon's kinetic-energy loss to its next crossing (last crossings: among the cell's last crossings); no hard draw
    t  (class D only, P = g or s) as k, nearest in the descendant energy Geant4 attached to the crossing (diagnostic:
       uses the Geant4 descendants it replaces)

An arm is three letters P, D, S plus an optional @tag for an independent response seed: ggg, fff, gfg (Geant4 primary,
fast descendants, Geant4 showers), grg, g--, ggg@2. The pure arms keep the table's row order, so fff equals the
`fasttables chain` output of the same seed. Rows of one event are joined by (run, subrun, event); a class keeps its
own absolute positions, so in a hybrid the descendants sit around the muon they were made for (the r, k, t sources do
not have that leak; `same_bars` marks the events whose muons cross the same bars in both sources).

    python -m valhist.attrib shard --shard 22 --arms ggg,fff,gfg,fgg --g4 'G/g4all.{s}.root' --fast 'F/fast1b.{s}.root' \
        --calib C.csv --deviation D.csv --tier1b-dir T --out OUT
    python -m valhist.attrib refs --ref a0g='DIR/valhist.{s}.root' --shards 22-120 --out refs.npz
    python -m valhist.attrib report OUT --shards 22-120 [--refs refs.npz] --json REPORT.json

`shard` writes OUT/<arm>/<s>.npz (one row per cluster, with the visible energy of each class in the cluster's bars and
the identity of the primary that dominates it) and OUT/truth/<s>.npz. Cluster identity across arms is truth identity:
(event, primary, sector, rank in time within those), never distance.
"""
import argparse
import json
from pathlib import Path
import time

import awkward as ak
import numpy as np
import pandas as pd

from crvfastsim import tables, tier1b as T
from crvfastsim.analyses.census import classify
from crvfastsim.transport import all_primary_muons

KEY = tables.KEY
CLASSES = ("P", "D", "S")
P, D, S = 0, 1, 2
SHOWER_BASE = T.SHOWER_KEY_BASE
SOURCE_OFFSET = {"g": 0, "f": 1 << 40, "r": 2 << 40, "s": 3 << 40, "t": 4 << 40, "k": 5 << 40}
DRAW_SEED = {("r", "g"): 77, ("r", "f"): 78, ("t", "g"): 79, ("k", "g"): 80, ("k", "f"): 81}
MATCH_WINDOW = 4          # library crossings nearest in the matched quantity among which a donor is drawn
LAST_CROSSING = -50.0     # matched value of a muon's last crossing (no next crossing), below any energy loss
STEP_COLS = ["nstep", "bar", "sp", "t0", "t1", "x0", "y0", "z0", "x1", "y1", "z1", "px0", "py0", "pz0", "p1", "len",
             "edep"]
PE_EDGES = np.linspace(0.0, 2000.0, 101)
GATE_NS = (-200.0, 100.0)  # step t0 relative to the cluster's [start, end]; retuned in shard tests, see report
SECTOR_GROUP = np.array([0, 1, 2] + [3] * 8)
GROUP_NAMES = ("EX", "T1", "T2", "M")
E_CLASS = np.array([2.0, 5.0, 15.0, 50.0])  # MeV of descendant energy in the cluster's bars


def parse_arm(spec):
    """'gfg@2' -> ('g', 'f', 'g', 2)."""
    letters, _, tag = spec.partition("@")
    if len(letters) != 3 or letters[0] not in "gfs-" or letters[1] not in "gfrkt-" or letters[2] not in "gf-":
        raise ValueError(f"bad arm {spec!r}")
    if letters[1] in "rkt" and (letters[0] == "-" or (letters[1] == "t" and letters[0] == "f")):
        raise ValueError(f"bad arm {spec!r}: {letters[1]} needs {'Geant4' if letters[1] == 't' else 'primary'} crossings")
    return letters[0], letters[1], letters[2], int(tag or 0)


def draw_key(letters):
    """Source key of a drawn descendant class: (letter, crossings source), the s primary sharing the Geant4 crossings."""
    return letters[1], "f" if letters[0] == "f" else "g"


def shard_name(k):
    return f"001400_{k:08d}"


def shard_list(text):
    """'22-120' (even shards), '22,24' or a mix."""
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1, 2)) if b else [int(a)]
    return out


def tag_g4(steps, parts, gen, sector):
    """Geant4 steps with cls, root (generated-muon key of the owner, 0 for non-muon primaries, -1 unresolved), pdg."""
    s = classify(steps, parts, sector)
    mu = all_primary_muons(gen)[KEY + ["idx", "mukey"]]
    prim = parts.loc[parts.isprim == 1, KEY + ["key", "genidx"]].copy()
    prim["idx"] = prim.genidx.fillna(-1).astype(np.int64)
    prim = prim.merge(mu.astype({"idx": np.int64}), on=KEY + ["idx"], how="left")[KEY + ["key", "mukey"]]
    s = s.merge(prim.rename(columns={"key": "prim_key"}), on=KEY + ["prim_key"], how="left")
    ismu = s.cls.isin(("mu_primary", "mu_descendant")).to_numpy()
    s["root"] = np.where(ismu, s.mukey.fillna(-1).to_numpy(), 0).astype(np.int64)
    s["cls"] = np.select([s.cls == "mu_primary", s.cls == "mu_descendant"], [P, D], S).astype(np.int8)
    s = s.merge(parts[KEY + ["key", "pdg"]].rename(columns={"key": "sp"}), on=KEY + ["sp"], how="left")
    s["pdg"] = s.pdg.fillna(0).astype(np.int32)
    return s.drop(columns=["prim_key", "prim_pdg", "group", "mukey"]), {"unresolved_muon_steps": int((s.root < 0).sum())}


def tag_fast(steps, parts):
    """Tier-1b output (tier1b.sample numbering) with cls, root, pdg."""
    p = parts[KEY + ["key", "parent", "pdg", "isprim"]].rename(columns={"key": "sp"})
    s = steps.merge(p, on=KEY + ["sp"], how="left", validate="many_to_one")
    shower = s.sp.to_numpy() >= SHOWER_BASE
    prim = (s.isprim.to_numpy() == 1) & ~shower
    s["cls"] = np.select([prim, shower], [P, S], D).astype(np.int8)
    s["root"] = np.select([prim, shower], [s.sp.to_numpy(), 0], s.parent.to_numpy()).astype(np.int64)
    s["pdg"] = s.pdg.fillna(0).astype(np.int32)
    return s.drop(columns=["parent", "isprim"])


def g4_crossings(g):
    """Geant4 primary-muon steps as tier-1a crossings: one row per (event, muon, bar), sp = generated-muon key."""
    p = g[(g.cls == P) & (g.root > 0)].sort_values(KEY + ["root", "t0"], kind="stable")
    agg = {c: "first" for c in ("t0", "x0", "y0", "z0", "px0", "py0", "pz0")}
    agg.update({c: "last" for c in ("t1", "x1", "y1", "z1", "p1")})
    agg.update({"len": "sum", "edep": "sum"})
    x = p.groupby(KEY + ["root", "bar"], sort=False).agg(agg).reset_index().rename(columns={"root": "sp"})
    x["nstep"] = x.groupby(KEY).bar.transform("size")
    return x.sort_values(KEY + ["t0"], kind="stable").reset_index(drop=True)


def fast_crossings(f):
    """Fast primary-muon rows (one per crossing, sp = muon key) in tier-1a form."""
    x = f.loc[f.cls == P, KEY + STEP_COLS].sort_values(KEY + ["t0"], kind="stable").reset_index(drop=True)
    x["nstep"] = x.groupby(KEY).bar.transform("size")
    return x


def tkin(p):
    return np.sqrt(np.asarray(p, float) ** 2 + T.MASS[13] ** 2) - T.MASS[13]


def loss_to_next(x):
    """Muon kinetic-energy loss from each crossing's entry to the entry of its next crossing in time [MeV];
    LAST_CROSSING for a muon's last crossing."""
    p = np.sqrt(x.px0.to_numpy() ** 2 + x.py0.to_numpy() ** 2 + x.pz0.to_numpy() ** 2)
    mu = x.groupby(KEY + ["sp"], sort=False).ngroup().to_numpy()
    order = np.lexsort((x.t0.to_numpy(), mu))
    v = np.full(len(x), LAST_CROSSING)
    has = np.r_[mu[order][1:] == mu[order][:-1], False]
    a = order[has]
    v[a] = tkin(p[a]) - tkin(p[order[np.flatnonzero(has) + 1]])
    return v


def library_loss_to_next(lib):
    """loss_to_next of every library crossing (index xid), from the template sidecar's entry |p| and next crossing."""
    g = lib["xgeo"]
    nx = np.asarray(g["nxt"], np.int64)
    t = tkin(g["p"])
    return np.where(nx >= 0, t - t[np.maximum(nx, 0)], LAST_CROSSING)


def attached_energy(x, d):
    """Descendant energy attached to each crossing of x by the library-building rule (tier1b._build_one): a
    descendant step goes to the crossing of its event in the same bar, else to the one whose entry is nearest to it."""
    if not len(d) or not len(x):
        return np.zeros(len(x))
    dd = d[KEY + ["bar", "x0", "y0", "z0", "x1", "y1", "z1", "edep"]].reset_index(drop=True)
    dd["did"] = np.arange(len(dd))
    xx = x[KEY + ["bar", "x0", "y0", "z0"]].rename(columns={"bar": "xbar", "x0": "ex", "y0": "ey", "z0": "ez"})
    xx["xi"] = np.arange(len(x))
    m = dd.merge(xx, on=KEY)
    mid = 0.5 * (m[["x0", "y0", "z0"]].to_numpy() + m[["x1", "y1", "z1"]].to_numpy())
    dist = np.linalg.norm(mid - m[["ex", "ey", "ez"]].to_numpy(), axis=1)
    m["score"] = np.where(m.bar.to_numpy() == m.xbar.to_numpy(), -1.0, dist)
    best = m.loc[m.groupby("did").score.idxmin()]
    return np.bincount(best.xi.to_numpy(), best.edep.to_numpy(), len(x))


class _FirstDraw:
    """Generator whose first random() call returns given numbers; every other call goes to the wrapped generator."""

    def __init__(self, rng, u):
        self._rng, self._u = rng, u

    def random(self, size=None, *args, **kw):
        if self._u is not None:
            u, self._u = self._u, None
            if np.shape(u) != (np.prod(size, dtype=np.int64),):
                raise ValueError("matched draw: first random() call is not the donor draw")
            return u
        return self._rng.random(size, *args, **kw)

    def __getattr__(self, name):
        return getattr(self._rng, name)


def matched_draw(x, lib, geom, options, v_lib, v_x, rng):
    """Uniform numbers that make tier1b._desc_for_crossings pick, for each crossing of x, a library crossing of its own
    draw support (the same cell as the unmatched draw) among the MATCH_WINDOW nearest in value (v_lib[xid] vs v_x).
    Ties are broken at random by a jitter far below any physical difference."""
    lx, lo, hi = T._support(x, lib, geom, options.get("context", False), T._indexed(lib))
    sup, sid = np.unique(np.stack([lo, hi], 1), axis=0, return_inverse=True)
    sid = sid.ravel()
    size = sup[:, 1] - sup[:, 0]
    pos = np.repeat(sup[:, 0] - np.r_[0, np.cumsum(size)[:-1]], size) + np.arange(size.sum())
    seg = np.repeat(np.arange(len(sup)), size)
    span = 4e6
    clip = lambda v: np.clip(v, -1e3, 1e6) + 1e3 + 1e-4 * rng.random(len(v))
    key = seg * span + clip(v_lib[lx.xid.to_numpy()[pos]])
    order = np.argsort(key, kind="stable")
    key, pos = key[order], pos[order]
    start = np.r_[0, np.cumsum(size)[:-1]]
    q = np.searchsorted(key, sid * span + clip(np.asarray(v_x, float)))
    j = q - MATCH_WINDOW // 2 + (rng.random(len(q)) * MATCH_WINDOW).astype(np.int64)
    j = np.clip(j, start[sid], start[sid] + size[sid] - 1)
    got, want = v_lib[lx.xid.to_numpy()[pos[j]]], np.asarray(v_x, float)
    hard = want > 10.0
    print(f"# matched draw: {len(want):,} crossings, median |donor - target| {np.median(np.abs(got - want)):.3f}, "
          f"above 10: {hard.sum():,} targets, median donor/target {np.median(got[hard] / want[hard]) if hard.any() else 0:.3f},"
          f" last-crossing mismatches {int(((got == LAST_CROSSING) != (want == LAST_CROSSING)).sum())}", flush=True)
    return (pos[j] - lo + 0.5) / (hi - lo)


def redraw(x, fast_parts, gen, lib, geom, options, seed, match=None, g_desc=None):
    """Tier-1b descendants drawn for the crossings x: class D rows of a tag_fast table. match None: the default draw
    (options as given); 'k': donors matched in loss_to_next; 't': matched in the descendant energy of g_desc attached to
    each crossing. Matched draws turn the hard and follow draws off (the match replaces them)."""
    muons = fast_parts[(fast_parts.isprim == 1) & (fast_parts.key < SHOWER_BASE)]
    rng = np.random.default_rng(seed)
    if match is not None:
        options = dict(options, hard=False, follow=False)
        mrng = np.random.default_rng([*np.atleast_1d(seed), 5])
        if match == "k":
            u = matched_draw(x, lib, geom, options, library_loss_to_next(lib), loss_to_next(x), mrng)
        else:
            ld = lib["libdesc"]
            v_lib = np.bincount(ld.xid.to_numpy(), ld.edep.to_numpy(), len(lib["libxing"]))
            u = matched_draw(x, lib, geom, options, v_lib, attached_energy(x, g_desc), mrng)
        rng = _FirstDraw(rng, u)
    steps, parts, _ = T.sample(x, muons[KEY + ["key", "parent", "pdg", "isprim"]], gen, lib, geom, rng,
                               showers=False, **options)
    r = tag_fast(steps, parts)
    return r[r.cls == D].reset_index(drop=True)


def shuffle_primary(g, lib, geom, rng):
    """Geant4 table whose primary-muon steps take the dE/dx of another primary step of the same tier-1b bin (group,
    momentum, path length), at random: the muon's visible energy loses any link to its own descendants."""
    s = g.copy()
    i = np.flatnonzero((s.cls.to_numpy() == P) & (s.len.to_numpy() > 0))
    p = np.sqrt(s.px0.to_numpy()[i] ** 2 + s.py0.to_numpy()[i] ** 2 + s.pz0.to_numpy()[i] ** 2)
    ln = s.len.to_numpy()[i]
    cell = T._bin(lib["libbins"], T._group_index(geom, s.bar.to_numpy()[i]), p, ln)
    a, b = np.lexsort((rng.random(len(i)), cell)), np.lexsort((rng.random(len(i)), cell))
    dedx = s.edep.to_numpy()[i] / ln
    new = np.empty(len(i))
    new[b] = dedx[a]
    e = s.edep.to_numpy().copy()
    e[i] = new * ln
    s["edep"] = e
    return s


def build_arm(spec, sources):
    """(steps, parts) of arm spec from {source letter: tagged table}. Pure arms keep their source's row order."""
    letters = parse_arm(spec)[:3]
    used = {c for c in letters if c != "-"}
    frames = []
    pure = len(used) == 1 and "-" not in letters
    for cls, c in enumerate(letters[:1] if pure else letters):
        if c == "-":
            continue
        t = sources[c] if pure else sources[c][sources[c].cls == cls]
        off = SOURCE_OFFSET[c] if len(used) > 1 else 0
        frames.append(t.assign(sp=t.sp.to_numpy() + off) if off else t)
    if len(frames) == 1:
        s = frames[0].reset_index(drop=True)
    else:
        s = pd.concat(frames, ignore_index=True).sort_values(KEY + ["t0"], kind="stable").reset_index(drop=True)
    s = s.astype({"bar": np.int64, "sp": np.int64})
    s["nstep"] = s.groupby(KEY).bar.transform("size").astype(np.int64)
    parts = s[KEY + ["sp", "pdg"]].drop_duplicates(KEY + ["sp"]).rename(columns={"sp": "key"}).reset_index(drop=True)
    return s, parts


def cluster_table(pulses, cl, geom, steps, event_key):
    """One row per cluster: reco quantities + per-class visible energy of the steps in its bars, in time with it."""
    n = len(cl["ev"])
    columns = ("ev", "idx", "sectortype", "start", "end", "pe", "nlayers", "nbars", "sector", "owner", "owner_cls",
               "e_p", "e_d", "e_s", "l_p", "avgtime", "x", "y", "z", "npulses")
    if n == 0:
        return {k: np.zeros(0) for k in columns}
    ev = np.asarray(cl["ev"], np.int64)
    lists = ak.Array(cl["pulses"])
    count = ak.to_numpy(ak.num(lists)).astype(np.int64)
    flat = ak.to_numpy(ak.flatten(lists)).astype(np.int64)
    cid = np.repeat(np.arange(n), count)
    pev = np.asarray(pulses["ev"], np.int64)
    ue, lo = np.unique(pev, return_index=True)
    row = lo[np.searchsorted(ue, ev[cid])] + flat
    cb = pd.DataFrame({"cid": cid, "ev": ev[cid], "bar": pulses["bar"][row], "pe": pulses["pe"][row]})
    start, end = np.asarray(cl["start"], float), np.asarray(cl["end"], float)
    perbar = cb.groupby(["cid", "ev", "bar"], as_index=False, sort=False).pe.sum()
    nbars = np.bincount(perbar.cid.to_numpy(), minlength=n)
    head = perbar.sort_values(["cid", "pe"], ascending=[True, False], kind="stable").drop_duplicates("cid")
    sector = np.full(n, -1, np.int64)
    sector[head.cid.to_numpy()] = geom.sector[head.bar.to_numpy()]
    st = pd.DataFrame({"ev": event_key(steps), "bar": steps.bar.to_numpy(), "t0": steps.t0.to_numpy(),
                       "edep": steps.edep.to_numpy(), "len": steps.len.to_numpy(), "cls": steps.cls.to_numpy(),
                       "root": steps.root.to_numpy()})
    m = perbar[["cid", "ev", "bar"]].merge(st, on=["ev", "bar"])
    c = m.cid.to_numpy()
    keep = (m.t0.to_numpy() >= start[c] + GATE_NS[0]) & (m.t0.to_numpy() <= end[c] + GATE_NS[1])
    m = m[keep]
    c = m.cid.to_numpy()
    e = [np.bincount(c[m.cls.to_numpy() == k], m.edep.to_numpy()[m.cls.to_numpy() == k], n) for k in (P, D, S)]
    l_p = np.bincount(c[m.cls.to_numpy() == P], m.len.to_numpy()[m.cls.to_numpy() == P], n)
    own = m.groupby(["cid", "root"], as_index=False, sort=False).edep.sum()
    own = own.sort_values(["cid", "edep"], ascending=[True, False], kind="stable").drop_duplicates("cid")
    owner = np.full(n, -2, np.int64)
    owner[own.cid.to_numpy()] = own.root.to_numpy()
    domcls = np.stack(e).argmax(0).astype(np.int64)
    return {"ev": ev, "idx": np.asarray(cl["idx"], np.int64), "sectortype": np.asarray(cl["sectortype"], np.int64),
            "start": start, "end": end, "pe": np.asarray(cl["pe"], float),
            "nlayers": ak.to_numpy(ak.num(ak.Array(cl["layers"]))).astype(np.int64), "nbars": nbars, "sector": sector,
            "owner": owner, "owner_cls": domcls, "e_p": e[0], "e_d": e[1], "e_s": e[2], "l_p": l_p,
            "avgtime": np.asarray(cl["avgtime"], float), "x": np.asarray(cl["x"], float),
            "y": np.asarray(cl["y"], float), "z": np.asarray(cl["z"], float), "npulses": count}


def same_bar_events(g, f):
    """Event keys and, per event, whether every muon crosses the same bars in the Geant4 and the fast table."""
    cols = KEY + ["root", "bar"]
    a = g.loc[(g.cls == P), cols].drop_duplicates()
    b = f.loc[(f.cls == P), cols].drop_duplicates()
    m = a.merge(b, on=cols, how="outer", indicator=True)
    m["same"] = m._merge == "both"
    r = m.groupby(KEY, sort=False).same.all().reset_index()
    return r


def write_truth(path, g, f, gen, event_key):
    mu = all_primary_muons(gen)
    p = np.sqrt(mu.px.to_numpy() ** 2 + mu.py.to_numpy() ** 2 + mu.pz.to_numpy() ** 2)
    same = same_bar_events(g, f)
    x = g[(g.cls == P) & (g.root > 0)]
    xf = f[f.cls == P]
    ncross = x.groupby(KEY + ["root"]).bar.nunique().rename("ng").reset_index().rename(columns={"root": "mukey"})
    ncross_f = xf.groupby(KEY + ["root"]).bar.nunique().rename("nf").reset_index().rename(columns={"root": "mukey"})
    t = mu[KEY + ["mukey"]].merge(ncross, on=KEY + ["mukey"], how="left").merge(ncross_f, on=KEY + ["mukey"], how="left")
    nmu = mu.groupby(KEY).mukey.transform("size").to_numpy()
    np.savez_compressed(path, ev=event_key(mu), mukey=mu.mukey.to_numpy(), p=p, cosy=-mu.py.to_numpy() / p, nmu=nmu,
                        pdg=mu.pdg.to_numpy(), ncross_g=t.ng.fillna(0).to_numpy(), ncross_f=t.nf.fillna(0).to_numpy(),
                        same_ev=event_key(same), same=same.same.to_numpy())


def run_shard(a):
    from crvfastsim import chain, geometry as _geometry  # noqa: F401
    from crvfastsim.response import photons as PH
    from crvfastsim.response import sipm as SI
    from . import fasttables as F
    tick = time.monotonic()
    s = shard_name(a.shard)
    out = Path(a.out)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    geom, ccfg, cond, cfg, _ = F.configure(calib=a.calib, deviation=a.deviation, counters=a.counters)
    tabs, pmap = PH.load_tables(cfg.response), SI.photon_map()
    g4p, fp = a.g4.format(s=s), a.fast.format(s=s)
    gparts = tables.read(g4p, "particles", KEY + ["key", "parent", "pdg", "isprim", "genidx"])
    fparts = tables.read(fp, "particles", KEY + ["key", "parent", "pdg", "isprim"])
    gen = tables.read(fp, "gen")
    g, info = tag_g4(tables.read(g4p, "steps"), gparts, gen, geom.sector)
    f = tag_fast(tables.read(fp, "steps"), fparts)
    print(f"# shard {a.shard}: g4 {len(g):,} steps, fast {len(f):,} steps, {info}", flush=True)
    write_truth(out / "truth" / f"{s}.npz", g, f, gen, chain.event_key)
    sources = {"g": g, "f": f}
    tier1b = {}

    def library():
        if not tier1b:
            from crvfastsim import tier1b_assets as AS
            spec = AS.resolve(directory=a.tier1b_dir)
            tier1b.update(spec=spec, lib=AS.load(spec, geom))
        return tier1b["lib"], tier1b["spec"].options

    for arm in a.arms.split(","):
        path = Path(a.write_tables) / f"{arm}.{s}.root" if a.write_tables else out / arm / f"{s}.npz"
        if path.exists():
            continue
        t0 = time.monotonic()
        letters = parse_arm(arm)
        local = dict(sources)
        if letters[0] == "s":
            if "s" not in sources:
                lib, _ = library()
                sources["s"] = shuffle_primary(g, lib, geom, np.random.default_rng([a.shard, 1000, 82]))
            local["s"] = sources["s"]
        if letters[1] in "rkt":
            dk = draw_key(letters)
            if dk not in sources:
                lib, options = library()
                x = g4_crossings(g) if dk[1] == "g" else fast_crossings(f)
                match = None if dk[0] == "r" else dk[0]
                sources[dk] = redraw(x, fparts, gen, lib, geom, options, [a.shard, 1000, DRAW_SEED[dk]], match,
                                     g[g.cls == D] if match == "t" else None)
            local[letters[1]] = sources[dk]
        steps, parts = build_arm(arm, local)
        if a.write_tables:
            write_table(path, steps, parts)
            print(f"# {arm}: wrote {path} ({len(steps):,} steps), {time.monotonic() - t0:.0f} s", flush=True)
            continue
        rng = np.random.default_rng(a.shard if not letters[3] else [a.shard, letters[3]])
        D_ = F.fast_digis(steps[KEY + STEP_COLS], parts, geom, cond, cfg, rng, None, tabs, pmap)
        pulses, cl = F.fast_reco(D_, cond, geom, ccfg, fcfg=cfg.finder)
        ct = cluster_table(pulses, cl, geom, steps, chain.event_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        evk, nst = np.unique(chain.event_key(steps), return_counts=True)
        evd, ndg = np.unique(D_["ev"], return_counts=True)
        np.savez_compressed(tmp, **ct, nsteps=len(steps), ndigi=len(D_["ev"]), npulse=len(pulses["ev"]),
                            mult_ev=evk, mult_nstep=nst, mult_ndigi=ndg, mult_digi_ev=evd)
        tmp.rename(path)
        print(f"# {arm}: {len(steps):,} steps, {len(ct['pe']):,} clusters, {time.monotonic() - t0:.0f} s", flush=True)
    print(f"# shard {a.shard} done in {time.monotonic() - tick:.0f} s", flush=True)


def write_table(path, steps, parts):
    """An arm's steps in the CrvStepTableDumper layout for the stock response (art CrvStepsFromTable): one particle per
    step owner, primary muons marked primary; parents are not kept (the response does not use them)."""
    st = steps[KEY + STEP_COLS].copy()
    for c in KEY:
        st[c] = st[c].astype(np.uint32)
    for c in ("nstep", "bar", "sp"):
        st[c] = st[c].astype(np.int64)
    prim = steps.loc[steps.cls == P, KEY + ["sp"]].drop_duplicates().assign(isprim=1)
    q = parts.rename(columns={"key": "sp"}).merge(prim, on=KEY + ["sp"], how="left").rename(columns={"sp": "key"})
    q = pd.DataFrame({**{c: q[c].astype(np.uint32) for c in KEY}, "key": q.key.astype(np.int64),
                      "parent": np.full(len(q), -1, np.int64), "pdg": q.pdg.astype(np.int32),
                      "isprim": q.isprim.fillna(0).astype(np.int64)})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tables.write(str(tmp), st, q, as_ttree=True)
    tmp.rename(path)


def histogram(pe):
    """Counts in the deck's 100 bins plus overflow as a last entry."""
    pe = np.asarray(pe, float)
    over = pe >= PE_EDGES[-1]
    return np.r_[np.histogram(pe[~over], PE_EDGES)[0], over.sum()].astype(np.int64)


def run_refs(a):
    """Per-shard PE histograms of stock art valhist files, for the shards named only."""
    import uproot
    key = "Validation/CrvCoincidenceClusters_CrvCoincidenceClusterFinder_noName/PE"
    shards = shard_list(a.shards)
    out = {"shards": np.array(shards)}
    for spec in a.ref:
        name, _, pattern = spec.partition("=")
        rows = []
        for k in shards:
            with uproot.open(pattern.format(s=shard_name(k))) as f:
                v = f[key].values(flow=True).astype(np.int64)
            rows.append(np.r_[v[1:-1], v[-1]])
        out[name] = np.array(rows)
    np.savez_compressed(a.out, **out)


def deck_chi2(n1, n2):
    """Deck statistic: Poisson variance n1 + n2 over the booked bins with entries (overflow excluded)."""
    a, b = np.asarray(n1, float)[:-1], np.asarray(n2, float)[:-1]
    m = (a + b) > 0
    return float(((a[m] - b[m]) ** 2 / (a[m] + b[m])).sum() / max(int(m.sum()), 1)), int(m.sum())


def paired_terms(x, y, weight):
    """Unbiased sum over bins of weight * <mean_x * mean_y> for per-shard arrays x, y [shards, bins]; totals scale."""
    n = len(x)
    dx, dy = x - x.mean(0), y - y.mean(0)
    cov = (dx * dy).sum(0) / (n - 1) * n      # covariance of the shard sums
    return float((weight * (x.sum(0) * y.sum(0) - cov)).sum())


def signal(d, weight):
    """Noise-subtracted chi2 sum of a per-shard difference d (positive part not forced): excess over the null."""
    return paired_terms(d, d, weight)


def flip_null(d, nflip=20000, seed=20261101):
    """Paired chi2/ndf of per-shard differences d [shards, bins] with the bias factor, and its sign-flip null.

    var_b = N/(N-1) sum_s (d_sb - mean d_b)^2, D_b = sum_s d_sb; the sign flips keep the correlations between bins.
    Returns value, ndf, null mean, null sd, p (flips at or above the value; 1/(nflip+1) at best)."""
    n = len(d)
    use = (d != 0).any(0)
    if not use.any():
        return 0.0, 0, 1.0, 0.0, 1.0
    d = np.asarray(d, float)[:, use]
    ss = (d * d).sum(0)
    def value(total):
        var = n / (n - 1.0) * (ss - total * total / n)
        t2 = np.where(var > 0, total * total / np.where(var > 0, var, 1.0), 0.0)
        return t2.mean(-1) * (n - 3.0) / (n - 1.0)
    rng = np.random.default_rng(seed)
    obs = float(value(d.sum(0)))
    null = np.concatenate([value(rng.choice((-1.0, 1.0), (1000, n)) @ d) for _ in range(nflip // 1000)])
    return obs, int(use.sum()), float(null.mean()), float(null.std()), float(((null >= obs).sum() + 1) / (len(null) + 1))


def decompose(h):
    """Yates effects of the 2^3 design from {arm: [shards, bins]}; they sum exactly to fff - ggg."""
    g = h["ggg"]
    P_ = h["fgg"] - g
    D_ = h["gfg"] - g
    S_ = h["ggf"] - g
    PD = h["ffg"] - h["fgg"] - h["gfg"] + g
    PS = h["fgf"] - h["fgg"] - h["ggf"] + g
    DS = h["gff"] - h["gfg"] - h["ggf"] + g
    total = h["fff"] - g
    return {"P": P_, "D": D_, "S": S_, "PD": PD, "PS": PS, "DS": DS, "PDS": total - P_ - D_ - S_ - PD - PS - DS}, total


def jackknife(fn, n):
    """Value of fn(keep mask) on all shards and its delete-one jackknife error."""
    keep = np.ones(n, bool)
    v0 = fn(keep)
    vals = []
    for i in range(n):
        keep[i] = False
        vals.append(fn(keep))
        keep[i] = True
    vals = np.array(vals, float)
    return v0, float(np.sqrt((n - 1) / n * ((vals - vals.mean(0)) ** 2).sum(0)))


def load_clusters(out, arm, shards):
    frames = []
    for k in shards:
        p = Path(out) / arm / f"{shard_name(k)}.npz"
        with np.load(p) as z:
            d = {c: z[c] for c in z.files if z[c].ndim == 1 and not c.startswith("mult_")}
        d = pd.DataFrame(d)
        d["shard"] = k
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_truth(out, shards):
    mu, same = [], []
    for k in shards:
        with np.load(Path(out) / "truth" / f"{shard_name(k)}.npz") as z:
            mu.append(pd.DataFrame({c: z[c] for c in ("ev", "mukey", "p", "cosy", "nmu", "ncross_g", "ncross_f")}))
            same.append(pd.DataFrame({"ev": z["same_ev"], "same": z["same"]}).assign(shard=k))
        mu[-1]["shard"] = k
    return pd.concat(mu, ignore_index=True), pd.concat(same, ignore_index=True)


def match(a, b):
    """Clusters of two arms paired by truth identity: (shard, event, owner, sector) and rank in time among those."""
    ids = ["shard", "ev", "owner", "sector"]
    for t in (a, b):
        t["rank"] = t.sort_values("start", kind="stable").groupby(ids).cumcount()
    return a.merge(b, on=ids + ["rank"], how="outer", suffixes=("_a", "_b"), indicator=True)


def share_class(e_d, e_p, e_s):
    return np.digitize(e_d / np.maximum(e_p + e_d + e_s, 1e-9), [0.02, 0.05, 0.1, 0.2, 0.4])


def split_table(m, by, name):
    """Mean PE b - a per category of matched clusters with its error, and the mean PE of a."""
    g = m.groupby(by, observed=True)
    r = pd.DataFrame({"n": g.size(), "pe_a": g.pe_a.mean(), "dpe": g.dpe.mean(), "edpe": g.dpe.sem(),
                      "rms": g.dpe.std()})
    r["rel_pct"] = 100 * r.dpe / r.pe_a
    print(f"\n{name}\n{r.round(3).to_string()}")
    return {str(k): {c: float(v[c]) for c in r.columns} for k, v in r.iterrows()}


BANDS = np.array([0, 480, 640, 880, 1200, 1280, 2000, 1e9])
BAND_NAMES = ["<480", "480-640", "640-880", "880-1200", "1200-1280", "1280-2000", ">2000"]


def band(pe):
    return np.digitize(pe, BANDS) - 1


def multiplicity(out, C, shards):
    """Per arm, mean and tail of the event and cluster multiplicities that survive X1: difference to ggg, paired."""
    def per_shard(x):
        rows = []
        for k in shards:
            with np.load(Path(out) / x / f"{shard_name(k)}.npz") as z:
                ns, nd = z["mult_nstep"], z["mult_ndigi"]
                np_ = C[x].npulses[C[x].shard == k].to_numpy()
                rows.append([ns.sum(), (ns >= 20).sum(), (ns >= 50).sum(), nd.sum(), (nd >= 50).sum(), (nd >= 100).sum(),
                             np_.sum(), len(np_)])
        return np.array(rows, float)
    labels = ["steps", "ev steps>=20", "ev steps>=50", "digis", "ev digis>=50", "ev digis>=100", "pulses in clusters",
              "clusters"]
    ref = per_shard("ggg")
    n = len(shards)
    print("\nmultiplicities: total per arm, relative difference to ggg in percent with the paired error\n" + " " * 8
          + "".join(f"{x:>22s}" for x in labels))
    r = {}
    for x in C:
        c = per_shard(x)
        d = c - ref
        rel, e = d.sum(0) / ref.sum(0), np.sqrt(n * d.var(0, ddof=1)) / ref.sum(0)
        print(f"{x:8s}" + "".join(f"{t:11.0f} {100 * a_:+5.2f}+-{100 * b_:.2f}" for t, a_, b_ in zip(c.sum(0), rel, e)))
        r[x] = {l: {"n": float(t), "rel": float(a_), "e": float(b_)} for l, t, a_, b_ in zip(labels, c.sum(0), rel, e)}
    return r


FACTORIAL = ("ggg", "fff", "fgg", "gfg", "ggf", "ffg", "fgf", "gff")


def attribution(h, weight, n, verbose):
    """Share of the residual fff - ggg carried by each effect (noise-subtracted projection), and the chi2 excess that
    each hybrid arm removes, with shard jackknife errors."""
    eff, total = decompose(h)
    den = lambda k: paired_terms(total[k], total[k], weight)
    frac, removed = {}, {}
    if verbose:
        print("\nresidual share by effect (noise-subtracted projection on fff - ggg), then its overflow change")
    for name, d_ in eff.items():
        f, ef = jackknife(lambda k: paired_terms(d_[k], total[k], weight) / den(k), n)
        ov, eov = float(d_[:, -1].sum()), float(np.sqrt(n * d_[:, -1].var(ddof=1)))
        pv, pn, nm, ns, pp = flip_null(d_[:, :-1])
        frac[name] = {"f": f, "ef": ef, "overflow": ov, "e_overflow": eov, "paired": pv, "null_mean": nm, "null_sd": ns,
                      "p": pp}
        print(f"{name:5s} {f:7.3f} +- {ef:5.3f}   d(overflow) {ov:7.0f} +- {eov:4.0f}   paired chi2/ndf {pv:6.3f} "
              f"(null {nm:.2f} +- {ns:.2f}, p {pp:.1e})")
    print(f"total d(overflow) {total[:, -1].sum():.0f} +- {np.sqrt(n * total[:, -1].var(ddof=1)):.0f}")
    ref = h["ggg"]
    for label, arm in (("G4 primary", "gff"), ("G4 descendants", "fgf"), ("G4 showers", "ffg"),
                       ("G4 P+D", "ggf"), ("G4 P+S", "gfg"), ("G4 D+S", "fgg")):
        f, ef = jackknife(lambda k: 1 - signal((h[arm] - ref)[k], weight) / signal(total[k], weight), n)
        removed[label] = {"arm": arm, "removed": f, "e": ef}
        if verbose:
            print(f"chi2 excess removed by {label:15s} ({arm}): {f:6.3f} +- {ef:5.3f}")
    return frac, removed


PATHS = ("fff,grg,ggf,ggg", "fff,frf,grg,ggf,ggg", "fff,grg,gkg,ggf,ggg", "fff,grg,gtg,ggf,ggg", "fff,fkf,gkg,ggf,ggg",
         "ggf,sgf")


def path_shares(h, weight, n, path):
    """Steps of a path of arms a0 -> a1 -> ... -> ggg: each step's share of fff - ggg (noise-subtracted projection,
    the shares of a path from fff to ggg add to 1), its paired chi2/ndf, overflow and band changes."""
    total = h["fff"] - h["ggg"]
    den = lambda k: paired_terms(total[k], total[k], weight)
    out = []
    print(f"\npath {' -> '.join(path)}: share of fff - ggg, paired chi2/ndf of the step, d(overflow), d(bands)")
    for a_, b_ in zip(path[:-1], path[1:]):
        d = h[a_] - h[b_]
        f, ef = jackknife(lambda k: paired_terms(d[k], total[k], weight) / den(k), n)
        pv, _, nm, ns, pp = flip_null(d[:, :-1])
        bands = np.c_[np.add.reduceat(d[:, :-1], np.searchsorted(PE_EDGES[:-1], BANDS[:-2]), axis=1), d[:, -1]]
        db, eb = bands.sum(0), np.sqrt(n * bands.var(0, ddof=1))
        out.append({"from": a_, "to": b_, "share": f, "e_share": ef, "paired": pv, "null_sd": ns, "p": pp,
                    "d_bands": db.tolist(), "e_bands": eb.tolist()})
        print(f"  {a_} -> {b_}: share {f:6.3f} +- {ef:5.3f}  paired {pv:6.3f} (null {nm:.2f} +- {ns:.2f}, p {pp:.1e})  "
              + " ".join(f"{bn}:{x:+.0f}+-{e:.0f}" for bn, x, e in zip(BAND_NAMES, db, eb)))
    return out


def population_by(C, mu, shards, col, edges, names, arms=("fff", "grg", "gkg", "gtg", "fkf")):
    """Clusters per (PE band, owner-muon kinematic class) of each arm minus ggg, each arm classed by its own owner
    (no selection on a partner): do the tail deficit and the bulk excess sit in the same muons?"""
    ml = mu.rename(columns={"mukey": "owner"})[["shard", "ev", "owner", col]]
    def table(t):
        v = t.merge(ml, on=["shard", "ev", "owner"], how="left")[col].to_numpy()
        j = np.where(np.isfinite(v), np.digitize(v, edges), len(edges) + 1)
        c = np.zeros((len(shards), len(BAND_NAMES), len(edges) + 2), np.int64)
        np.add.at(c, (np.searchsorted(shards, t.shard.to_numpy()), band(t.pe.to_numpy()), j), 1)
        return c
    ref = table(C["ggg"])
    labels = list(names) + ["no muon"]
    r = {"labels": labels, "ggg": ref.sum(0).tolist()}
    n = len(shards)
    for x in arms:
        if x not in C:
            continue
        d = table(C[x]) - ref
        e = np.sqrt(n * d.var(0, ddof=1))
        print(f"\n{x} - ggg clusters by PE band (rows) and owner {col} (columns)\n" + " " * 12
              + "".join(f"{l:>16s}" for l in labels))
        for i, b in enumerate(BAND_NAMES):
            print(f"{b:12s}" + "".join(f"{ref[:, i, j].sum():7d}{d[:, i, j].sum():+5d}+-{e[i, j]:3.0f}"
                                       for j in range(len(labels))))
        r[x] = {"d": d.sum(0).tolist(), "e": e.tolist()}
    return r


def selections(C, out, shards):
    """Named cluster subsets for the attribution: sector groups and events whose muons cross the same bars."""
    _, same = load_truth(out, shards)
    same = same.set_index(["shard", "ev"]).same
    def in_same(t):
        ix = pd.MultiIndex.from_arrays([t.shard.to_numpy(), t.ev.to_numpy()])
        return pd.Series(same.reindex(ix).fillna(True).to_numpy(dtype=bool), index=t.index)
    out_ = [(f"sector {name}", (lambda i: (lambda t: SECTOR_GROUP[t.sector.to_numpy()] == i))(i))
            for i, name in enumerate(GROUP_NAMES)]
    return out_ + [("events with the same primary bars in G4 and fast", in_same)]


def band_table(C, shards):
    """Clusters per PE band and arm; difference to ggg with the paired shard error."""
    def counts(t):
        return np.stack([np.bincount(band(t.pe[t.shard == k].to_numpy()), minlength=len(BAND_NAMES)) for k in shards])
    n = len(shards)
    ref = counts(C["ggg"])
    r = {}
    print(f"\nband counts, difference to ggg (paired shard error)\n{'arm':8s}" + "".join(f"{b:>17s}" for b in BAND_NAMES))
    for x, t in C.items():
        c = counts(t)
        d = c - ref
        cell = [f"{c[:, i].sum():7d} {d[:, i].sum():+5d}+-{np.sqrt(n * d[:, i].var(ddof=1)):3.0f}" for i in range(c.shape[1])]
        print(f"{x:8s}" + "".join(f"{v:>17s}" for v in cell))
        r[x] = {"n": c.sum(0).tolist(), "d": d.sum(0).tolist(), "e": np.sqrt(n * d.var(0, ddof=1)).tolist()}
    return r


def quantile_shift(C, shards):
    """PE quantiles of pairs of arms: the shift q(b) - q(a) with its shard jackknife error."""
    qs = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995])
    out = {}
    for a_, b_ in (("ggg", "fff"), ("ggg", "gfg"), ("ggg", "fgg"), ("ggg", "grg"), ("g--", "f--")):
        if a_ not in C or b_ not in C:
            continue
        pa, pb = C[a_].pe.to_numpy(), C[b_].pe.to_numpy()
        sa, sb = C[a_].shard.to_numpy(), C[b_].shard.to_numpy()
        v0 = np.quantile(pb, qs) - np.quantile(pa, qs)
        jk = np.stack([np.quantile(pb[sb != k], qs) - np.quantile(pa[sa != k], qs) for k in shards])
        n = len(shards)
        e = np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))
        print(f"\nquantile shift {b_} - {a_}: " + "  ".join(f"{q:.3g}:{d:+.1f}+-{x:.1f}" for q, d, x in zip(qs, v0, e)))
        print("  quantiles of " + a_ + ": " + " ".join(f"{v:.0f}" for v in np.quantile(pa, qs)))
        out[f"{b_}-{a_}"] = {"q": qs.tolist(), "shift": v0.tolist(), "e": e.tolist(), "ref": np.quantile(pa, qs).tolist()}
    return out


def population(C, shards):
    """Clusters per (PE band, descendant energy class) for ggg and fff, each tagged by its own steps (no selection on
    the partner, so no regression to the mean), and the means of the class energies per band."""
    def table(t):
        i, j = band(t.pe.to_numpy()), np.digitize(t.e_d.to_numpy(), E_CLASS)
        c = np.zeros((len(shards), len(BAND_NAMES), len(E_CLASS) + 1), np.int64)
        np.add.at(c, (np.searchsorted(shards, t.shard.to_numpy()), i, j), 1)
        return c
    a_, b_ = table(C["ggg"]), table(C["fff"])
    d = b_ - a_
    n = len(shards)
    e = np.sqrt(n * d.var(0, ddof=1))
    print("\nfff - ggg clusters by PE band (rows) and descendant energy in the cluster's bars (columns, MeV), own tags")
    labels = ["<2", "2-5", "5-15", "15-50", ">50"]
    print(" " * 12 + "".join(f"{x:>16s}" for x in labels))
    for i, b in enumerate(BAND_NAMES):
        print(f"{b:12s}" + "".join(f"{a_[:, i, j].sum():7d}{d[:, i, j].sum():+5d}+-{e[i, j]:3.0f}" for j in range(5)))
    r = {"labels": labels, "ggg": a_.sum(0).tolist(), "d": d.sum(0).tolist(), "e": e.tolist()}
    for x in ("ggg", "fff", "grg", "gfg", "fgg"):
        if x not in C:
            continue
        t = C[x]
        g = t.groupby(band(t.pe.to_numpy()))
        m = pd.DataFrame({"n": g.size(), "pe": g.pe.mean(), "e_p": g.e_p.mean(), "e_d": g.e_d.mean(), "e_s": g.e_s.mean(),
                          "l_p": g.l_p.mean()})
        m.index = [BAND_NAMES[i] for i in m.index]
        print(f"\nmeans per band, {x}\n{m.round(3).to_string()}")
        r["means_" + x] = m.to_dict("index")
        print(f"{x}: all clusters mean e_p {t.e_p.mean():.3f} e_d {t.e_d.mean():.3f} e_s {t.e_s.mean():.3f} "
              f"l_p {t.l_p.mean():.2f} mean(pe | pe<2000) {t.pe[t.pe < 2000].mean():.2f}")
    return r


def run_report(a):
    shards = shard_list(a.shards)
    out = Path(a.out)
    arms = sorted(p.name for p in out.iterdir() if p.is_dir() and p.name != "truth"
                  and all((p / f"{shard_name(k)}.npz").exists() for k in shards))
    print(f"arms with all {len(shards)} shards: {arms}")
    C = {x: load_clusters(out, x, shards) for x in arms}
    n = len(shards)
    h = {x: np.stack([histogram(C[x].pe[C[x].shard == k]) for k in shards]) for x in arms}
    R = {"shards": shards, "arms": {}}
    ref = h["ggg"]
    bins = np.arange(ref.shape[1])
    weight = 1.0 / np.maximum(ref.sum(0) + h["fff"].sum(0), 1.0)
    weight[-1] = 0.0            # the deck's statistic has no overflow; the overflow is listed by itself
    sig0 = lambda keep: signal((h["fff"] - ref)[keep], weight)
    print(f"\n{'arm':10s}{'chi2/ndf':>9s}{'ndf':>5s}{'paired':>8s}{'null':>12s}{'p':>9s}{'excess/ndf':>11s}{'+-':>7s}"
          f"{'clusters':>9s}{'overflow':>9s}{'ov/ggg':>8s}{'+-':>7s}{'d(ov)':>7s}{'+-':>5s}")
    for x in arms:
        chi, ndf = deck_chi2(ref.sum(0), h[x].sum(0))
        d = h[x] - ref
        ex, eex = (0.0, 0.0) if x == "ggg" else jackknife(lambda k: signal(d[k], weight) / ndf, n)
        pv, pn, nm, ns, pp = flip_null(d[:, :-1])
        dov, edov = float(d[:, -1].sum()), float(np.sqrt(n * d[:, -1].var(ddof=1)))
        ov0 = float(ref[:, -1].sum())
        R["arms"][x] = {"chi2ndf": chi, "ndf": ndf, "paired": pv, "paired_ndf": pn, "null_mean": nm, "null_sd": ns, "p": pp,
                        "excess_ndf": ex, "excess_err": eex, "clusters": int(h[x].sum()),
                        "overflow": int(h[x][:, -1].sum()), "d_overflow": dov, "e_d_overflow": edov,
                        "ov_ratio": 1 + dov / ov0, "e_ov_ratio": edov / ov0,
                        "mean_pe": float(C[x].pe[C[x].pe < 2000].mean())}
        print(f"{x:10s}{chi:9.3f}{ndf:5d}{pv:8.3f}{nm:6.2f}+-{ns:4.2f}{pp:9.1e}{ex:11.3f}{eex:7.3f}{h[x].sum():9d}"
              f"{h[x][:, -1].sum():9d}{1 + dov / ov0:8.3f}{edov / ov0:7.3f}{dov:7.0f}{edov:5.0f}")
    if all(k in h for k in FACTORIAL):
        R["fractions"], R["removed"] = attribution(h, weight, n, True)
        for name, mask in selections(C, out, shards):
            hs = {x: np.stack([histogram(C[x].pe[(C[x].shard == k) & mask(C[x])]) for k in shards]) for x in FACTORIAL}
            ws = 1.0 / np.maximum(hs["ggg"].sum(0) + hs["fff"].sum(0), 1.0)
            ws[-1] = 0.0
            chi, ndf = deck_chi2(hs["ggg"].sum(0), hs["fff"].sum(0))
            print(f"\n--- {name}: fff vs ggg chi2/ndf {chi:.3f} ({ndf}), clusters {hs['ggg'].sum()}")
            R.setdefault("subsets", {})[name] = {"chi2ndf": chi, "ndf": ndf, "fractions": attribution(hs, ws, n, False)[0]}
    R["paths"] = [path_shares(h, weight, n, p.split(",")) for p in PATHS if all(x in h for x in p.split(","))]
    R["bands"] = band_table(C, shards)
    R["qq"] = quantile_shift(C, shards)
    R["population"] = population(C, shards)
    mu, same = load_truth(out, shards)
    R["population_cos"] = population_by(C, mu, shards, "cosy", [0.3, 0.6, 0.8, 0.9, 0.97],
                                        ["<0.3", "0.3-0.6", "0.6-0.8", "0.8-0.9", "0.9-0.97", ">0.97"])
    R["population_p"] = population_by(C, mu, shards, "p", [500, 1000, 2000, 5000, 20000],
                                      ["<0.5", "0.5-1", "1-2", "2-5", "5-20", ">20 GeV/c"])
    R["multiplicity"] = multiplicity(out, C, shards)
    if a.refs:
        Z = np.load(a.refs)
        idx = [list(Z["shards"]).index(k) for k in shards]
        R["refs"] = {}
        for name in Z.files:
            if name == "shards":
                continue
            r = Z[name][idx]
            rows = {}
            for x, cand in (("ggg", "ggg"), ("fff", "fff")):
                if cand not in h:
                    continue
                chi, ndf = deck_chi2(r.sum(0), h[cand].sum(0))
                pv, _, nm, ns, pp = flip_null((h[cand] - r)[:, :-1])
                dov = h[cand][:, -1] - r[:, -1]
                rows[x] = {"chi2ndf": chi, "ndf": ndf, "n_ref": int(r.sum()), "n_arm": int(h[cand].sum()),
                           "ov_ref": int(r[:, -1].sum()), "ov_arm": int(h[cand][:, -1].sum()), "paired": pv,
                           "null_mean": nm, "null_sd": ns, "p": pp, "ov_ratio": float(h[cand][:, -1].sum() / r[:, -1].sum()),
                           "e_ov_ratio": float(np.sqrt(n * dov.var(ddof=1)) / r[:, -1].sum())}
            R["refs"][name] = rows
            print(f"ref {name}: " + ", ".join(f"{x} chi2/ndf {v['chi2ndf']:.3f} clusters {v['n_ref']} vs {v['n_arm']} "
                                              f"overflow {v['ov_ref']} vs {v['ov_arm']} ratio {v['ov_ratio']:.3f} +- {v['e_ov_ratio']:.3f} "
                                              f"paired {v['paired']:.3f} (null {v['null_mean']:.2f} +- {v['null_sd']:.2f}, p {v['p']:.1e})"
                                              for x, v in rows.items()))
        stock = {nm[3:]: Z[nm][idx] for nm in Z.files if nm.startswith("st_")}
        if "ggg" in stock:
            R["stock"] = stock_table(stock, h, n)
        if "a0g" in Z.files and "a1" in Z.files:
            chi, ndf = deck_chi2(Z["a0g"][idx].sum(0), Z["a1"][idx].sum(0))
            R["refs"]["a1_vs_a0g"] = {"chi2ndf": chi, "ndf": ndf}
            pv, _, nm, ns, pp = flip_null((Z["a1"][idx] - Z["a0g"][idx])[:, :-1])
            R["refs"]["a1_vs_a0g"].update(paired=pv, null_mean=nm, null_sd=ns, p=pp)
            print(f"stock A1 vs A0g on these shards: chi2/ndf {chi:.3f} ({ndf}), paired {pv:.3f} (null {nm:.2f} +- {ns:.2f})")
    R["paired"] = paired_report(C, mu, same, h)
    if a.json:
        json.dump(R, open(a.json, "w"), indent=1)


def stock_table(stock, h, n):
    """Arms through the stock response (art CrvStepsFromTable on the arm's table): each against stock ggg, with the
    fast-response value beside it, and each stock arm against the same arm through the fast response."""
    ref = stock["ggg"]
    weight = 1.0 / np.maximum(ref.sum(0) + stock.get("fff", ref).sum(0), 1.0)
    weight[-1] = 0.0
    R = {}
    print(f"\nstock response, {n} shards: arm vs stock ggg (excess/ndf, paired, overflow ratio), then stock vs fast "
          f"response of the same arm")
    for x, v in stock.items():
        chi, ndf = deck_chi2(ref.sum(0), v.sum(0))
        d = v - ref
        ex, eex = (0.0, 0.0) if x == "ggg" else jackknife(lambda k: signal(d[k], weight) / ndf, n)
        pv, _, nm, ns, pp = flip_null(d[:, :-1])
        dov = d[:, -1]
        r = {"excess_ndf": ex, "excess_err": eex, "paired": pv, "null_sd": ns, "p": pp, "clusters": int(v.sum()),
             "ov_ratio": float(v[:, -1].sum() / ref[:, -1].sum()), "e_ov_ratio": float(np.sqrt(n * dov.var(ddof=1))
                                                                                      / ref[:, -1].sum())}
        if x in h:
            pf, _, _, nsf, ppf = flip_null((v - h[x])[:, :-1])
            r.update(vs_fast=pf, vs_fast_null_sd=nsf, vs_fast_p=ppf,
                     fast_ov_ratio=float(h[x][:, -1].sum() / v[:, -1].sum()))
        R[x] = r
        print(f"  st_{x:6s} excess {ex:6.3f} +- {eex:5.3f}  paired {pv:6.3f} (null sd {ns:.2f}, p {pp:.1e})  overflow "
              f"{r['ov_ratio']:.3f} +- {r['e_ov_ratio']:.3f}  clusters {r['clusters']}"
              + (f"  | vs fast {x}: paired {r['vs_fast']:.3f} (p {r['vs_fast_p']:.1e}), fast/stock overflow "
                 f"{r['fast_ov_ratio']:.3f}" if x in h else ""))
    return R


def attach(m, mu, same):
    ml = mu.rename(columns={"mukey": "owner"})[["shard", "ev", "owner", "p", "cosy", "nmu"]]
    m = m.merge(ml, on=["shard", "ev", "owner"], how="left")
    return m.merge(same, on=["shard", "ev"], how="left")


def paired_report(C, mu, same, h):
    R = {}
    for label, a_, b_ in (("fff vs ggg", "ggg", "fff"), ("g fast D vs ggg", "ggg", "gfg"), ("fast P vs ggg", "ggg", "fgg"),
                          ("primary only f vs g", "g--", "f--"), ("redraw vs ggg", "ggg", "grg"),
                          ("redraw vs fast D", "gfg", "grg"), ("descendants added, g", "g--", "ggg"),
                          ("descendants added, f", "f--", "fff"), ("loss-matched vs ggg", "ggg", "gkg"),
                          ("energy-matched vs ggg", "ggg", "gtg"), ("muon dE/dx shuffled vs ggf", "ggf", "sgf"),
                          ("fast loss-matched vs ggg", "ggg", "fkf")):
        if a_ not in C or b_ not in C:
            continue
        m = match(C[a_].copy(), C[b_].copy())
        tot = len(m)
        both = m[m._merge == "both"].copy()
        print(f"\n=== {label}: units {tot:,}, matched {len(both):,}, only {a_} {(m._merge == 'left_only').sum():,}, "
              f"only {b_} {(m._merge == 'right_only').sum():,}")
        both["dpe"] = both.pe_b - both.pe_a
        both = attach(both, mu, same)
        both["pgroup"] = pd.cut(both.p, [0, 500, 1000, 2000, 5000, 20000, 1e9])
        both["cosgroup"] = pd.cut(both.cosy, [-1.01, 0.3, 0.6, 0.8, 0.9, 0.97, 1.01])
        both["multi"] = both.nmu > 1
        both["sectname"] = both.sector
        both["lay"] = both.nlayers_a
        both["share"] = share_class(both.e_d_a.to_numpy(), both.e_p_a.to_numpy(), both.e_s_a.to_numpy())
        both["peband"] = band(0.5 * (both.pe_a.to_numpy() + both.pe_b.to_numpy()))
        both["own"] = np.sign(both.owner)
        d = {"units": tot, "matched": len(both)}
        for by, nm in (("sectname", "sector"), ("lay", "layers (arm a)"), ("pgroup", "muon p [MeV/c]"),
                       ("cosgroup", "cos(vertical)"), ("multi", "multi-muon event"), ("same", "same bars in both"),
                       ("own", "owner sign (1 muon, 0 shower)"), ("peband", "mean PE band"),
                       ("share", "descendant share of arm a (<2, 5, 10, 20, 40, >40 %)")):
            d[nm] = split_table(both, by, f"{label}: dPE by {nm}")
        R[label] = d
        bi, bj = band(both.pe_a.to_numpy()), band(both.pe_b.to_numpy())
        mig = np.zeros((len(BAND_NAMES), len(BAND_NAMES)), np.int64)
        np.add.at(mig, (bi, bj), 1)
        d["migration"] = mig.tolist()
        print(f"migration {a_} (rows) to {b_} (columns), bands {BAND_NAMES}\n{mig}")
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("shard")
    r.add_argument("--shard", type=int, required=True)
    r.add_argument("--arms", required=True)
    r.add_argument("--g4", required=True)
    r.add_argument("--fast", required=True)
    r.add_argument("--calib", required=True)
    r.add_argument("--deviation")
    r.add_argument("--counters")
    r.add_argument("--tier1b-dir")
    r.add_argument("--out", required=True)
    r.add_argument("--write-tables", help="write each arm's step table to DIR/<arm>.<shard>.root instead of running "
                   "the fast response (stock response input)")
    r.set_defaults(fn=run_shard)
    x = sub.add_parser("refs")
    x.add_argument("--ref", action="append", required=True, metavar="NAME=PATTERN")
    x.add_argument("--shards", required=True)
    x.add_argument("--out", required=True)
    x.set_defaults(fn=run_refs)
    p = sub.add_parser("report")
    p.add_argument("out")
    p.add_argument("--shards", required=True)
    p.add_argument("--refs")
    p.add_argument("--json")
    p.set_defaults(fn=run_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
