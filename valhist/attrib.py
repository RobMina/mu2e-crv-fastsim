"""Attribution of the cluster-PE residual between fast and Geant4 steps to the step classes that make the clusters.

A step table has three classes: the primary muon itself (P), the muon's descendants (D) and everything from non-muon
primaries (S). An arm takes each class from one source and runs the same fast response and reco on the union:

    g  Geant4 steps            f  fast steps (stage C + tier 1b)       -  class left out
    r  (class D only) tier 1b drawn on the Geant4 primary crossings

An arm is three letters P, D, S plus an optional @tag for an independent response seed: ggg, fff, gfg (Geant4 primary,
fast descendants, Geant4 showers), grg, g--, ggg@2. The pure arms keep the table's row order, so fff equals the
`fasttables chain` output of the same seed. Rows of one event are joined by (run, subrun, event); a class keeps its
own absolute positions, so in a hybrid the descendants sit around the muon they were made for (the r source does not
have that leak; `same_bars` marks the events whose muons cross the same bars in both sources).

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
SOURCE_OFFSET = {"g": 0, "f": 1 << 40, "r": 2 << 40}
STEP_COLS = ["nstep", "bar", "sp", "t0", "t1", "x0", "y0", "z0", "x1", "y1", "z1", "px0", "py0", "pz0", "p1", "len",
             "edep"]
PE_EDGES = np.linspace(0.0, 2000.0, 101)
GATE_NS = (-200.0, 100.0)  # step t0 relative to the cluster's [start, end]; retuned in shard tests, see report
SECTORS = np.array(["T1", "T2", "EX"], dtype=object)


def parse_arm(spec):
    """'gfg@2' -> ('g', 'f', 'g', 2)."""
    letters, _, tag = spec.partition("@")
    if len(letters) != 3 or letters[0] not in "gf-" or letters[1] not in "gfr-" or letters[2] not in "gf-":
        raise ValueError(f"bad arm {spec!r}")
    return letters[0], letters[1], letters[2], int(tag or 0)


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


def redraw(g, fast_parts, gen, lib, geom, options, seed):
    """Tier-1b descendants drawn for the Geant4 primary crossings: class D rows of a tag_fast table."""
    x = g4_crossings(g)
    muons = fast_parts[(fast_parts.isprim == 1) & (fast_parts.key < SHOWER_BASE)]
    steps, parts, _ = T.sample(x, muons[KEY + ["key", "parent", "pdg", "isprim"]], gen, lib, geom,
                               np.random.default_rng([seed, 1000, 77]), showers=False, **options)
    r = tag_fast(steps, parts)
    return r[r.cls == D].reset_index(drop=True)


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
               "e_p", "e_d", "e_s", "l_p", "avgtime", "x", "y", "z")
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
            "y": np.asarray(cl["y"], float), "z": np.asarray(cl["z"], float)}


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
    for arm in a.arms.split(","):
        path = out / arm / f"{s}.npz"
        if path.exists():
            continue
        t0 = time.monotonic()
        letters = parse_arm(arm)
        if letters[1] == "r" and "r" not in sources:
            from crvfastsim import tier1b_assets as AS
            spec = AS.resolve(directory=a.tier1b_dir)
            lib = AS.load(spec, geom)
            sources["r"] = redraw(g, fparts, gen, lib, geom, spec.options, a.shard)
        steps, parts = build_arm(arm, sources)
        rng = np.random.default_rng(a.shard if not letters[3] else [a.shard, letters[3]])
        D_ = F.fast_digis(steps[KEY + STEP_COLS], parts, geom, cond, cfg, rng, None, tabs, pmap)
        pulses, cl = F.fast_reco(D_, cond, geom, ccfg, fcfg=cfg.finder)
        ct = cluster_table(pulses, cl, geom, steps, chain.event_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **ct, nsteps=len(steps), ndigi=len(D_["ev"]), npulse=len(pulses["ev"]))
        tmp.rename(path)
        print(f"# {arm}: {len(steps):,} steps, {len(ct['pe']):,} clusters, {time.monotonic() - t0:.0f} s", flush=True)
    print(f"# shard {a.shard} done in {time.monotonic() - tick:.0f} s", flush=True)


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
            d = {c: z[c] for c in z.files if z[c].ndim == 1}
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
    print(f"\n{'arm':10s}{'chi2/ndf':>9s}{'ndf':>5s}{'excess/ndf':>12s}{'+-':>7s}{'clusters':>10s}{'overflow':>9s}"
          f"{'d(ov) vs ggg':>14s}{'+-':>7s}")
    for x in arms:
        chi, ndf = deck_chi2(ref.sum(0), h[x].sum(0))
        d = h[x] - ref
        ex, eex = (0.0, 0.0) if x == "ggg" else jackknife(lambda k: signal(d[k], weight) / ndf, n)
        dov, edov = float(d[:, -1].sum()), float(np.sqrt(n * d[:, -1].var(ddof=1)))
        R["arms"][x] = {"chi2ndf": chi, "ndf": ndf, "excess_ndf": ex, "excess_err": eex, "clusters": int(h[x].sum()),
                        "overflow": int(h[x][:, -1].sum()), "d_overflow": dov, "e_d_overflow": edov,
                        "mean_pe": float(C[x].pe[C[x].pe < 2000].mean())}
        print(f"{x:10s}{chi:9.3f}{ndf:5d}{ex:12.3f}{eex:7.3f}{h[x].sum():10d}{h[x][:, -1].sum():9d}{dov:14.0f}{edov:7.0f}")
    if all(k in h for k in ("fgg", "gfg", "ggf", "ffg", "fgf", "gff", "fff")):
        eff, total = decompose(h)
        den = lambda k: paired_terms(total[k], total[k], weight)
        R["fractions"] = {}
        print(f"\nresidual fraction by effect (noise-subtracted projection on fff - ggg): overflow share in the last column")
        for name, d_ in eff.items():
            f, ef = jackknife(lambda k: paired_terms(d_[k], total[k], weight) / den(k), n)
            ov = float(d_[:, -1].sum())
            eov = float(np.sqrt(n * d_[:, -1].var(ddof=1)))
            R["fractions"][name] = {"f": f, "ef": ef, "overflow": ov, "e_overflow": eov}
            print(f"{name:5s} {f:7.3f} +- {ef:5.3f}   d(overflow) {ov:7.0f} +- {eov:4.0f}")
        R["removed"] = {}
        for label, arm in (("all G4 primary", "gff"), ("all G4 descendants", "fgf"), ("all G4 showers", "ffg"),
                           ("P+D G4", "ggf"), ("P+S G4", "gfg"), ("D+S G4", "fgg")):
            f, ef = jackknife(lambda k: 1 - signal((h[arm] - ref)[k], weight) / signal(total[k], weight), n)
            R["removed"][label] = {"removed": f, "e": ef}
            print(f"chi2 excess removed by {label:20s} ({arm}): {f:6.3f} +- {ef:5.3f}")
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
                rows[x] = {"chi2ndf": chi, "ndf": ndf, "n_ref": int(r.sum()), "n_arm": int(h[cand].sum()),
                           "ov_ref": int(r[:, -1].sum()), "ov_arm": int(h[cand][:, -1].sum())}
            R["refs"][name] = rows
            print(f"ref {name}: " + ", ".join(f"{x} chi2/ndf {v['chi2ndf']:.3f} clusters {v['n_ref']} vs {v['n_arm']} "
                                              f"overflow {v['ov_ref']} vs {v['ov_arm']}" for x, v in rows.items()))
        if "a0g" in Z.files and "a1" in Z.files:
            chi, ndf = deck_chi2(Z["a0g"][idx].sum(0), Z["a1"][idx].sum(0))
            R["refs"]["a1_vs_a0g"] = {"chi2ndf": chi, "ndf": ndf}
            print(f"stock A1 vs A0g on these shards: chi2/ndf {chi:.3f} ({ndf})")
    mu, same = load_truth(out, shards)
    R["paired"] = paired_report(C, mu, same, h)
    print(f"\nband counts (ggg, fff, ratio, pull)")
    if a.json:
        json.dump(R, open(a.json, "w"), indent=1)


def attach(m, mu, same):
    ml = mu.rename(columns={"mukey": "owner"})[["shard", "ev", "owner", "p", "cosy", "nmu"]]
    m = m.merge(ml, on=["shard", "ev", "owner"], how="left")
    return m.merge(same, on=["shard", "ev"], how="left")


def paired_report(C, mu, same, h):
    R = {}
    for label, a_, b_ in (("fff vs ggg", "ggg", "fff"), ("g fast D vs ggg", "ggg", "gfg"), ("fast P vs ggg", "ggg", "fgg"),
                          ("primary only f vs g", "g--", "f--")):
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
                       ("own", "owner sign (1 muon, 0 shower)"), ("peband", "mean PE band")):
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
