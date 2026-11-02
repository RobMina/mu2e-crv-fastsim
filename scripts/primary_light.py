"""Primary-muon light chain, Geant4 vs stage-C transport: per-crossing visible energy and step structure.

    primary_light.py extract G4TABLE FASTC OUT.npz      one shard: per-crossing arrays of both arms (matched by key, bar)
    primary_light.py report OUT.json A.npz [B.npz ...]  mean, median, MPV, width by momentum, path, sector with
                                                        leave-one-shard-out jackknife errors
    primary_light.py structure X.npz OUT.npz CALIB      one shard: step structure, position and expected photons per arm
    primary_light.py structure-report A.npz [B.npz ...] arm ratios with jackknife errors
    primary_light.py arms G4TABLE OUTDIR                primary-only step tables: g4p, g4c (one row per crossing),
                                                        g4s (each crossing cut into SPLIT equal steps)
    primary_light.py birks                              one bar crossing in n Urban steps with Birks per step
    primary_light.py hybrid G4TABLE FASTC OUT.root      G4 steps with the primary muons' steps taken from FASTC
    primary_light.py pe CHAIN.root OUT.npz              cluster and per-bar pulse PE of a chain output
    primary_light.py pe-report OUT.json REF ARM [ARM..] PEDIR    cluster-PE ratios to the reference arm, per shard errors
    primary_light.py share OUT.json PDIR PREF PARM FDIR FREF FARM primary-muon share of the full mean cluster-PE shift

Arms in a PE directory are files ARM.SHARD.npz written by `pe`; paths are absolute when run in a container.

Crossing: one muon's steps in one bar, summed. G4 arm: mu_primary class of the step table.
"""
import json
import sys

import numpy as np
import pandas as pd

from crvfastsim import tables
from crvfastsim.analyses.census import classify

K = tables.KEY
SECTOR = np.repeat(np.arange(11), [256, 128, 128] + [8] * 8)
GROUP = np.array([0, 1, 2] + [3] * 8)[SECTOR]
GROUP_NAMES = ("EX", "T1", "T2", "M")
COLS = K + ["bar", "sp", "edep", "len", "px0", "py0", "pz0", "p1", "t0", "t1", "x0", "y0", "z0", "x1", "y1", "z1"]
PBINS = (0, 150, 300, 1000, 3000, 10000, 1e9)
LBINS = (0, 10, 19, 21, 25, 35, 60, 1e9)
VBINS = np.linspace(0, 12, 1201)   # visible dE/dx [MeV/cm], 0.01 wide


def crossings(s):
    """One row per (key, muon, bar): summed energy and path, first entry, last exit, number of steps."""
    s = s.sort_values(K + ["sp", "bar", "t0"], kind="stable")
    s["pin"] = np.sqrt(s.px0 ** 2 + s.py0 ** 2 + s.pz0 ** 2)
    g = s.groupby(K + ["sp", "bar"], sort=False)
    c = g.agg(edep=("edep", "sum"), len=("len", "sum"), pin=("pin", "first"), p1=("p1", "last"), t0=("t0", "first"),
              t1=("t1", "last"), x0=("x0", "first"), y0=("y0", "first"), z0=("z0", "first"), x1=("x1", "last"),
              y1=("y1", "last"), z1=("z1", "last"), nst=("edep", "size"),
              edmax=("edep", "max")).reset_index()
    c["chord"] = np.sqrt((c.x1 - c.x0) ** 2 + (c.y1 - c.y0) ** 2 + (c.z1 - c.z0) ** 2)
    return c


def g4_table(path):
    s = tables.read(path, "steps", COLS)
    s = classify(s, tables.read(path, "particles", K + ["key", "parent", "pdg"]), SECTOR)
    return s[s.cls == "mu_primary"].drop(columns=["cls", "group", "prim_key", "prim_pdg"])


def extract(g4, fast, out):
    a = crossings(g4_table(g4))
    b = crossings(tables.read(fast, "steps", COLS))
    m = a.merge(b, on=K + ["sp", "bar"], how="left", suffixes=("", "_f"), indicator=True)
    a["match"] = (m["_merge"] == "both").to_numpy()
    a["e_f"], a["len_f"], a["pin_f"] = m.edep_f.to_numpy(), m.len_f.to_numpy(), m.pin_f.to_numpy()
    arr = {"g_" + c: a[c].to_numpy() for c in a.columns}
    arr.update({"f_" + c: b[c].to_numpy() for c in b.columns})
    np.savez_compressed(out, **arr)


def stats(h, n, s):
    """(mean, median, MPV, half 16-84 width, FWHM) of a VBINS histogram h with exact count n and sum s."""
    mid = 0.5 * (VBINS[1:] + VBINS[:-1])
    cum = np.cumsum(h) / max(h.sum(), 1)
    qs = np.interp([0.16, 0.5, 0.84], cum, VBINS[1:])
    # MPV: parabola in log counts over the region above 60 % of the peak, on a 5-bin running mean
    sm = np.convolve(h, np.ones(9) / 9, mode="same")
    pk = int(np.argmax(sm))
    lo, hi = pk, pk
    while lo > 0 and sm[lo] > 0.6 * sm[pk]:
        lo -= 1
    while hi < len(sm) - 1 and sm[hi] > 0.6 * sm[pk]:
        hi += 1
    x, y = mid[lo:hi + 1], np.log(np.maximum(h[lo:hi + 1], 0.5))
    c = np.polyfit(x, y, 3)
    xx = np.linspace(x[0], x[-1], 501)
    mpv = float(xx[np.argmax(np.polyval(c, xx))])
    half = np.flatnonzero(sm >= 0.5 * sm[pk])
    return np.array([s / max(n, 1), qs[1], mpv, 0.5 * (qs[2] - qs[0]), mid[half[-1]] - mid[half[0]]])


def category_masks(c, pin, ln):
    grp = GROUP[c["bar"]]
    out = {"all": np.ones(len(grp), bool)}
    for i, n in enumerate(GROUP_NAMES):
        out[f"group {n}"] = grp == i
    for lo, hi in zip(PBINS[:-1], PBINS[1:]):
        out[f"p {lo}-{hi}"] = (pin >= lo) & (pin < hi)
    for lo, hi in zip(LBINS[:-1], LBINS[1:]):
        out[f"path {lo}-{hi}"] = (ln >= lo) & (ln < hi)
    for sid in range(11):
        out[f"sector {sid}"] = SECTOR[c["bar"]] == sid
    mip = (pin > 1000) & (ln > 19) & (ln < 22)
    out["MIP normal (p>1, L 19-22)"] = mip
    for i, n in enumerate(GROUP_NAMES):
        out[f"MIP normal {n}"] = mip & (grp == i)
    return out


def report(out, paths):
    H = {}
    for ish, p in enumerate(paths):
        z = np.load(p)
        for arm, pre in (("g4", "g_"), ("fast", "f_")):
            c = {k[2:]: z[k] for k in z.files if k.startswith(pre)}
            keep = c["p1"] >= 1.0       # exiting crossings: stops are a separate population
            ln, pin = c["len"], c["pin"]
            v = 10 * c["edep"] / np.maximum(ln, 1e-3)
            for name, m in category_masks(c, pin, ln).items():
                m = m & keep
                e = H.setdefault((arm, name), [np.zeros((len(paths), len(VBINS) - 1)), np.zeros(len(paths)),
                                               np.zeros(len(paths)), np.zeros(len(paths))])
                e[0][ish] = np.histogram(v[m], VBINS)[0]
                e[1][ish] = m.sum()
                e[2][ish] = v[m].sum()
                e[3][ish] = c["edep"][m].sum()
    res = {}
    names = [k[1] for k in H if k[0] == "g4"]
    ns = len(paths)
    for name in names:
        row = {}
        vals = {}
        for arm in ("g4", "fast"):
            h, n, s, ed = H[(arm, name)]
            full = stats(h.sum(0), n.sum(), s.sum())
            jk = np.array([stats(h.sum(0) - h[i], n.sum() - n[i], s.sum() - s[i]) for i in range(ns)])
            vals[arm] = (full, jk, ed.sum() / max(n.sum(), 1), n.sum())
        rj = vals["fast"][1] / vals["g4"][1] - 1
        r = vals["fast"][0] / vals["g4"][0] - 1
        err = np.sqrt((ns - 1) / ns * ((rj - rj.mean(0)) ** 2).sum(0))
        for i, q in enumerate(("mean", "median", "mpv", "width", "fwhm")):
            row[q] = dict(g4=float(vals["g4"][0][i]), fast=float(vals["fast"][0][i]), pct=float(100 * r[i]),
                          err=float(100 * err[i]))
        row["n_g4"], row["n_fast"] = int(vals["g4"][3]), int(vals["fast"][3])
        res[name] = row
    json.dump(res, open(out, "w"), indent=0)
    print(f"{'category':28s}{'n g4':>9s}{'mean g4':>9s}" + "".join(f"{q + ' f/g-1 %':>18s}" for q in ("mean", "median", "mpv", "width")))
    for name, r in res.items():
        print(f"{name:28s}{r['n_g4']:9d}{r['mean']['g4']:9.3f}" + "".join(
            f"{r[q]['pct']:+10.2f} +-{r[q]['err']:5.2f}" for q in ("mean", "median", "mpv", "width")))


METRICS = ("n", "edep", "len", "chord", "nst2", "e_nst2", "ph_scint", "ph_cks", "ph_ckf", "ph_mid", "ph_3pt",
           "ph_scint_e0", "stops", "e_stops", "v_abs", "v2", "w", "w2", "x_abs")


def bin_prob(L, x, y, z, s_type="scint"):
    """Scintillation arrival probability at local (x, y, z), 0 outside the scintillator (stock substep rule)."""
    from crvfastsim.response import photons as PH
    E = L.edges
    nb = {k: len(v) - 1 for k, v in E.items()}
    insc = PH._inside_scint(L, x, y, z)
    bx, by, bz = PH.find_bin(E["x"], np.abs(x)), PH.find_bin(E["y"], y), PH.find_bin(E["z"], z)
    ok = insc & (bx >= 0) & (by >= 0) & (bz >= 0)
    b = bz + by * nb["z"] + bx * nb["y"] * nb["z"]
    return np.where(ok, L.tables["scint"].prob[np.where(ok, b, 0)].astype(float), 0.0)


def expected_photons(steps, geom, cfg, tabs, rng):
    """Per crossing, per-type arrivals summed over SiPMs along the chord (0.1 mm substeps, stock counts), and the same
    scintillation sum from one chord point and from three points."""
    from crvfastsim.response import photons as PH
    n = len(steps)
    pdg = np.full(n, 13)
    dev = np.zeros(geom.nbars * 4)
    out = {k: np.zeros(n) for k in ("ph_scint", "ph_cks", "ph_ckf", "ph_mid", "ph_3pt", "ph_scint_e0")}
    for L, r in PH.iter_rows(steps, pdg, geom, cfg, tabs, dev, rng):
        nsub, step = r["nsub"], r["step"]
        E = L.edges
        nb = {k: len(v) - 1 for k, v in E.items()}
        for a in range(0, len(nsub), 2000):
            idx = np.arange(a, min(a + 2000, len(nsub)))
            ri = np.repeat(idx, nsub[idx])
            k = np.arange(len(ri)) - np.repeat(np.r_[0, np.cumsum(nsub[idx])[:-1]], nsub[idx])
            frac = (k + 0.5) / nsub[ri]
            p = r["s"][ri] + r["d"][ri] * frac[:, None]
            x, y, z = p[:, 0], p[:, 1], p[:, 2]
            ps = bin_prob(L, x, y, z) * r["dev"][ri]
            np.add.at(out["ph_scint"], step[ri], ps * r["n_scint"][ri])
            ncount = steps.edep.to_numpy()[step[ri]]
            np.add.at(out["ph_scint_e0"], step[ri], ps * ncount / nsub[ri])
            insc = PH._inside_scint(L, x, y, z)
            bx, by, bz = PH.find_bin(E["x"], np.abs(x)), PH.find_bin(E["y"], y), PH.find_bin(E["z"], z)
            bb = PH.find_bin(E["beta"], r["beta"][ri])
            ok = insc & (bx >= 0) & (by >= 0) & (bz >= 0) & (bb >= 0)
            c = bb + bz * nb["beta"] + by * nb["z"] * nb["beta"] + bx * nb["y"] * nb["z"] * nb["beta"]
            pc = np.where(ok, L.tables["ckscint"].prob[np.where(ok, c, 0)].astype(float), 0.0) * r["dev"][ri]
            np.add.at(out["ph_cks"], step[ri], pc * r["n_ck_s"][ri])
            infb, rr, phi = PH._inside_fiber(L, x, y, z, r["d"][ri, 0], r["d"][ri, 1])
            infb &= ~insc
            bth, bph, br = (PH.find_bin(E["theta"], r["theta"][ri]), PH.find_bin(E["phi"], phi), PH.find_bin(E["r"], rr))
            fk = infb & (bb >= 0) & (bth >= 0) & (bph >= 0) & (br >= 0) & (bz >= 0)
            f = bz + br * nb["z"] + bph * nb["r"] * nb["z"] + bth * nb["phi"] * nb["r"] * nb["z"] \
                + bb * nb["theta"] * nb["phi"] * nb["r"] * nb["z"]
            pf = np.where(fk, L.tables["ckfiber"].prob[np.where(fk, f, 0)].astype(float), 0.0) * r["dev"][ri]
            np.add.at(out["ph_ckf"], step[ri], pf * r["n_ck_f"][ri])
        e = steps.edep.to_numpy()[step]
        mid = r["s"] + 0.5 * r["d"]
        out["ph_mid"][:] += np.bincount(step, e * bin_prob(L, *mid.T) * r["dev"], n)
        pts = [bin_prob(L, *(r["s"] + f * r["d"]).T) for f in (1 / 6, 0.5, 5 / 6)]
        out["ph_3pt"][:] += np.bincount(step, e * np.mean(pts, 0) * r["dev"], n)
    return out


def structure(xfile, out, calib):
    from valhist import fasttables
    from crvfastsim.response import photons as PH
    geom, _, _, cfg, _ = fasttables.configure(calib=calib)
    tabs = PH.load_tables(cfg.response)
    z = np.load(xfile)
    res = np.zeros((2, 5, len(METRICS)))
    for ia, pre in enumerate(("g_", "f_")):
        c = pd.DataFrame({k[2:]: z[k] for k in z.files if k.startswith(pre) and z[k].ndim == 1})
        c["px0"], c["py0"], c["pz0"] = (c.pin * (c.x1 - c.x0) / c.chord.clip(lower=1e-6),
                                        c.pin * (c.y1 - c.y0) / c.chord.clip(lower=1e-6),
                                        c.pin * (c.z1 - c.z0) / c.chord.clip(lower=1e-6))
        c = c.reset_index(drop=True)
        ex = expected_photons(c, geom, cfg.response, tabs, np.random.default_rng(1))
        bar = c.bar.to_numpy().astype(np.int64)
        g = GROUP[bar]
        l2w = geom.local_to_world[bar]
        mid = 0.5 * (c[["x0", "y0", "z0"]].to_numpy() + c[["x1", "y1", "z1"]].to_numpy()) - geom.center[bar]
        loc = mid[np.arange(len(c))[:, None], l2w]
        hw, hl = geom.halflen[bar[0]], None
        stop = c.p1.to_numpy() < 1.0
        v = dict(n=np.ones(len(c)), edep=c.edep, len=c.len, chord=c.chord, nst2=c.nst > 1, e_nst2=c.edep * (c.nst > 1),
                 stops=stop, e_stops=c.edep * stop, v_abs=np.abs(loc[:, 1]), v2=loc[:, 1] ** 2, w=loc[:, 2],
                 w2=loc[:, 2] ** 2, x_abs=np.abs(loc[:, 0]), **ex)
        for j, m in enumerate(METRICS):
            a = np.asarray(v[m], float)
            for ig in range(4):
                res[ia, ig, j] = a[g == ig].sum()
            res[ia, 4, j] = a.sum()
    np.savez(out, res=res)


def structure_report(paths):
    R = np.array([np.load(p)["res"] for p in paths])     # [shard, arm, group, metric]
    ns = len(R)
    T = R.sum(0)
    print("metric per crossing (ph_*: arrivals summed over SiPMs per crossing); ratio fast/G4 - 1 [%] with jackknife error")
    names = list(GROUP_NAMES) + ["all"]
    for ig, gn in enumerate(names):
        print(f"--- {gn}: crossings G4 {T[0, ig, 0]:.0f} fast {T[1, ig, 0]:.0f}")
        for j, m in enumerate(METRICS):
            g4, f = T[0, ig, j] / T[0, ig, 0], T[1, ig, j] / T[1, ig, 0]
            rj = [((T[1, ig, j] - R[i, 1, ig, j]) / (T[1, ig, 0] - R[i, 1, ig, 0]))
                  / ((T[0, ig, j] - R[i, 0, ig, j]) / (T[0, ig, 0] - R[i, 0, ig, 0])) - 1 for i in range(ns)]
            rj = np.array(rj)
            e = np.sqrt((ns - 1) / ns * ((rj - rj.mean()) ** 2).sum())
            print(f"  {m:12s} {g4:12.5g} {f:12.5g}  {100 * (f / g4 - 1) if g4 else 0:+7.3f} +- {100 * e:5.3f}")


SPLIT = 4
HYB_KEY = 1 << 40


def arms(g4, outdir):
    """Primary-muon step tables of one shard: the G4 steps, collapsed to one straight step per crossing, and split."""
    s = tables.read(g4, "steps")
    parts = tables.read(g4, "particles")
    c = classify(s, parts[K + ["key", "parent", "pdg"]], SECTOR)
    keep = (c.cls == "mu_primary").to_numpy()
    s = s[keep].reset_index(drop=True)
    prim = s[K + ["sp"]].drop_duplicates().rename(columns={"sp": "key"})
    parts = parts.merge(prim, on=K + ["key"], how="inner")
    tables.write(f"{outdir}/g4p.root", s, parts, as_ttree=True)
    o = s.sort_values(K + ["sp", "bar", "t0"], kind="stable")
    g = o.groupby(K + ["sp", "bar"], sort=False)
    first = g.first().reset_index()
    for col in ("x1", "y1", "z1", "t1", "p1"):
        first[col] = g[col].last().to_numpy()
    first["edep"], first["len"] = g.edep.sum().to_numpy(), g.len.sum().to_numpy()
    tables.write(f"{outdir}/g4c.root", first[s.columns], parts, as_ttree=True)
    f = first[s.columns]
    idx = np.repeat(np.arange(len(f)), SPLIT)
    d = f.iloc[idx].reset_index(drop=True)
    k = np.tile(np.arange(SPLIT), len(f)) / SPLIT
    h = 1.0 / SPLIT
    b = f.iloc[idx]
    for col in ("x", "y", "z"):
        a0, a1 = b[f"{col}0"].to_numpy(), b[f"{col}1"].to_numpy()
        d[f"{col}0"], d[f"{col}1"] = a0 + (a1 - a0) * k, a0 + (a1 - a0) * (k + h)
    t0, t1 = b.t0.to_numpy(), b.t1.to_numpy()
    d["t0"], d["t1"] = t0 + (t1 - t0) * k, t0 + (t1 - t0) * (k + h)
    d["edep"], d["len"] = b.edep.to_numpy() * h, b.len.to_numpy() * h
    pin = np.sqrt(b.px0 ** 2 + b.py0 ** 2 + b.pz0 ** 2).to_numpy()
    ein = np.sqrt(pin ** 2 + 105.6584 ** 2)
    eout = np.sqrt(b.p1.to_numpy() ** 2 + 105.6584 ** 2)
    e0, e1 = ein + (eout - ein) * k, ein + (eout - ein) * (k + h)
    ps, pe = (np.sqrt(np.maximum(v ** 2 - 105.6584 ** 2, 0.0)) for v in (e0, e1))
    for col in ("px0", "py0", "pz0"):
        d[col] = b[col].to_numpy() * ps / np.maximum(pin, 1e-9)
    d["p1"] = pe
    tables.write(f"{outdir}/g4s.root", d, parts, as_ttree=True)


def hybrid(g4, fast, out):
    """Full G4 table with every primary-muon step replaced by the stage-C steps of the same event. Stage-C keys are
    muon ranks, not G4 SimParticle keys, so the fast muons enter as new particles at HYB_KEY + rank."""
    s = tables.read(g4, "steps")
    parts = tables.read(g4, "particles")
    c = classify(s, parts[K + ["key", "parent", "pdg"]], SECTOR)
    keep = (c.cls != "mu_primary").to_numpy()
    f = tables.read(fast, "steps")
    fp = tables.read(fast, "particles")
    f["sp"] = f.sp + HYB_KEY
    fp["key"] = fp.key + HYB_KEY
    for col in s.columns:
        if col not in f.columns:
            f[col] = 0
    for col in parts.columns:
        if col not in fp.columns:
            fp[col] = 0
    o = pd.concat([s[keep], f[s.columns].astype(s.dtypes.to_dict())], ignore_index=True)
    q = pd.concat([parts, fp[parts.columns].astype(parts.dtypes.to_dict())], ignore_index=True)
    tables.write(out, o, q, as_ttree=True)


def pe_tables(chain, out):
    import uproot
    with uproot.open(chain) as f:
        cl = f["crvdump/clusters"].arrays(["subrun", "event", "pe", "sectortype", "start", "end"], library="np")
        pu = f["crvdump/pulses"].arrays(["subrun", "event", "bar", "sipm", "pe", "time", "seq"], library="np")
    ev = (cl["subrun"].astype(np.int64) << 20) | cl["event"].astype(np.int64)
    pev = (pu["subrun"].astype(np.int64) << 20) | pu["event"].astype(np.int64)
    np.savez_compressed(out, ev=ev, cpe=cl["pe"], sec=cl["sectortype"], pev=pev, bar=pu["bar"], sipm=pu["sipm"],
                        ppe=pu["pe"], ptime=pu["time"], pseq=pu["seq"])


PE_EDGES = np.r_[np.arange(0, 2000.1, 40), 1e7]
PE_BANDS = ((0, 200), (200, 480), (480, 640), (640, 800), (800, 880), (880, 1200), (1200, 2000), (2000, 1e7))


def pe_report(out, ref, arms_, pedir):
    """Per arm against the reference arm on the same shards: cluster count, mean PE below 2000, overflow, ratio per PE
    band, per-bar PE per pulse; chi2/ndf with the paired per-shard variance of the 40 PE histogram."""
    import glob
    import os
    shards = sorted(os.path.basename(p).split(".")[1] for p in glob.glob(f"{pedir}/{ref}.*.npz"))
    shards = [s for s in shards if all(os.path.exists(f"{pedir}/{a}.{s}.npz") for a in arms_)]
    if len(shards) < 3:
        sys.exit(f"pe-report: {len(shards)} common shards for {ref} {arms_} in {pedir}")

    def load(a):
        hs, ms, pp, st = [], [], [], []
        for s in shards:
            z = np.load(f"{pedir}/{a}.{s}.npz")
            c, sec = z["cpe"], z["sec"]
            hs.append(np.histogram(c, PE_EDGES)[0])
            st.append(np.bincount(sec.astype(np.int64).clip(0, 15), minlength=16))
            m = c < 2000
            w = (c >= 400) & (c < 1400)
            ms.append((m.sum(), c[m].sum(), w.sum(), c[w].sum()))
            bk = z["ppe"]
            ok = z["pseq"] == 0
            pp.append((ok.sum(), bk[ok].sum()))
        return np.array(hs, float), np.array(ms), np.array(pp, float), np.array(st, float)

    R = load(ref)
    res = {"shards": len(shards), "ref": ref}
    print(f"{len(shards)} shards; reference {ref}: {R[0].sum():.0f} clusters, mean PE<2000 {R[1][:, 1].sum() / R[1][:, 0].sum():.2f}")
    for a in arms_:
        A = load(a)
        ns = len(shards)

        def jack(f):
            full = f(A, R)
            jk = np.array([f(tuple(np.delete(x, i, 0) for x in A), tuple(np.delete(x, i, 0) for x in R)) for i in range(ns)])
            return full, np.sqrt((ns - 1) / ns * ((jk - jk.mean(0)) ** 2).sum(0))

        mean = jack(lambda u, v: u[1][:, 1].sum() / u[1][:, 0].sum() / (v[1][:, 1].sum() / v[1][:, 0].sum()) - 1)
        cnt = jack(lambda u, v: u[0].sum() / v[0].sum() - 1)
        bulk = jack(lambda u, v: u[1][:, 3].sum() / u[1][:, 2].sum() / (v[1][:, 3].sum() / v[1][:, 2].sum()) - 1)
        over = jack(lambda u, v: u[0][:, -1].sum() / v[0][:, -1].sum() - 1)
        pulse = jack(lambda u, v: u[2][:, 1].sum() / u[2][:, 0].sum() / (v[2][:, 1].sum() / v[2][:, 0].sum()) - 1)
        rows = {}
        for lo, hi in PE_BANDS:
            b = (PE_EDGES[:-1] >= lo) & (PE_EDGES[:-1] < hi)
            r = jack(lambda u, v: u[0][:, b].sum() / u[0].sum() / (v[0][:, b].sum() / v[0].sum()) - 1)
            rows[f"{lo}-{hi}"] = (float(r[0]), float(r[1]))
        secs = {}
        for t in np.flatnonzero(R[3].sum(0) >= 100):
            r = jack(lambda u, v: u[3][:, t].sum() / v[3][:, t].sum() - 1)
            secs[int(t)] = (float(r[0]), float(r[1]))
        d = A[0] - R[0]
        var = ns / (ns - 1) * ((d - d.mean(0)) ** 2).sum(0)
        tot = d.sum(0)
        use = (R[0].sum(0) >= 25) & (var > 0)
        chi2 = float((tot[use] ** 2 / var[use]).sum() / use.sum())
        res[a] = dict(mean_pct=100 * float(mean[0]), mean_err=100 * float(mean[1]), bulk_pct=100 * float(bulk[0]),
                      bulk_err=100 * float(bulk[1]), overflow_pct=100 * float(over[0]), overflow_err=100 * float(over[1]), count_pct=100 * float(cnt[0]),
                      count_err=100 * float(cnt[1]), pulse_pct=100 * float(pulse[0]), pulse_err=100 * float(pulse[1]),
                      chi2ndf=chi2, ndf=int(use.sum()), bands=rows, count_by_sectortype=secs)
        print(f"{a:8s} clusters {100 * cnt[0]:+.2f} +- {100 * cnt[1]:.2f} %, mean PE<2000 {100 * mean[0]:+.3f} +- {100 * mean[1]:.3f} %, "
              f"mean 400-1400 {100 * bulk[0]:+.3f} +- {100 * bulk[1]:.3f} %, overflow {100 * over[0]:+.2f} +- {100 * over[1]:.2f} %, "
              f"seq-0 pulse PE {100 * pulse[0]:+.3f} +- {100 * pulse[1]:.3f} %, chi2/ndf {chi2:.2f} ({use.sum()})")
        print("     shape ratio-1 [%]: " + "  ".join(f"{k}: {100 * v[0]:+.2f}+-{100 * v[1]:.2f}" for k, v in rows.items()))
        print("     count by sector type, ratio-1 [%]: " + "  ".join(f"{k}: {100 * v[0]:+.2f}+-{100 * v[1]:.2f}"
                                                                    for k, v in secs.items()))
    json.dump(res, open(out, "w"), indent=0)


def pe_sums(pedir, arm, shards, lo, hi):
    """Per shard: summed PE and number of clusters with lo <= PE < hi."""
    out = np.zeros((len(shards), 2))
    for i, s in enumerate(shards):
        c = np.load(f"{pedir}/{arm}.{s}.npz")["cpe"]
        c = c[(c >= lo) & (c < hi)]
        out[i] = c.sum(), len(c)
    return out


SHARE_WINDOWS = ((0.0, 2000.0), (400.0, 1400.0))


def share(out, pdir, pref, parm, fdir, fref, farm):
    """Share of the full-simulation mean cluster-PE shift carried by the primary muons. Cluster PE is close to additive
    in primary and descendant light, so the primary's part of the fast/G4 shift of the mean is w * dp, with w the ratio
    of mean primary-only to mean full cluster PE (G4) and dp the fast/G4 mean shift of primary-only clusters. Cluster
    counts are reported beside it. Jackknife over common shards."""
    import glob
    import os

    def have(d, a):
        return {os.path.basename(p).split(".")[1] for p in glob.glob(f"{d}/{a}.*.npz")}
    shards = sorted(have(pdir, pref) & have(pdir, parm) & have(fdir, fref) & have(fdir, farm))
    if len(shards) < 3:
        sys.exit(f"share: {len(shards)} common shards")
    ns = len(shards)
    res = {"shards": ns, "primary": [pdir, pref, parm], "full": [fdir, fref, farm]}
    names = ("w", "dp", "dfull", "primary_part", "share", "count_primary", "count_full")
    for lo, hi in SHARE_WINDOWS:
        P, Pa, F, Fa = (pe_sums(d, a, shards, lo, hi) for d, a in ((pdir, pref), (pdir, parm), (fdir, fref), (fdir, farm)))

        def q(m):
            mean = [x[m, 0].sum() / x[m, 1].sum() for x in (P, Pa, F, Fa)]
            w, dp, df = mean[0] / mean[2], mean[1] / mean[0] - 1, mean[3] / mean[2] - 1
            return np.array([w, dp, df, w * dp, w * dp / df, Pa[m, 1].sum() / P[m, 1].sum() - 1,
                             Fa[m, 1].sum() / F[m, 1].sum() - 1])
        full = q(np.ones(ns, bool))
        jk = np.array([q(np.arange(ns) != i) for i in range(ns)])
        err = np.sqrt((ns - 1) / ns * ((jk - jk.mean(0)) ** 2).sum(0))
        tag = f"{lo:.0f}-{hi:.0f} PE"
        res[tag] = {n: [float(full[i]), float(err[i])] for i, n in enumerate(names)}
        res[tag]["share_abs_95"] = float(abs(full[4]) + 1.96 * err[4])
        print(f"{tag}: {ns} shards, w {full[0]:.4f} +- {err[0]:.4f}, mean shift primary {100 * full[1]:+.3f} +- "
              f"{100 * err[1]:.3f} %, full {100 * full[2]:+.3f} +- {100 * err[2]:.3f} %, primary part {100 * full[3]:+.3f} +- "
              f"{100 * err[3]:.3f} %, share {100 * full[4]:+.1f} +- {100 * err[4]:.1f} % (|share| < "
              f"{100 * res[tag]['share_abs_95']:.1f} % at 95 %); counts primary {100 * full[5]:+.2f} +- {100 * err[5]:.2f} %, "
              f"full {100 * full[6]:+.2f} +- {100 * err[6]:.2f} %")
    json.dump(res, open(out, "w"), indent=0)


def birks_table():
    """Mean, median, MPV and width of the visible energy of one crossing, one step vs n steps with Birks per step."""
    from mu2efastsim import eloss, materials
    mat = materials.POLYSTYRENE
    rng = np.random.default_rng(1)
    N, L, cut = 150000, 20.0, mat.cuts_e_MeV["crv"]
    print(f"kB {mat.birks_mm_per_MeV} mm/MeV, L {L} mm, cut {cut} MeV, {N} crossings per row")
    print(f"{'p MeV/c':>8s}{'steps':>10s}{'dep MeV':>9s}{'vis MeV':>9s}{'vis/dep':>9s}{'median':>9s}{'mpv~':>9s}{'q16-84/2':>9s}{'vis/one-1 %':>12s}")

    def line(p, tag, dep, vis, base):
        q = np.percentile(vis, [16, 50, 84])
        h, e = np.histogram(vis, np.linspace(0, 8, 801))
        sm = np.convolve(h, np.ones(15) / 15, mode="same")
        print(f"{p:8.0f}{tag:>10s}{dep.mean():9.4f}{vis.mean():9.4f}{vis.mean() / dep.mean():9.4f}{q[1]:9.4f}"
              f"{e[np.argmax(sm)]:9.3f}{(q[2] - q[0]) / 2:9.4f}{100 * (vis.mean() / base - 1):+12.3f}")

    for p in (300.0, 1000.0, 3000.0, 10000.0):
        pp = np.full(N, p)
        dep = eloss.sample_loss(rng, mat, pp, np.full(N, L), np.full(N, cut))
        vis = eloss.birks(mat, dep, np.full(N, L))
        base = vis.mean()
        line(p, "1", dep, vis, base)
        for n in (2, 3, 5, 10, 20, 50):
            d = eloss.sample_loss(rng, mat, np.full(N * n, p), np.full(N * n, L / n), np.full(N * n, cut)).reshape(N, n)
            v = eloss.birks(mat, d, np.full((N, n), L / n))
            line(p, str(n), d.sum(1), v.sum(1), base)
        # delta-limited: a step ends at each delta above the cut, positions uniform along the crossing
        lam = eloss.delta_rate_per_mm(mat, pp, cut) * L
        nd = rng.poisson(lam)
        mx = int(nd.max()) + 1
        u = np.sort(np.where(np.arange(mx)[None, :] < nd[:, None], rng.random((N, mx)), 1.0), axis=1)
        edges = np.c_[np.zeros(N), u, np.ones(N)]
        ln = np.diff(edges, axis=1) * L
        use = ln.ravel() > 0
        d = np.zeros(ln.size)
        d[use] = eloss.sample_loss(rng, mat, np.full(use.sum(), p), ln.ravel()[use], np.full(use.sum(), cut))
        v = np.where(use, eloss.birks(mat, d, np.where(use, ln.ravel(), 1.0)), 0.0)
        line(p, "delta", d.reshape(ln.shape).sum(1), v.reshape(ln.shape).sum(1), base)


if __name__ == "__main__":
    if sys.argv[1] == "birks":
        birks_table()
    elif sys.argv[1] == "arms":
        arms(*sys.argv[2:4])
    elif sys.argv[1] == "hybrid":
        hybrid(*sys.argv[2:5])
    elif sys.argv[1] == "pe":
        pe_tables(*sys.argv[2:4])
    elif sys.argv[1] == "pe-report":
        pe_report(sys.argv[2], sys.argv[3], sys.argv[4:-1], sys.argv[-1])
    elif sys.argv[1] == "share":
        share(*sys.argv[2:9])
    elif sys.argv[1] == "extract":
        extract(*sys.argv[2:5])
    elif sys.argv[1] == "structure":
        structure(*sys.argv[2:5])
    elif sys.argv[1] == "structure-report":
        structure_report(sys.argv[2:])
    else:
        report(sys.argv[2], sys.argv[3:])
