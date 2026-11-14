"""
Piecewise Linear Regression — Streamlit App
Run:      streamlit run piecewise_app.py
Requires: pip install streamlit numpy pandas "scipy>=1.7" matplotlib
"""
import io
import os
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import differential_evolution

BIG = 1e30  # penalty value for infeasible breakpoint sets


# ----------------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------------
def read_xy(file, has_header: bool, xcol: int, ycol: int):
    """Read a CSV and return sorted, NaN-free (x, y) arrays."""
    file.seek(0)
    df = pd.read_csv(file, header=0 if has_header else None, sep=None, engine="python")
    if df.shape[1] <= max(xcol, ycol):
        raise ValueError(f"File has only {df.shape[1]} columns (needs column index {max(xcol, ycol)})")
    x = pd.to_numeric(df.iloc[:, xcol], errors="coerce")
    y = pd.to_numeric(df.iloc[:, ycol], errors="coerce")
    m = x.notna() & y.notna()
    x, y = x[m].to_numpy(float), y[m].to_numpy(float)
    order = np.argsort(x)
    return x[order], y[order]


def _segment_ids(x, bps):
    """Segment index (0..K) of each x: x < bp1 -> 0, bp1 <= x < bp2 -> 1, ..."""
    return np.searchsorted(bps, x, side="right")


def _min_points_ok(seg, K, min_pts):
    return np.bincount(seg, minlength=K + 1).min() >= min_pts


def _design_continuous(x, bps):
    cols = [np.ones_like(x), x] + [np.maximum(0.0, x - b) for b in bps]
    return np.column_stack(cols)


def _sse_given_bps(x, y, bps, continuous, min_pts):
    """Fast SSE for a candidate breakpoint set (used by the optimizer)."""
    bps = np.sort(bps)
    K = len(bps)
    seg = _segment_ids(x, bps)
    if not _min_points_ok(seg, K, min_pts):
        return BIG
    if continuous:
        X = _design_continuous(x, bps)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ beta
        return float(r @ r)
    sse = 0.0
    for j in range(K + 1):
        m = seg == j
        X = np.column_stack([np.ones(m.sum()), x[m]])
        beta = np.linalg.lstsq(X, y[m], rcond=None)[0]
        r = y[m] - X @ beta
        sse += float(r @ r)
    return sse


def _fit_given_bps(x, y, bps, continuous):
    """
    Full linear fit for fixed breakpoints. Returns per-segment slope/intercept
    plus the *unscaled* variances (multiply by sigma^2 for standard errors).
    """
    bps = np.sort(bps)
    K = len(bps)
    seg = _segment_ids(x, bps)
    slopes = np.zeros(K + 1)
    ints = np.zeros(K + 1)
    var_s = np.zeros(K + 1)
    var_a = np.zeros(K + 1)

    if continuous:
        X = _design_continuous(x, bps)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        C = np.linalg.pinv(X.T @ X)
        Ls = np.zeros((K + 1, K + 2))
        La = np.zeros((K + 1, K + 2))
        for j in range(K + 1):
            Ls[j, 1] = 1.0
            La[j, 0] = 1.0
            for k in range(j):  # hinge terms active in segment j
                Ls[j, 2 + k] = 1.0
                La[j, 2 + k] = -bps[k]
        slopes, ints = Ls @ beta, La @ beta
        var_s = np.einsum("ij,jk,ik->i", Ls, C, Ls)
        var_a = np.einsum("ij,jk,ik->i", La, C, La)
    else:
        for j in range(K + 1):
            m = seg == j
            X = np.column_stack([np.ones(m.sum()), x[m]])
            beta = np.linalg.lstsq(X, y[m], rcond=None)[0]
            C = np.linalg.pinv(X.T @ X)
            ints[j], slopes[j] = beta
            var_a[j], var_s[j] = C[0, 0], C[1, 1]

    fit = ints[seg] + slopes[seg] * x
    return dict(bps=bps, seg=seg, slopes=slopes, ints=ints,
                var_s=var_s, var_a=var_a, fit=fit)


def fit_one(x, y, bp_specs, continuous=True, optimize=True, min_pts=3, maxiter=300):
    """
    bp_specs: list of (bp_min, bp_max, bp_init)
    Returns a dict with tables, curve data and metrics; contains 'error' on failure.
    """
    n, K = len(x), len(bp_specs)
    xmin, xmax = float(x.min()), float(x.max())
    if n < min_pts * (K + 1):
        return {"error": f"Only {n} points; need at least {min_pts * (K + 1)} "
                         f"({min_pts} per segment x {K + 1} segments)."}

    lo = np.clip([s[0] for s in bp_specs], xmin, xmax).astype(float)
    hi = np.clip([s[1] for s in bp_specs], xmin, xmax).astype(float)
    init = np.clip([s[2] for s in bp_specs], lo, hi).astype(float)
    if optimize and np.any(hi <= lo):
        bad = int(np.argmax(hi <= lo)) + 1
        return {"error": f"Breakpoint {bad}: search range lies outside this file's x-range "
                         f"[{xmin:g}, {xmax:g}]."}

    # --- find breakpoints ---
    if optimize:
        res = differential_evolution(
            lambda b: _sse_given_bps(x, y, b, continuous, min_pts),
            bounds=list(zip(lo, hi)), x0=init, seed=0, maxiter=int(maxiter),
            popsize=20, tol=1e-10, polish=bool(continuous),
        )
        bps = np.sort(res.x)
    else:
        bps = np.sort(init)

    seg = _segment_ids(x, bps)
    if not _min_points_ok(seg, K, min_pts):
        return {"error": "No feasible breakpoint set: a segment would contain fewer than "
                         f"{min_pts} points. Adjust the ranges or lower 'minimum points'."}

    f = _fit_given_bps(x, y, bps, continuous)
    fit = f["fit"]
    resid = y - fit
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())

    # --- metrics ---
    k = (2 + K if continuous else 2 * (K + 1)) + (K if optimize else 0)
    dof = n - k
    sigma2 = sse / dof if dof > 0 else np.nan
    r2 = 1 - sse / sst if sst > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / dof if dof > 0 and np.isfinite(r2) else np.nan
    ll = n * np.log(max(sse, 1e-300) / n)
    metrics = dict(n=n, params=k, SSE=sse, RMSE=float(np.sqrt(sse / n)), R2=r2,
                   adj_R2=adj, AIC=ll + 2 * k, BIC=ll + k * np.log(n))

    # --- segment table ---
    edges = np.concatenate([[xmin], f["bps"], [xmax]])
    rows = []
    for j in range(K + 1):
        m = seg == j
        yj, fj = y[m], fit[m]
        ssr_j = float(((yj - fj) ** 2).sum())
        sst_j = float(((yj - yj.mean()) ** 2).sum())
        rows.append({
            "Segment": j + 1,
            "x_start": edges[j], "x_end": edges[j + 1], "n_points": int(m.sum()),
            "slope": f["slopes"][j], "slope_err": np.sqrt(f["var_s"][j] * sigma2),
            "intercept": f["ints"][j], "intercept_err": np.sqrt(f["var_a"][j] * sigma2),
            "R2_segment": 1 - ssr_j / sst_j if (m.sum() >= 3 and sst_j > 0) else np.nan,
            "RMSE_segment": float(np.sqrt(ssr_j / m.sum())),
        })
    seg_df = pd.DataFrame(rows)

    # --- breakpoint table ---
    brow = []
    for j, b in enumerate(f["bps"]):
        yl = f["ints"][j] + f["slopes"][j] * b
        yr = f["ints"][j + 1] + f["slopes"][j + 1] * b
        brow.append({
            "Breakpoint": j + 1, "x_bp": b, "y_left": yl, "y_right": yr,
            "slope_left": f["slopes"][j], "slope_right": f["slopes"][j + 1],
            "slope_change": f["slopes"][j + 1] - f["slopes"][j],
        })
    bp_df = pd.DataFrame(brow)

    curve = pd.DataFrame({"x": x, "y": y, "fit": fit, "residual": resid, "segment": seg + 1})
    return dict(segments=seg_df, breakpoints=bp_df, curve=curve, metrics=metrics,
                bps=f["bps"], continuous=continuous)


def make_figure(x, y, res, bp_specs, title, show_ranges=True, optimized=True):
    seg_df, bp_df, curve = res["segments"], res["breakpoints"], res["curve"]
    cmap = plt.get_cmap("tab10")
    K = len(bp_df)
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
    seg_ids = curve["segment"].to_numpy() - 1
    ax.scatter(x, y, s=12, c=[cmap(i % 10) for i in seg_ids], alpha=0.65, zorder=2)

    for j, r in seg_df.iterrows():
        xs = np.array([r.x_start, r.x_end])
        ax.plot(xs, r.intercept + r.slope * xs, color=cmap(j % 10), lw=2.4, zorder=4,
                label=f"S{j + 1}: slope={r.slope:.4g}, intercept={r.intercept:.4g}")
    for j, r in bp_df.iterrows():
        for a in (ax, axr):
            a.axvline(r.x_bp, color="k", ls="--", lw=0.9, zorder=3)
        ax.annotate(f"bp{j + 1}={r.x_bp:.4g}", (r.x_bp, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=8)
        if show_ranges and optimized:
            lo, hi, _ = sorted(bp_specs, key=lambda s: s[2])[j]
            for a in (ax, axr):
                a.axvspan(lo, hi, color="0.5", alpha=0.10, zorder=1)

    ax.set_ylabel("y")
    ax.set_title(f"{title}   (R² = {res['metrics']['R2']:.5f})", pad=18)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(alpha=0.25)
    axr.axhline(0, color="0.4", lw=0.8)
    axr.scatter(x, curve["residual"], s=8, c=[cmap(i % 10) for i in seg_ids])
    axr.set_xlabel("x")
    axr.set_ylabel("Residual")
    axr.grid(alpha=0.25)
    return fig


def fig_to_bytes(fig, fmt, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")  # opens correctly in Excel


def build_zip(results: list) -> bytes:
    buf = io.BytesIO()
    seg_all, bp_all, summary = [], [], []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            if "error" in r:
                continue
            s = r["stem"]
            zf.writestr(f"{s}/{s}_segments.csv", df_to_csv_bytes(r["segments"]))
            zf.writestr(f"{s}/{s}_breakpoints.csv", df_to_csv_bytes(r["breakpoints"]))
            zf.writestr(f"{s}/{s}_curves.csv", df_to_csv_bytes(r["curve"]))
            zf.writestr(f"{s}/{s}_fit.png", r["png"])
            zf.writestr(f"{s}/{s}_fit.svg", r["svg"])
            zf.writestr(f"{s}/{s}_fit.pdf", r["pdf"])
            for lst, df in ((seg_all, r["segments"]), (bp_all, r["breakpoints"])):
                d = df.copy()
                d.insert(0, "File", r["name"])
                lst.append(d)
            summary.append({"File": r["name"], **r["metrics"]})
        if summary:
            zf.writestr("all_segments.csv", df_to_csv_bytes(pd.concat(seg_all, ignore_index=True)))
            zf.writestr("all_breakpoints.csv", df_to_csv_bytes(pd.concat(bp_all, ignore_index=True)))
            zf.writestr("summary.csv", df_to_csv_bytes(pd.DataFrame(summary)))
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Piecewise Linear Regression", layout="wide")
    st.title("📐 Piecewise Linear Regression")
    st.caption("Fit a series of straight-line segments joined at breakpoints "
               "(continuous) or fitted independently (discontinuous).")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("① Upload data")
        files = st.file_uploader("Select one or more CSV files (x, y columns)",
                                 type=["csv", "txt"], accept_multiple_files=True)
        has_header = st.checkbox("First row is a header", value=True)
        c1, c2 = st.columns(2)
        xcol = c1.number_input("x column index", 0, 50, 0, help="0-based")
        ycol = c2.number_input("y column index", 0, 50, 1)

        st.header("④ Model options")
        continuous = st.checkbox("Continuous (segments joined at breakpoints)", value=True,
                                 help="Unchecked = each segment is an independent line, "
                                      "so jumps at breakpoints are allowed.")
        mode = st.radio("Breakpoints", ["Optimize within ranges", "Fix at initial values"])
        min_pts = st.number_input("Minimum points per segment", 2, 1000, 3)
        maxiter = st.number_input("Optimizer max iterations", 20, 5000, 300, step=50,
                                  disabled=(mode != "Optimize within ranges"))

        st.header("⑤ Figure options")
        dpi = st.slider("PNG resolution (dpi)", 72, 600, 200)
        show_ranges = st.checkbox("Show breakpoint search ranges", value=True)

    if not files:
        st.info("👈 Upload one or more CSV files in the sidebar to begin.")
        return

    # ---------------- Read data ----------------
    data = {}
    for f in files:
        try:
            x, y = read_xy(f, has_header, int(xcol), int(ycol))
            if len(x) < 4:
                raise ValueError("fewer than 4 valid data points")
            data[f.name] = (x, y)
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ Could not read {f.name}: {e}")
    if not data:
        return

    # ---------------- Step 2: breakpoints ----------------
    st.header("② Breakpoints")
    first = next(iter(data))
    x0, _ = data[first]
    K = int(st.number_input("Number of breakpoints (segments = breakpoints + 1)", 1, 15, 2))

    sig = (K, first, float(x0.min()), float(x0.max()))
    if st.session_state.get("bp_sig") != sig:
        edges = np.linspace(x0.min(), x0.max(), K + 2)
        inner = edges[1:-1]
        half = (edges[1] - edges[0]) / 2
        st.session_state["bp_default"] = pd.DataFrame({
            "bp_min": inner - half, "bp_max": inner + half, "bp_init": inner})
        st.session_state["bp_sig"] = sig

    st.markdown("Each row is one breakpoint. `bp_min`/`bp_max` bound where the optimizer may "
                "place it, and `bp_init` is the starting guess (it is the fixed location if "
                "*Fix at initial values* is selected).")
    bp_df = st.data_editor(
        st.session_state["bp_default"], key=f"bp_editor_{hash(sig)}", num_rows="fixed",
        use_container_width=True,
        column_config={c: st.column_config.NumberColumn(c, format="%.6g")
                       for c in ("bp_min", "bp_max", "bp_init")},
    )

    valid, specs = True, []
    for i, r in bp_df.iterrows():
        lo, hi, b = r["bp_min"], r["bp_max"], r["bp_init"]
        if pd.isna(lo) or pd.isna(hi):
            st.error(f"Breakpoint {i + 1}: bp_min / bp_max cannot be empty")
            valid = False
            continue
        if lo >= hi:
            st.error(f"Breakpoint {i + 1}: bp_min must be smaller than bp_max")
            valid = False
            continue
        if pd.isna(b):
            b = (lo + hi) / 2
        if not (lo <= b <= hi):
            st.error(f"Breakpoint {i + 1}: bp_init must lie within [{lo:g}, {hi:g}]")
            valid = False
            continue
        specs.append((float(lo), float(hi), float(b)))
    if valid:
        s_sorted = sorted(specs, key=lambda s: s[2])
        if any(a[1] > b[0] for a, b in zip(s_sorted[:-1], s_sorted[1:])):
            st.warning("Some breakpoint ranges overlap. This is allowed, but breakpoints "
                       "may end up very close together.")

    with st.expander("🔍 Data preview with breakpoint ranges", expanded=True):
        pv = st.selectbox("Preview file", list(data.keys()))
        px, py = data[pv]
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(px, py, ".", ms=3, color="0.3")
        if valid:
            for lo, hi, b in specs:
                ax.axvspan(lo, hi, color="tab:blue", alpha=0.15)
                ax.axvline(b, color="tab:blue", ls="--", lw=1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

    # ---------------- Step 3: run ----------------
    st.header("③ Run regression")
    optimize = mode.startswith("Optimize")
    if st.button("🚀 Fit all files", type="primary", disabled=not valid):
        results = []
        prog = st.progress(0.0, text="Fitting…")
        for k, (name, (x, y)) in enumerate(data.items()):
            res = fit_one(x, y, specs, continuous=continuous, optimize=optimize,
                          min_pts=int(min_pts), maxiter=int(maxiter))
            res["name"], res["stem"] = name, os.path.splitext(name)[0]
            if "error" not in res:
                fig = make_figure(x, y, res, specs, name, show_ranges, optimize)
                res["png"] = fig_to_bytes(fig, "png", dpi)
                res["svg"] = fig_to_bytes(fig, "svg")
                res["pdf"] = fig_to_bytes(fig, "pdf")
                plt.close(fig)
            results.append(res)
            prog.progress((k + 1) / len(data), text=f"Finished {name}")
        prog.empty()
        st.session_state["pw_results"] = results

    # ---------------- Results ----------------
    results = st.session_state.get("pw_results")
    if not results:
        return
    st.header("④ Results")
    if any("error" not in r for r in results):
        st.download_button("📦 Download everything (ZIP: figures + tables)", build_zip(results),
                           file_name="piecewise_results.zip", mime="application/zip")

    tabs = st.tabs([r["name"] for r in results])
    for k, (tab, r) in enumerate(zip(tabs, results)):
        with tab:
            if "error" in r:
                st.error(r["error"])
                continue
            m = r["metrics"]
            cols = st.columns(5)
            cols[0].metric("R²", f"{m['R2']:.5f}")
            cols[1].metric("Adj. R²", "—" if np.isnan(m["adj_R2"]) else f"{m['adj_R2']:.5f}")
            cols[2].metric("RMSE", f"{m['RMSE']:.4g}")
            cols[3].metric("AIC", f"{m['AIC']:.2f}")
            cols[4].metric("BIC", f"{m['BIC']:.2f}")

            st.image(r["png"], use_container_width=True)
            d1, d2, d3 = st.columns(3)
            d1.download_button("🖼 PNG", r["png"], f"{r['stem']}_fit.png", "image/png", key=f"png{k}")
            d2.download_button("🖼 SVG", r["svg"], f"{r['stem']}_fit.svg", "image/svg+xml", key=f"svg{k}")
            d3.download_button("🖼 PDF", r["pdf"], f"{r['stem']}_fit.pdf", "application/pdf", key=f"pdf{k}")

            st.subheader("Segments")
            st.dataframe(r["segments"].style.format(
                {c: "{:.5g}" for c in r["segments"].columns
                 if c not in ("Segment", "n_points")}),
                use_container_width=True, hide_index=True)
            st.subheader("Breakpoints")
            st.dataframe(r["breakpoints"].style.format(
                {c: "{:.5g}" for c in r["breakpoints"].columns if c != "Breakpoint"}),
                use_container_width=True, hide_index=True)
            st.caption("Standard errors are conditional on the fitted breakpoint locations.")

            t1, t2, t3 = st.columns(3)
            t1.download_button("📄 Segments (CSV)", df_to_csv_bytes(r["segments"]),
                               f"{r['stem']}_segments.csv", "text/csv", key=f"s{k}")
            t2.download_button("📄 Breakpoints (CSV)", df_to_csv_bytes(r["breakpoints"]),
                               f"{r['stem']}_breakpoints.csv", "text/csv", key=f"b{k}")
            t3.download_button("📄 Fitted curve data (CSV)", df_to_csv_bytes(r["curve"]),
                               f"{r['stem']}_curves.csv", "text/csv", key=f"c{k}")


if __name__ == "__main__":
    main()
