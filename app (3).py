"""
Multi-Gaussian Peak Fitting — Streamlit App
Run:       streamlit run app.py
Requires:  pip install streamlit numpy pandas scipy matplotlib
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
from scipy.optimize import curve_fit

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.3548


# ----------------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------------
def build_model(n_peaks: int, offset: bool):
    """Return a model function for n summed Gaussians (optional constant baseline). Parameter order: A1,mu1,s1,...,[c]"""

    def model(x, *p):
        y = np.zeros_like(x, dtype=float)
        for i in range(n_peaks):
            A, mu, s = p[3 * i: 3 * i + 3]
            y += A * np.exp(-((x - mu) ** 2) / (2.0 * s ** 2))
        if offset:
            y += p[3 * n_peaks]
        return y

    return model


def read_xy(file, has_header: bool, xcol: int, ycol: int):
    """Read a CSV and return sorted, NaN-free (x, y) numpy arrays."""
    file.seek(0)
    df = pd.read_csv(file, header=0 if has_header else None, sep=None, engine="python")
    if df.shape[1] <= max(xcol, ycol):
        raise ValueError(f"File has only {df.shape[1]} columns; cannot read column {max(xcol, ycol)} (0-based)")
    x = pd.to_numeric(df.iloc[:, xcol], errors="coerce")
    y = pd.to_numeric(df.iloc[:, ycol], errors="coerce")
    m = x.notna() & y.notna()
    x, y = x[m].to_numpy(float), y[m].to_numpy(float)
    order = np.argsort(x)
    return x[order], y[order]


def fit_one(x, y, peaks, offset=False, allow_neg_A=False,
            window="all", margin_pct=10.0, max_nfev=20000):
    """
    peaks: list of (x_min, x_max, mu_init)
    Returns a dict with the fitted-parameter table, curve data, R², etc.; contains 'error' on failure.
    """
    n = len(peaks)
    result = {"n_peaks": n}

    # --- Fit window ---
    if window == "peaks":
        lo_all = min(p[0] for p in peaks)
        hi_all = max(p[1] for p in peaks)
        pad = (hi_all - lo_all) * margin_pct / 100.0
        mask = (x >= lo_all - pad) & (x <= hi_all + pad)
    else:
        mask = np.ones_like(x, dtype=bool)
    xf, yf = x[mask], y[mask]
    n_par = 3 * n + (1 if offset else 0)
    if len(xf) <= n_par:
        result["error"] = f"Only {len(xf)} data points in the fit window; not enough to fit {n_par} parameters"
        return result

    dx = np.median(np.diff(xf)) if len(xf) > 1 else 1.0
    s_min = max(abs(dx) * 0.5, 1e-12)
    y_ptp = max(np.ptp(yf), 1e-12)
    base = float(np.min(yf)) if offset else 0.0

    # --- Initial guesses and bounds ---
    p0, lb, ub = [], [], []
    for lo, hi, mu0 in peaks:
        idx = int(np.argmin(np.abs(xf - mu0)))
        A0 = float(yf[idx] - base)
        if not allow_neg_A:
            A0 = max(A0, 1e-3 * y_ptp)
        s_max = max(hi - lo, 4 * s_min)
        s0 = float(np.clip((hi - lo) / 4.0, s_min * 1.01, s_max * 0.99))
        p0 += [A0, mu0, s0]
        lb += [-np.inf if allow_neg_A else 0.0, lo, s_min]
        ub += [np.inf, hi, s_max]
    if offset:
        p0.append(base)
        lb.append(-np.inf)
        ub.append(np.inf)

    model = build_model(n, offset)
    try:
        popt, pcov = curve_fit(model, xf, yf, p0=p0, bounds=(lb, ub), max_nfev=max_nfev)
    except Exception as e:  # noqa: BLE001
        result["error"] = f"Fit failed: {e}"
        return result

    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(n_par, np.nan)

    # --- Parameter table ---
    rows = []
    for i in range(n):
        A, mu, s = popt[3 * i: 3 * i + 3]
        eA, emu, es = perr[3 * i: 3 * i + 3]
        rows.append({
            "Peak": i + 1,
            "A": A, "mu": mu, "sigma": s,
            "FWHM": FWHM_FACTOR * s,
            "Area": A * s * np.sqrt(2 * np.pi),
            "A_err": eA, "mu_err": emu, "sigma_err": es,
        })
    params_df = pd.DataFrame(rows)

    # --- Fit quality ---
    fit_f = model(xf, *popt)
    ss_res = float(np.sum((yf - fit_f) ** 2))
    ss_tot = float(np.sum((yf - yf.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / len(xf)))

    # --- Curve data (values at the original data points) ---
    c = popt[3 * n] if offset else 0.0
    curve = {"x": x, "y": y, "fit": model(x, *popt), "residual": y - model(x, *popt)}
    for i in range(n):
        A, mu, s = popt[3 * i: 3 * i + 3]
        curve[f"peak{i + 1}"] = A * np.exp(-((x - mu) ** 2) / (2 * s ** 2))
    curve_df = pd.DataFrame(curve)
    curve_df["in_fit_window"] = mask

    result.update(params=params_df, curve=curve_df, r2=r2, rmse=rmse,
                  offset=float(c) if offset else None,
                  mask=mask, popt=popt)
    return result


def make_figure(x, y, res, peaks, title, show_ranges=True):
    """Create the fit figure (top: data + fit + peaks; bottom: residuals)."""
    curve = res["curve"]
    n = res["n_peaks"]
    offset = res["offset"] or 0.0
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
    ax.scatter(x, y, s=10, color="0.35", alpha=0.7, label="Data", zorder=2)
    ax.plot(curve["x"], curve["fit"], color="crimson", lw=2, label="Total fit", zorder=4)

    cmap = plt.get_cmap("tab10")
    for i in range(n):
        row = res["params"].iloc[i]
        col = cmap(i % 10)
        comp = curve[f"peak{i + 1}"].to_numpy() + offset
        ax.fill_between(curve["x"], offset, comp, color=col, alpha=0.30, zorder=3)
        ax.plot(curve["x"], comp, color=col, lw=1.2, ls="--",
                label=f"P{i + 1}: A={row.A:.3g}, μ={row.mu:.4g}, σ={row.sigma:.3g}", zorder=3)
        if show_ranges:
            lo, hi, _ = peaks[i]
            ax.axvspan(lo, hi, color=col, alpha=0.06, zorder=1)
            axr.axvspan(lo, hi, color=col, alpha=0.06, zorder=1)

    ax.set_ylabel("y")
    ax.set_title(f"{title}   (R² = {res['r2']:.5f})")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(alpha=0.25)

    axr.axhline(0, color="0.4", lw=0.8)
    axr.scatter(curve["x"], curve["residual"], s=8, color="0.35")
    axr.set_xlabel("x")
    axr.set_ylabel("Residual")
    axr.grid(alpha=0.25)
    return fig


def fig_to_bytes(fig, fmt, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")  # utf-8-sig lets Excel open the file correctly


def build_zip(results: list) -> bytes:
    buf = io.BytesIO()
    summary_rows, all_params = [], []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            if "error" in r:
                continue
            stem = r["stem"]
            zf.writestr(f"{stem}/{stem}_params.csv", df_to_csv_bytes(r["params"]))
            zf.writestr(f"{stem}/{stem}_curves.csv", df_to_csv_bytes(r["curve"]))
            zf.writestr(f"{stem}/{stem}_fit.png", r["png"])
            zf.writestr(f"{stem}/{stem}_fit.svg", r["svg"])
            zf.writestr(f"{stem}/{stem}_fit.pdf", r["pdf"])
            p = r["params"].copy()
            p.insert(0, "File", r["name"])
            all_params.append(p)
            summary_rows.append({"File": r["name"], "R2": r["r2"], "RMSE": r["rmse"],
                                 "Offset": r["offset"]})
        if all_params:
            zf.writestr("all_params.csv", df_to_csv_bytes(pd.concat(all_params, ignore_index=True)))
            zf.writestr("summary.csv", df_to_csv_bytes(pd.DataFrame(summary_rows)))
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Multi-Gaussian Peak Fitting", layout="wide")
    st.title("📈 Multi-Gaussian Peak Fitting")
    st.caption("Model: y = Σ Aᵢ·exp(−(x−μᵢ)² / (2σᵢ²)) [+ constant baseline]")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("① Upload data")
        files = st.file_uploader("Select one or more CSV files (with x and y columns)",
                                 type=["csv", "txt"], accept_multiple_files=True)
        has_header = st.checkbox("First row of the file is a header", value=True)
        c1, c2 = st.columns(2)
        xcol = c1.number_input("x column index", 0, 50, 0, help="Counting starts at 0")
        ycol = c2.number_input("y column index", 0, 50, 1)

        st.header("④ Fit options")
        offset = st.checkbox("Add constant baseline (offset)", value=False)
        allow_neg = st.checkbox("Allow negative amplitude A", value=False)
        window = st.radio("Fit data range", ["All data", "Peak ranges only (with margin)"])
        margin = st.slider("Margin extension (% of range)", 0, 100, 20,
                           disabled=(window == "All data"))
        max_nfev = st.number_input("Max iterations (max_nfev)", 100, 1_000_000, 20000, step=1000)

        st.header("⑤ Figure options")
        dpi = st.slider("PNG resolution (dpi)", 72, 600, 200)
        show_ranges = st.checkbox("Show peak ranges on figure", value=True)

        st.divider()
        st.caption("The development of this app is assisted by Claude.")

    if not files:
        st.info("👈 Please upload CSV files in the sidebar first.")
        return

    # ---------------- Read data ----------------
    data, errors = {}, {}
    for f in files:
        try:
            x, y = read_xy(f, has_header, int(xcol), int(ycol))
            if len(x) < 4:
                raise ValueError("Fewer than 4 valid data points")
            data[f.name] = (x, y)
        except Exception as e:  # noqa: BLE001
            errors[f.name] = str(e)
    for name, msg in errors.items():
        st.error(f"❌ Could not read {name}: {msg}")
    if not data:
        return

    # ---------------- Step 2: number of peaks and ranges ----------------
    st.header("② Set the expected number of peaks and the x range of each peak")
    first_name = next(iter(data))
    x0, y0 = data[first_name]
    n_peaks = int(st.number_input("Expected number of peaks", 1, 20, 2))

    sig = (n_peaks, first_name, float(x0.min()), float(x0.max()))
    if st.session_state.get("peak_sig") != sig:
        edges = np.linspace(x0.min(), x0.max(), n_peaks + 1)
        st.session_state["peak_default"] = pd.DataFrame({
            "x_min": edges[:-1],
            "x_max": edges[1:],
            "mu_init": (edges[:-1] + edges[1:]) / 2,
        })
        st.session_state["peak_sig"] = sig

    st.markdown("Edit the table below: each row is one peak. `x_min`/`x_max` are the search range for the peak's μ, "
                "`mu_init` is the initial guess for μ (must lie within the range).")
    peak_df = st.data_editor(
        st.session_state["peak_default"],
        key=f"peak_editor_{hash(sig)}",
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "x_min": st.column_config.NumberColumn("x_min", format="%.6g"),
            "x_max": st.column_config.NumberColumn("x_max", format="%.6g"),
            "mu_init": st.column_config.NumberColumn("mu_init (initial guess)", format="%.6g"),
        },
    )

    # Validation
    valid = True
    peaks = []
    for i, r in peak_df.iterrows():
        lo, hi, mu = r["x_min"], r["x_max"], r["mu_init"]
        if pd.isna(lo) or pd.isna(hi):
            st.error(f"Peak {i + 1}: x_min / x_max cannot be empty")
            valid = False
            continue
        if lo >= hi:
            st.error(f"Peak {i + 1}: x_min must be smaller than x_max")
            valid = False
            continue
        if pd.isna(mu):
            mu = (lo + hi) / 2
        if not (lo <= mu <= hi):
            st.error(f"Peak {i + 1}: mu_init must lie within [{lo:g}, {hi:g}]")
            valid = False
            continue
        peaks.append((float(lo), float(hi), float(mu)))

    # Preview
    with st.expander("🔍 Data preview and peak ranges", expanded=True):
        pv_name = st.selectbox("Preview file", list(data.keys()))
        px, py = data[pv_name]
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(px, py, ".", ms=3, color="0.3")
        if valid:
            cmap = plt.get_cmap("tab10")
            for i, (lo, hi, mu) in enumerate(peaks):
                ax.axvspan(lo, hi, color=cmap(i % 10), alpha=0.2)
                ax.axvline(mu, color=cmap(i % 10), ls="--", lw=1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

    # ---------------- Steps 3-4: run the fit ----------------
    st.header("③ Run the fit")
    if st.button("🚀 Start fitting", type="primary", disabled=not valid):
        results = []
        progress = st.progress(0.0, text="Fitting…")
        for k, (name, (x, y)) in enumerate(data.items()):
            res = fit_one(
                x, y, peaks, offset=offset, allow_neg_A=allow_neg,
                window="peaks" if window.startswith("Peak") else "all",
                margin_pct=margin, max_nfev=int(max_nfev),
            )
            res["name"] = name
            res["stem"] = os.path.splitext(name)[0]
            if "error" not in res:
                fig = make_figure(x, y, res, peaks, name, show_ranges)
                res["png"] = fig_to_bytes(fig, "png", dpi)
                res["svg"] = fig_to_bytes(fig, "svg")
                res["pdf"] = fig_to_bytes(fig, "pdf")
                plt.close(fig)
            results.append(res)
            progress.progress((k + 1) / len(data), text=f"Finished {name}")
        progress.empty()
        st.session_state["results"] = results

    # ---------------- Steps 5-7: display and download ----------------
    results = st.session_state.get("results")
    if not results:
        return

    st.header("④ Fit results")
    ok = [r for r in results if "error" not in r]
    if ok:
        st.download_button("📦 Download all results (ZIP: figures + tables)", build_zip(results),
                           file_name="gaussian_fit_results.zip", mime="application/zip")

    tabs = st.tabs([r["name"] for r in results])
    for k, (tab, r) in enumerate(zip(tabs, results)):
        with tab:
            if "error" in r:
                st.error(r["error"])
                st.caption("Suggestion: adjust the peak ranges / initial guesses, enable the baseline, or switch to 'Peak ranges only'.")
                continue

            m1, m2, m3 = st.columns(3)
            m1.metric("R²", f"{r['r2']:.5f}")
            m2.metric("RMSE", f"{r['rmse']:.4g}")
            m3.metric("Offset", "—" if r["offset"] is None else f"{r['offset']:.4g}")

            st.image(r["png"], use_container_width=True)
            d1, d2, d3 = st.columns(3)
            d1.download_button("🖼 PNG", r["png"], f"{r['stem']}_fit.png", "image/png", key=f"png{k}")
            d2.download_button("🖼 SVG", r["svg"], f"{r['stem']}_fit.svg", "image/svg+xml", key=f"svg{k}")
            d3.download_button("🖼 PDF", r["pdf"], f"{r['stem']}_fit.pdf", "application/pdf", key=f"pdf{k}")

            st.subheader("Fit parameter table")
            st.dataframe(
                r["params"].style.format({
                    "A": "{:.5g}", "mu": "{:.6g}", "sigma": "{:.5g}", "FWHM": "{:.5g}",
                    "Area": "{:.5g}", "A_err": "{:.2g}", "mu_err": "{:.2g}", "sigma_err": "{:.2g}",
                }),
                use_container_width=True, hide_index=True,
            )
            t1, t2 = st.columns(2)
            t1.download_button("📄 Download parameter table (CSV)", df_to_csv_bytes(r["params"]),
                               f"{r['stem']}_params.csv", "text/csv", key=f"pcsv{k}")
            t2.download_button("📄 Download fitted curve data (CSV)", df_to_csv_bytes(r["curve"]),
                               f"{r['stem']}_curves.csv", "text/csv", key=f"ccsv{k}")


if __name__ == "__main__":
    main()
