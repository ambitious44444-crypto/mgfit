"""
Multi-Gaussian Peak Fitting — Streamlit App
執行方式:  streamlit run app.py
需求套件:  pip install streamlit numpy pandas scipy matplotlib
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
# 核心函式
# ----------------------------------------------------------------------------
def build_model(n_peaks: int, offset: bool):
    """回傳 n 個高斯疊加(可含常數基線)的模型函式。參數順序: A1,mu1,s1,...,[c]"""

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
    """讀取 CSV，回傳排序後、無 NaN 的 (x, y) numpy arrays。"""
    file.seek(0)
    df = pd.read_csv(file, header=0 if has_header else None, sep=None, engine="python")
    if df.shape[1] <= max(xcol, ycol):
        raise ValueError(f"檔案只有 {df.shape[1]} 欄，無法讀取第 {max(xcol, ycol)} 欄(從 0 起算)")
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
    回傳 dict，內含擬合參數表、曲線資料、R² 等；失敗時含 'error'。
    """
    n = len(peaks)
    result = {"n_peaks": n}

    # --- 擬合區間 ---
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
        result["error"] = f"擬合區間內資料點({len(xf)})不足以擬合 {n_par} 個參數"
        return result

    dx = np.median(np.diff(xf)) if len(xf) > 1 else 1.0
    s_min = max(abs(dx) * 0.5, 1e-12)
    y_ptp = max(np.ptp(yf), 1e-12)
    base = float(np.min(yf)) if offset else 0.0

    # --- 初始猜測與邊界 ---
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
        result["error"] = f"擬合失敗: {e}"
        return result

    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(n_par, np.nan)

    # --- 參數表 ---
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

    # --- 擬合品質 ---
    fit_f = model(xf, *popt)
    ss_res = float(np.sum((yf - fit_f) ** 2))
    ss_tot = float(np.sum((yf - yf.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / len(xf)))

    # --- 曲線資料(原始資料點上的值) ---
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
    """產生擬合圖(上: 資料+擬合+各峰；下: 殘差)。"""
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
    return df.to_csv(index=False).encode("utf-8-sig")  # utf-8-sig 讓 Excel 正確開啟


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
    st.title("📈 多重高斯峰擬合 (Multi-Gaussian Fitting)")
    st.caption("模型: y = Σ Aᵢ·exp(−(x−μᵢ)² / (2σᵢ²)) [+ 常數基線]")

    # ---------------- 側邊欄 ----------------
    with st.sidebar:
        st.header("① 上傳資料")
        files = st.file_uploader("選擇一或多個 CSV 檔 (含 x, y 兩欄)",
                                 type=["csv", "txt"], accept_multiple_files=True)
        has_header = st.checkbox("檔案第一列為欄位名稱", value=True)
        c1, c2 = st.columns(2)
        xcol = c1.number_input("x 欄位 index", 0, 50, 0, help="從 0 開始計數")
        ycol = c2.number_input("y 欄位 index", 0, 50, 1)

        st.header("④ 擬合選項")
        offset = st.checkbox("加入常數基線 (offset)", value=False)
        allow_neg = st.checkbox("允許負振幅 A", value=False)
        window = st.radio("擬合資料範圍", ["全部資料", "僅峰範圍聯集(含邊界)"])
        margin = st.slider("邊界延伸 (% of 範圍)", 0, 100, 20,
                           disabled=(window == "全部資料"))
        max_nfev = st.number_input("最大迭代次數 (max_nfev)", 100, 1_000_000, 20000, step=1000)

        st.header("⑤ 圖檔選項")
        dpi = st.slider("PNG 解析度 (dpi)", 72, 600, 200)
        show_ranges = st.checkbox("圖上標示峰範圍", value=True)

    if not files:
        st.info("👈 請先在左側上傳 CSV 檔案。")
        return

    # ---------------- 讀取資料 ----------------
    data, errors = {}, {}
    for f in files:
        try:
            x, y = read_xy(f, has_header, int(xcol), int(ycol))
            if len(x) < 4:
                raise ValueError("有效資料點少於 4 筆")
            data[f.name] = (x, y)
        except Exception as e:  # noqa: BLE001
            errors[f.name] = str(e)
    for name, msg in errors.items():
        st.error(f"❌ 無法讀取 {name}：{msg}")
    if not data:
        return

    # ---------------- 步驟 2: 峰數與範圍 ----------------
    st.header("② 設定理論峰數與各峰 x 範圍")
    first_name = next(iter(data))
    x0, y0 = data[first_name]
    n_peaks = int(st.number_input("理論峰數量", 1, 20, 2))

    sig = (n_peaks, first_name, float(x0.min()), float(x0.max()))
    if st.session_state.get("peak_sig") != sig:
        edges = np.linspace(x0.min(), x0.max(), n_peaks + 1)
        st.session_state["peak_default"] = pd.DataFrame({
            "x_min": edges[:-1],
            "x_max": edges[1:],
            "mu_init": (edges[:-1] + edges[1:]) / 2,
        })
        st.session_state["peak_sig"] = sig

    st.markdown("請編輯下表：每列代表一個峰。`x_min`/`x_max` 為該峰 μ 的搜尋範圍，"
                "`mu_init` 為 μ 的初始猜測值（需落在範圍內）。")
    peak_df = st.data_editor(
        st.session_state["peak_default"],
        key=f"peak_editor_{hash(sig)}",
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "x_min": st.column_config.NumberColumn("x_min", format="%.6g"),
            "x_max": st.column_config.NumberColumn("x_max", format="%.6g"),
            "mu_init": st.column_config.NumberColumn("mu_init (初始猜測)", format="%.6g"),
        },
    )

    # 驗證
    valid = True
    peaks = []
    for i, r in peak_df.iterrows():
        lo, hi, mu = r["x_min"], r["x_max"], r["mu_init"]
        if pd.isna(lo) or pd.isna(hi):
            st.error(f"Peak {i + 1}: x_min / x_max 不可為空")
            valid = False
            continue
        if lo >= hi:
            st.error(f"Peak {i + 1}: x_min 必須小於 x_max")
            valid = False
            continue
        if pd.isna(mu):
            mu = (lo + hi) / 2
        if not (lo <= mu <= hi):
            st.error(f"Peak {i + 1}: mu_init 必須落在 [{lo:g}, {hi:g}] 內")
            valid = False
            continue
        peaks.append((float(lo), float(hi), float(mu)))

    # 預覽
    with st.expander("🔍 資料預覽與峰範圍", expanded=True):
        pv_name = st.selectbox("預覽檔案", list(data.keys()))
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

    # ---------------- 步驟 3-4: 執行擬合 ----------------
    st.header("③ 執行擬合")
    if st.button("🚀 開始擬合", type="primary", disabled=not valid):
        results = []
        progress = st.progress(0.0, text="擬合中…")
        for k, (name, (x, y)) in enumerate(data.items()):
            res = fit_one(
                x, y, peaks, offset=offset, allow_neg_A=allow_neg,
                window="peaks" if window.startswith("僅") else "all",
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
            progress.progress((k + 1) / len(data), text=f"完成 {name}")
        progress.empty()
        st.session_state["results"] = results

    # ---------------- 步驟 5-7: 顯示與下載 ----------------
    results = st.session_state.get("results")
    if not results:
        return

    st.header("④ 擬合結果")
    ok = [r for r in results if "error" not in r]
    if ok:
        st.download_button("📦 下載全部結果 (ZIP: 圖 + 表)", build_zip(results),
                           file_name="gaussian_fit_results.zip", mime="application/zip")

    tabs = st.tabs([r["name"] for r in results])
    for k, (tab, r) in enumerate(zip(tabs, results)):
        with tab:
            if "error" in r:
                st.error(r["error"])
                st.caption("建議：調整峰範圍／初始猜測、啟用基線，或改用『僅峰範圍聯集』。")
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

            st.subheader("擬合參數表")
            st.dataframe(
                r["params"].style.format({
                    "A": "{:.5g}", "mu": "{:.6g}", "sigma": "{:.5g}", "FWHM": "{:.5g}",
                    "Area": "{:.5g}", "A_err": "{:.2g}", "mu_err": "{:.2g}", "sigma_err": "{:.2g}",
                }),
                use_container_width=True, hide_index=True,
            )
            t1, t2 = st.columns(2)
            t1.download_button("📄 下載參數表 (CSV)", df_to_csv_bytes(r["params"]),
                               f"{r['stem']}_params.csv", "text/csv", key=f"pcsv{k}")
            t2.download_button("📄 下載擬合曲線資料 (CSV)", df_to_csv_bytes(r["curve"]),
                               f"{r['stem']}_curves.csv", "text/csv", key=f"ccsv{k}")


if __name__ == "__main__":
    main()
