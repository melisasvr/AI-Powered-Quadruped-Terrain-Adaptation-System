"""streamlit run dashboard.py"""
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from quadruped import QuadrupedEnv, KINDS, XS, YS
from brain import make_actor, rollout, benchmark

st.set_page_config(page_title="Quadruped Terrain Adaptation", layout="wide")
st.title("🐕 Quadruped Terrain Adaptation (MuJoCo + JAX)")

with st.sidebar:
    terrain = st.selectbox("Terrain", KINDS, index=2)
    policy_path = st.text_input("Policy file (.npy)", "policy.npy")
    push = st.checkbox("Apply lateral push (recovery test)", True)
    seed = st.number_input("Seed", 0, 10_000, 0)
    run = st.button("▶ Run episode", type="primary")
    st.caption("Missing policy file → untrained trot baseline.")


def render3d(fr, h, size=(760, 460)):
    try:
        import pyvista as pv
        x0 = fr["pos"][0]
        i0, i1 = np.searchsorted(XS, x0 - 3), min(np.searchsorted(XS, x0 + 3), len(XS))
        X, Y = np.meshgrid(XS[i0:i1], YS, indexing="ij")
        Z = h[:, i0:i1].T
        grid = pv.StructuredGrid(X, Y, Z)
        grid["h"] = Z.ravel(order="F")
        pl = pv.Plotter(off_screen=True, window_size=size)
        pl.add_mesh(grid, scalars="h", cmap="terrain", show_scalar_bar=False)
        box = pv.Box(bounds=(-0.2, 0.2, -0.08, 0.08, -0.04, 0.04))
        M = np.eye(4); M[:3, :3] = fr["R"]; M[:3, 3] = fr["pos"]
        box.transform(M, inplace=True)
        pl.add_mesh(box, color="royalblue")
        for fp, c in zip(fr["feet"], fr["contacts"]):
            pl.add_mesh(pv.Sphere(radius=0.03, center=fp), color="limegreen" if c else "crimson")
            pl.add_mesh(pv.Line(fr["pos"], fp), color="gray", line_width=3)
        pl.camera_position = [(x0 - 1.6, -2.0, 1.2), (x0, 0, 0.25), (0, 0, 1)]
        img = pl.screenshot(return_img=True); pl.close()
        return img
    except Exception as e:                                   # fallback: matplotlib side view
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(XS, h[len(YS) // 2], "k")
        ax.scatter(fr["feet"][:, 0], fr["feet"][:, 2], c=["g" if c else "r" for c in fr["contacts"]])
        ax.scatter(fr["pos"][0], fr["pos"][2], marker="s")
        ax.set_xlim(fr["pos"][0] - 2, fr["pos"][0] + 2); ax.set_ylim(-0.1, 0.8); ax.set_aspect("equal")
        ax.set_title(f"PyVista unavailable ({type(e).__name__}) - side view")
        return fig


tab1, tab2 = st.tabs(["Live episode", "Benchmark"])

with tab1:
    if run:
        with st.spinner("Simulating…"):
            env = QuadrupedEnv(terrain, push=push)
            info, frames, log = rollout(env, make_actor(policy_path), seed=int(seed), record=True)
            st.session_state.ep = dict(info=info, frames=frames, log=pd.DataFrame(log), h=env.h.copy())
    ep = st.session_state.get("ep")
    if ep:
        info = ep["info"]
        c = st.columns(5)
        c[0].metric("Success", "✅" if info["success"] else "❌")
        c[1].metric("Distance (m)", f"{info['dist']:.2f}")
        c[2].metric("Energy (J)", f"{info['energy']:.0f}")
        c[3].metric("Fell", "yes" if info["fell"] else "no")
        c[4].metric("Push recovered", "—" if not info["pushed"] else ("yes" if info["recovered"] else "no"))
        k = st.slider("Frame", 0, len(ep["frames"]) - 1, 0)
        out = render3d(ep["frames"][k], ep["h"])
        if hasattr(out, "savefig"):
            st.pyplot(out)
        else:
            st.image(out)
        log = ep["log"]
        a, b, d = st.columns(3)
        fig, ax = plt.subplots(figsize=(4, 2.4))
        ax.imshow(log[[f"c{i}" for i in range(4)]].T.values, aspect="auto", cmap="Greens",
                  extent=[0, log.t.iloc[-1], 3.5, -0.5])
        ax.set_yticks(range(4)); ax.set_yticklabels(["FR", "FL", "RR", "RL"]); ax.set_title("Foot contacts")
        a.pyplot(fig)
        fig, ax = plt.subplots(figsize=(4, 2.4))
        ax.plot(log.t, np.degrees(log.roll), label="roll"); ax.plot(log.t, np.degrees(log.pitch), label="pitch")
        ax.set_title("Stability (deg)"); ax.legend(); b.pyplot(fig)
        fig, ax = plt.subplots(figsize=(4, 2.4))
        ax.plot(log.t, log.power); ax.set_title("Mechanical power (W)"); d.pyplot(fig)
    else:
        st.info("Pick a terrain and press **Run episode**.")

with tab2:
    n = st.slider("Episodes per terrain", 1, 20, 5)
    if st.button("Run benchmark"):
        with st.spinner("Benchmarking all terrains…"):
            df = pd.DataFrame(benchmark(make_actor(policy_path), KINDS, n, push))
            g = df.groupby("terrain").agg(success_rate=("success", "mean"), distance=("dist", "mean"),
                                          energy_J=("energy", "mean"), recovery_rate=("recovered", "mean"))
        st.dataframe(g.round(2))
        st.bar_chart(g[["success_rate", "recovery_rate"]])