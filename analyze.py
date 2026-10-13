"""Figures + stats for reports/ppo_report.md, from reports/runs.json (re-run after fetch_runs.py)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = {r["id"]: r for r in json.load(open("reports/runs.json"))["ppo-assignment"]}
G = {
    "PPO inicial s1": "s85oibgl", "PPO inicial s2": "p0vuk9dt",
    "PPO v1 (γ,λ) 1M s1": "u8mg9q9e", "PPO v2 (16 envs, 10 ép.) 1M s1": "bhhqzxcq",
    "PPO final s1": "5pdayags", "PPO final s2": "dzm2xwfo",
    "A2C s1": "cr63uqu9", "A2C s2": "prbn0gm8",
}
COL = {"PPO inicial s1": "#999", "PPO inicial s2": "#555", "PPO v1 (γ,λ) 1M s1": "#e69f00",
       "PPO v2 (16 envs, 10 ép.) 1M s1": "#cc79a7", "PPO final s1": "#0072b2", "PPO final s2": "#56b4e9",
       "A2C s1": "#d55e00", "A2C s2": "#a63603"}


def series(name, key, x="global_step"):
    h = [p for p in runs[G[name]]["history"] if p.get(key) is not None and p.get(x) is not None]
    return np.array([p[x] for p in h], float), np.array([p[key] for p in h], float)


def first_cross(name, thr):
    x, y = series(name, "charts/episodic_return_mean_last100")
    i = np.where(y >= thr)[0]
    return int(x[i[0]]) if len(i) else None


def time_cross(name, thr):
    t, y = series(name, "charts/episodic_return_mean_last100", "_runtime")
    i = np.where(y >= thr)[0]
    return float(t[i[0]]) if len(i) else None


def plot(names, key, fname, title, ylabel, x="global_step", smooth=1, hline=None):
    plt.figure(figsize=(7, 3.6))
    for n in names:
        xs, ys = series(n, key, x)
        if smooth > 1 and len(ys) > smooth:
            ys = np.convolve(ys, np.ones(smooth) / smooth, mode="valid"); xs = xs[smooth - 1:]
        plt.plot(xs, ys, label=n, color=COL[n], lw=1.4)
    if hline is not None:
        plt.axhline(hline, color="k", ls="--", lw=0.8)
    plt.xlabel("passos de ambiente" if x == "global_step" else "tempo de relógio (s)")
    plt.ylabel(ylabel); plt.title(title); plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"reports/{fname}", dpi=130); plt.close()


R = "charts/episodic_return_mean_last100"
init = ["PPO inicial s1", "PPO inicial s2"]
fin = ["PPO final s1", "PPO final s2"]
plot(init + fin, R, "fig1_init_vs_final.png", "PPO: config inicial vs final", "retorno médio (últimos 100)", hline=200)
diag = ["PPO inicial s1", "PPO v1 (γ,λ) 1M s1", "PPO v2 (16 envs, 10 ép.) 1M s1", "PPO final s1"]
fig, ax = plt.subplots(2, 2, figsize=(10, 6))
for a, (k, t) in zip(ax.ravel(), [("losses/approx_kl", "approx_kl"), ("losses/clipfrac", "clipfrac"),
                                   ("losses/entropy", "entropy"), ("losses/explained_variance", "explained_variance")]):
    for n in diag:
        xs, ys = series(n, k)
        if len(ys) > 9:
            ys = np.convolve(ys, np.ones(9) / 9, mode="valid"); xs = xs[8:]
        a.plot(xs, ys, label=n, color=COL[n], lw=1.2)
    a.set_title(t); a.grid(alpha=.3)
ax[0, 0].legend(fontsize=6); plt.tight_layout(); plt.savefig("reports/fig2_diagnostics.png", dpi=130); plt.close()
both = fin + ["A2C s1", "A2C s2"]
plot(both, R, "fig3_ppo_vs_a2c_steps.png", "PPO vs A2C no LunarLander (mesmo orçamento de 2M passos)",
     "retorno médio (últimos 100)", hline=200)
plot(both, R, "fig4_ppo_vs_a2c_time.png", "Retorno × tempo de relógio", "retorno médio (últimos 100)", x="_runtime", hline=200)
plot(both, "charts/SPS", "fig5_sps.png", "SPS", "passos/s")

print("== passos até retorno (last100) 0/100/200; tempo; SPS médio; num_updates final; resumo")
for n, i in G.items():
    r = runs[i]; s = r["summary"]
    xs, _ = series(n, R)
    rt = s.get("_runtime") or (r["history"][-1]["_runtime"] if r["history"] else float("nan"))
    sps = series(n, "charts/SPS")[1]
    print(n, i, r["state"],
          "steps@0/100/200:", [first_cross(n, t) for t in (0, 100, 200)],
          "time@0/100/200(s):", [None if (v := time_cross(n, t)) is None else round(v) for t in (0, 100, 200)],
          "runtime_s=%.0f" % rt, "SPS_med=%.0f" % np.median(sps), "updates=%s" % (s.get("charts/num_updates") or 2000000 // (r["config"]["num_envs"] * r["config"]["ppo" if "ppo" in r["config"] and r["config"]["ppo"] else "a2c"]["num_steps"])),
          "last100=%.1f" % s.get(R, np.nan), "eval=%.1f±%.1f" % (s.get("eval/mean_return", np.nan), s.get("eval/std_return", np.nan)))
print("== diagnósticos (média 1º terço / último terço do treino)")
for n in diag + ["PPO inicial s2", "PPO final s2"]:  # ppo only
    out = []
    for k in ("losses/approx_kl", "losses/clipfrac", "losses/entropy", "losses/explained_variance"):
        xs, ys = series(n, k); m = len(ys) // 3
        out.append("%s %.4f→%.4f" % (k.split('/')[1], ys[:m].mean(), ys[-m:].mean()))
    print(n, "|", " | ".join(out))
# worst dips of the final runs (stability)
for n in fin + ["A2C s1", "A2C s2"]:
    xs, ys = series(n, R)
    peak = np.maximum.accumulate(ys); print(n, "max drawdown from running peak: %.0f" % (peak - ys).max(), "peak %.0f" % ys.max())
