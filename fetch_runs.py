"""Pull run summaries/history from W&B (GraphQL, basic auth) into reports/runs.json."""
import json, os, sys, requests

env = dict(l.strip().split("=", 1) for l in open(".env") if "=" in l and not l.startswith("#"))
KEY, ENT = env["WANDB_API_KEY"], env["WANDB_ENTITY"]
URL = "https://api.wandb.ai/graphql"


def gql(query, variables=None):
    r = requests.post(URL, auth=("api", KEY), json={"query": query, "variables": variables or {}}, timeout=60)
    r.raise_for_status()
    j = r.json()
    if "errors" in j:
        raise RuntimeError(j["errors"])
    return j["data"]


RUNS_Q = """query($e:String!,$p:String!){project(name:$p,entityName:$e){runs(first:100){edges{node{
 name displayName state createdAt config summaryMetrics historyLineCount}}}}}"""
HIST_Q = """query($e:String!,$p:String!,$r:String!,$specs:[JSONString!]!){project(name:$p,entityName:$e){run(name:$r){
 sampledHistory(specs:$specs)}}}"""
KEYS = ["global_step", "_runtime", "charts/episodic_return_mean_last100", "losses/approx_kl", "losses/clipfrac",
        "losses/entropy", "losses/explained_variance", "losses/value_loss", "charts/SPS", "charts/num_updates"]

out = {}
for proj in sys.argv[1:]:
    runs = []
    for e in gql(RUNS_Q, {"e": ENT, "p": proj})["project"]["runs"]["edges"]:
        n = e["node"]
        cfg = json.loads(n["config"]) if isinstance(n["config"], str) else n["config"]
        summ = json.loads(n["summaryMetrics"]) if isinstance(n["summaryMetrics"], str) else n["summaryMetrics"]
        keys = KEYS if cfg.get("algorithm", {}).get("value") == "ppo" or cfg.get("algorithm") == "ppo" else [k for k in KEYS if k not in ("losses/approx_kl", "losses/clipfrac", "charts/num_updates")]
        spec = json.dumps({"keys": keys, "samples": 400})
        h = gql(HIST_Q, {"e": ENT, "p": proj, "r": n["name"], "specs": [spec]})["project"]["run"]["sampledHistory"]
        runs.append({"id": n["name"], "name": n["displayName"], "state": n["state"], "created": n["createdAt"],
                     "config": {k: (v.get("value") if isinstance(v, dict) else v) for k, v in cfg.items()},
                     "summary": {k: v for k, v in summ.items() if (not k.startswith("_") or k == "_runtime") and isinstance(v, (int, float))},
                     "history": h[0] if h else []})
    out[proj] = runs
os.makedirs("reports", exist_ok=True)
json.dump(out, open("reports/runs.json", "w"))
for proj, runs in out.items():
    for r in sorted(runs, key=lambda r: r["created"]):
        s = r["summary"]
        print(proj, r["id"], r["state"], r["config"].get("seed"), r["config"].get("total_timesteps"),
              "ret100=%.1f" % s.get("charts/episodic_return_mean_last100", float("nan")),
              "eval=%.1f" % s.get("eval/mean_return", float("nan")))
