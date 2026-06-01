"""
Gridlock Hackathon 2.0 - Traffic Demand Prediction - FINAL HONEST SOLUTION
==========================================================================
Public LB: 92.31506 (team sarthak).  NO external data, NO label lookup, NO calibration hacks.

Pipeline (each step LB-verified to help):
  1. GBM core      : 0.60 * LightGBM(2-seed) + 0.40 * CatBoost   on simple features
  2. NN diversity  : + 6% entity-embedding neural net (decorrelated member)
  3. Smoothing     : per-geohash Gaussian temporal smoothing (sigma=2.0), mixed 65%
                     -- exploits demand autocorrelation ~0.965 across time-of-day.

Run:  python solution.py   ->   writes submission.csv (41,778 rows: Index,demand)
Env:  python + pandas, numpy, scipy, scikit-learn, lightgbm, catboost, torch (CPU).
Memory-lean (float32, train-then-free) for 8 GB RAM.
"""
import warnings, gc, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
import lightgbm as lgb
from catboost import CatBoostRegressor
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings("ignore"); torch.set_num_threads(4)
DATA = "./dataset"   # adjust to your dataset path; expects train.csv, test.csv

# ---------- geohash6 base-32 decode -> lat/lon ----------
_b32 = "0123456789bcdefghjkmnpqrstuvwxyz"; _d = {c: i for i, c in enumerate(_b32)}
def gh_decode(g):
    a, b, c, e = -90., 90., -180., 180.; ev = True
    for ch in g:
        cd = _d[ch]
        for m in (16, 8, 4, 2, 1):
            if ev: mid = (c+e)/2; c, e = (mid, e) if cd & m else (c, mid)
            else:  mid = (a+b)/2; a, b = (mid, b) if cd & m else (a, mid)
            ev = not ev
    return (a+b)/2, (c+e)/2

train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
y = train["demand"].astype("float32").values; YMEAN = float(y.mean())
uq = pd.unique(pd.concat([train["geohash"], test["geohash"]]).dropna())
latm = {g: gh_decode(g)[0] for g in uq}; lonm = {g: gh_decode(g)[1] for g in uq}

def feat(df):
    df = df.copy(); hm = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"], df["minute"] = hm[0], hm[1]; df["mod"] = df["hour"]*60 + df["minute"]
    r = 2*np.pi*df["mod"]/1440.; df["mod_sin"], df["mod_cos"] = np.sin(r), np.cos(r)
    r2 = 2*np.pi*df["mod"]/720.; df["mod12_sin"], df["mod12_cos"] = np.sin(r2), np.cos(r2)
    for k in (3, 4, 5): df[f"gh{k}"] = df["geohash"].str.slice(0, k)
    df["lat"] = df["geohash"].map(latm).astype("float32"); df["lon"] = df["geohash"].map(lonm).astype("float32")
    for c in ("RoadType", "Temperature", "Weather"): df[f"{c}_na"] = df[c].isna().astype(int)
    return df
train, test = feat(train), feat(test)

CAT = ["geohash", "gh3", "gh4", "gh5", "RoadType", "LargeVehicles", "Landmarks", "Weather"]
for c in CAT:
    cats = pd.api.types.union_categoricals([train[c].astype("category"), test[c].astype("category")]).categories
    train[c] = pd.Categorical(train[c], categories=cats); test[c] = pd.Categorical(test[c], categories=cats)
F = ["geohash","gh3","gh4","gh5","day","hour","minute","mod","mod_sin","mod_cos","mod12_sin","mod12_cos",
     "lat","lon","RoadType","NumberofLanes","LargeVehicles","Landmarks","Temperature","Weather",
     "RoadType_na","Temperature_na","Weather_na"]
for df in (train, test):
    for c in df.columns:
        if df[c].dtype == "float64": df[c] = df[c].astype("float32")
X, Xt = train[F], test[F]

# ---------- 1. GBM core ----------
print("LightGBM (2 seeds)...", flush=True); lgb_te = np.zeros(len(Xt))
for s in (42, 7):
    p = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=127, feature_fraction=0.9,
             bagging_fraction=0.9, bagging_freq=5, min_data_in_leaf=40, lambda_l2=1.0, verbose=-1, seed=s, num_threads=4)
    m = lgb.train(p, lgb.Dataset(X, y, categorical_feature=CAT), num_boost_round=1300); lgb_te += m.predict(Xt)/2; del m; gc.collect()
print("CatBoost...", flush=True); gc.collect()
Xs, Xts = X.copy(), Xt.copy()
for c in CAT: Xs[c] = Xs[c].astype(str).fillna("NA"); Xts[c] = Xts[c].astype(str).fillna("NA")
cb = CatBoostRegressor(iterations=1300, learning_rate=0.05, depth=8, l2_leaf_reg=3.0, loss_function="RMSE",
                       random_seed=42, verbose=0, thread_count=4, cat_features=CAT, max_ctr_complexity=1, train_dir="/tmp/cb_final")
cb.fit(Xs, y); cb_te = cb.predict(Xts); del cb, Xs, Xts; gc.collect()
core = np.clip(0.60*lgb_te + 0.40*cb_te, 0, 1)

# ---------- 2. NN diversity member ----------
print("Neural net...", flush=True)
NCAT = ["geohash","gh3","gh4","gh5","RoadType","Weather"]
NCONT = ["mod_sin","mod_cos","mod12_sin","mod12_cos","lat","lon","Temperature"]
for df in (train, test): df["Temperature"] = df["Temperature"].fillna(train["Temperature"].median())
vocab, emb = {}, []
for c in NCAT:
    vals = pd.concat([train[c], test[c]]).astype(str).fillna("<NA>"); cats = ["<UNK>"]+sorted(vals.unique())
    vocab[c] = {v: i for i, v in enumerate(cats)}; emb.append((len(cats), 48 if c == "geohash" else min(24, (len(cats)+1)//2)))
def enc(df): return np.stack([df[c].astype(str).fillna("<NA>").map(lambda v: vocab[c].get(v, 0)).values for c in NCAT], 1).astype("int64")
Xc, Xct = enc(train), enc(test)
sc = StandardScaler().fit(train[NCONT].values)
Xn = sc.transform(train[NCONT].values).astype("float32"); Xnt = sc.transform(test[NCONT].values).astype("float32")
class Net(nn.Module):
    def __init__(s, emb, nc, ym):
        super().__init__(); s.embs = nn.ModuleList([nn.Embedding(a, b) for a, b in emb]); s.bn = nn.BatchNorm1d(nc)
        s.body = nn.Sequential(nn.Linear(sum(b for _, b in emb)+nc, 192), nn.Mish(), nn.BatchNorm1d(192), nn.Dropout(0.1),
                               nn.Linear(192, 96), nn.Mish(), nn.BatchNorm1d(96)); s.head = nn.Linear(96, 1)
        nn.init.zeros_(s.head.weight); s.head.bias.data.fill_(ym)
    def forward(s, xc, xn): return s.head(s.body(torch.cat([e(xc[:, i]) for i, e in enumerate(s.embs)]+[s.bn(xn)], 1))).squeeze(1)
EP, BS = 40, 4096; torch.manual_seed(42)
dl = DataLoader(TensorDataset(torch.tensor(Xc), torch.tensor(Xn), torch.tensor(y)), batch_size=BS, shuffle=True)
net = Net(emb, len(NCONT), YMEAN); opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
schd = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, epochs=EP, steps_per_epoch=len(dl), pct_start=0.2); lf = nn.MSELoss()
for _ in range(EP):
    net.train()
    for xb, nb, yb in dl:
        opt.zero_grad(); l = lf(net(xb, nb), yb); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); schd.step()
net.eval()
with torch.no_grad(): nn_te = net(torch.tensor(Xct), torch.tensor(Xnt)).numpy()

# ---------- 3. blend + per-geohash Gaussian temporal smoothing ----------
base = np.clip(0.94*core + 0.06*nn_te, 0, 1)
test["base"] = base; out = np.empty(len(test))
for _, ix in test.groupby("geohash").groups.items():
    sub = test.loc[ix].sort_values("mod"); out[sub.index.values] = gaussian_filter1d(sub["base"].values, 2.0, mode="nearest")
final = np.clip(0.35*base + 0.65*out, 0, 1)
pd.DataFrame({"Index": test["Index"].values, "demand": final}).to_csv("submission.csv", index=False)
print(f"submission.csv written | rows {len(final)} | mean {final.mean():.4f}", flush=True)
