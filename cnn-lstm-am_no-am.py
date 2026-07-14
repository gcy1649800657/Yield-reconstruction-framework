# ablation_fit_residual_reconstruct_yield_TimeAtt_mergePlain.py
# 消融实验：拟合 Residual，然后用 Trend 重构 Yield，进行精度评价
# 模型：CNN + LSTM + (Time-Att可选) + Region Emb + Stage Emb
#   ✅ 导出训练集/测试集每样本结果到一个 Excel（两个Sheet：train/test）
#   ✅ 仍然打印 Residual 与 Reconstructed Yield 的 Train/Val/Test 精度

import os
import random
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================
# 0) Seed & device
# =========================
def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================
# 1) Config
# =========================
DATA_PATH = r"D:\Shap_多维\分解时间序列\SIF_FPAR_EVI2_Tmpmin_Tmpmax_SM_汇总.xlsx"
SHEET_NAME = 0

COL_NAME  = "NAME"
COL_ID    = "ID"
COL_YEAR  = "YEAR"
COL_YIELD = "yield"
COL_RESID = "Residual"
COL_TREND = "Trend"

# Feature processing
ADD_YEAR_TO_X = True  # 把 YEAR 作为一个额外特征

# Split
SEED = 43
USE_STRATIFY_YEAR = True
TEST_SIZE = 0.20
VAL_SIZE  = 0.125  # from remaining -> about 7:1:2

# Train
BATCH_SIZE = 64
EPOCHS = 400
LR = 5e-4
WEIGHT_DECAY = 5e-3
PATIENCE = 60
MIN_DELTA = 1e-6

# Ablation modes (PLAIN 已合并进 NO_TATT)
RUN_ALL_MODES = True
MODE = "FULL"  # FULL / NO_TATT

# Time-att sharpening
TAU_TIME = 1.0  # <1 更尖锐, >1 更平滑；如 0.5 / 0.3 可让注意力更集中

# Export per-sample predictions
EXPORT_PRED_EXCEL = True
EXPORT_DIR = "outputs_noam"


# =========================
# 2) Utils
# =========================
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# =========================
# 3) Suffix / vars detection (6-digit numeric)
# =========================
def detect_suffixes_6digit_numeric(cols: List[str]) -> List[str]:
    suf = set()
    for c in cols:
        if "_" not in c:
            continue
        tail = c.split("_")[-1]
        if len(tail) == 6 and tail.isdigit():
            suf.add(tail)
    return sorted(list(suf))

def infer_used_vars(df: pd.DataFrame, suffixes: List[str]) -> List[str]:
    meta = {COL_NAME, COL_ID, COL_YEAR, COL_YIELD, COL_RESID, COL_TREND}
    vars_set = set()
    for c in df.columns:
        if c in meta:
            continue
        if "_" not in c:
            continue
        tail = c.split("_")[-1]
        if tail in suffixes:
            v = c.rsplit("_", 1)[0]
            vars_set.add(v)
    used = sorted(list(vars_set))
    if len(used) == 0:
        raise ValueError("No feature variables detected from suffixes.")
    return used

def build_X_and_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    X:      (N,T,D) (optionally with YEAR as last channel)
    y_res:  (N,1) Residual
    y_yld:  (N,1) Yield (raw)
    trend:  (N,1) Trend (raw)
    """
    suffixes = detect_suffixes_6digit_numeric(list(df.columns))
    if len(suffixes) == 0:
        raise ValueError("No 6-digit numeric time suffixes detected (e.g., _030226).")

    used_vars = infer_used_vars(df, suffixes)

    N = len(df)
    T = len(suffixes)
    D = len(used_vars) + (1 if ADD_YEAR_TO_X else 0)

    X = np.zeros((N, T, D), dtype=np.float32)
    for ti, suf in enumerate(suffixes):
        cols_t = [f"{v}_{suf}" for v in used_vars]
        missing = [c for c in cols_t if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns at time={suf}: {missing[:10]}")
        X[:, ti, :len(used_vars)] = df[cols_t].to_numpy(dtype=np.float32)

    if ADD_YEAR_TO_X:
        year_vals = df[COL_YEAR].to_numpy(dtype=np.float32).reshape(-1, 1)
        X[:, :, -1] = np.repeat(year_vals, T, axis=1)

    y_res = df[COL_RESID].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_yld = df[COL_YIELD].to_numpy(dtype=np.float32).reshape(-1, 1)
    trend = df[COL_TREND].to_numpy(dtype=np.float32).reshape(-1, 1)

    return X, y_res, y_yld, trend, used_vars, suffixes


# =========================
# 4) Dataset
# =========================
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, reg_idx: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.reg = torch.tensor(reg_idx, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.reg[i], self.y[i]


# =========================
# 5) Model: CNN + LSTM + (Time-Att optional) + Region/Stage Emb
# =========================
class CNN_LSTM_TimeAtt(nn.Module):
    """
    Modes (PLAIN 已合并进 NO_TATT):
      FULL   : time-att
      NO_TATT: last pooling
    """
    def __init__(
        self,
        in_dim: int,
        num_regions: int,
        time_steps: int,
        mode: str = "FULL",
        conv_channels: int = 16,
        lstm_hidden: int = 32,
        region_emb_dim: int = 4,
        stage_emb_dim: int = 8,
        dropout_p: float = 0.5,
        tau_time: float = 1.0,
    ):
        super().__init__()
        self.mode = mode.upper()
        assert self.mode in {"FULL", "NO_TATT"}

        self.use_tatt = (self.mode == "FULL")
        self.time_steps = int(time_steps)
        self.tau_time = float(tau_time)

        self.region_emb = nn.Embedding(num_regions, region_emb_dim)
        self.stage_emb  = nn.Embedding(time_steps, stage_emb_dim)

        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=conv_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(conv_channels)

        self.lstm = nn.LSTM(
            input_size=conv_channels + stage_emb_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.att_fc = nn.Linear(lstm_hidden, 1) if self.use_tatt else None
        self.drop = nn.Dropout(dropout_p)

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + region_emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1)
        )

    def forward(self, x, reg_idx):
        # x: (B,T,D)
        B, T, D = x.shape

        x = x.permute(0, 2, 1)                   # (B,D,T)
        x = torch.relu(self.bn1(self.conv1(x)))  # (B,C,T)
        x = x.permute(0, 2, 1)                   # (B,T,C)

        stage_ids = torch.arange(self.time_steps, device=x.device).unsqueeze(0).repeat(B, 1)
        s_emb = self.stage_emb(stage_ids)        # (B,T,S)

        lstm_in = torch.cat([x, s_emb], dim=-1)  # (B,T,C+S)
        lstm_out, _ = self.lstm(lstm_in)         # (B,T,H)
        lstm_out = self.drop(lstm_out)

        if self.use_tatt:
            logits = self.att_fc(lstm_out).squeeze(-1)  # (B,T)
            tau = max(self.tau_time, 1e-6)
            att_w = torch.softmax(logits / tau, dim=1)  # (B,T)
            context = torch.sum(lstm_out * att_w.unsqueeze(-1), dim=1)  # (B,H)
        else:
            context = lstm_out[:, -1, :]  # last pooling

        context = self.drop(context)

        r_emb = self.region_emb(reg_idx)         # (B,R)
        fused = torch.cat([context, r_emb], dim=-1)
        out = self.head(fused)
        return out


# =========================
# 6) Train / predict
# =========================
def train_one_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total, n = 0.0, 0
    for xb, rb, yb in loader:
        xb, rb, yb = xb.to(device), rb.to(device), yb.to(device)
        optim.zero_grad(set_to_none=True)
        pred = model(xb, rb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optim.step()
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(1, n)

@torch.no_grad()
def eval_loss(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for xb, rb, yb in loader:
        xb, rb, yb = xb.to(device), rb.to(device), yb.to(device)
        pred = model(xb, rb)
        loss = loss_fn(pred, yb)
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(1, n)

@torch.no_grad()
def predict(model, X_s: np.ndarray, reg: np.ndarray, device, batch_size=256) -> np.ndarray:
    ds = SeqDataset(X_s, np.zeros((len(X_s), 1), dtype=np.float32), reg)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    out = []
    for xb, rb, _ in dl:
        xb, rb = xb.to(device), rb.to(device)
        pred = model(xb, rb)
        out.append(pred.detach().cpu().numpy())
    return np.concatenate(out, axis=0).reshape(-1, 1)


def export_pred_excel_two_sheets(
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    res_true_train: np.ndarray,
    res_pred_train: np.ndarray,
    trd_train: np.ndarray,
    yld_true_train: np.ndarray,
    yld_pred_train: np.ndarray,
    res_true_test: np.ndarray,
    res_pred_test: np.ndarray,
    trd_test: np.ndarray,
    yld_true_test: np.ndarray,
    yld_pred_test: np.ndarray,
    out_path: str,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _build_sheet(idx, res_t, res_p, trd, y_t, y_p, split_name: str) -> pd.DataFrame:
        sub = df.iloc[idx].copy()
        if COL_NAME not in sub.columns:
            sub[COL_NAME] = ""

        out = pd.DataFrame({
            "split": split_name,
            COL_NAME: sub[COL_NAME].astype(str).values,
            COL_ID: sub[COL_ID].values,
            COL_YEAR: sub[COL_YEAR].values,

            "Residual_true": res_t.reshape(-1),
            "Residual_pred": res_p.reshape(-1),
            "Residual_err": (res_p - res_t).reshape(-1),

            "Trend": trd.reshape(-1),

            "Yield_true": y_t.reshape(-1),
            "Yield_pred": y_p.reshape(-1),
            "Yield_err": (y_p - y_t).reshape(-1),
        })

        # 可选：绝对误差列，便于排序查看
        out["AbsErr_Yield"] = np.abs(out["Yield_err"].values)
        out["AbsErr_Residual"] = np.abs(out["Residual_err"].values)

        # 让结果更直观：按绝对误差从大到小排一下（你想按 ID/YEAR 排也行）
        out = out.sort_values(["AbsErr_Yield"], ascending=False).reset_index(drop=True)
        return out

    df_train = _build_sheet(idx_train, res_true_train, res_pred_train, trd_train, yld_true_train, yld_pred_train, "train")
    df_test  = _build_sheet(idx_test,  res_true_test,  res_pred_test,  trd_test,  yld_true_test,  yld_pred_test,  "test")

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df_train.to_excel(w, sheet_name="train", index=False)
        df_test.to_excel(w, sheet_name="test", index=False)

    print(f"[EXPORT] Per-sample predictions saved -> {out_path}")


# =========================
# 7) One run per mode (Target=Residual; Eval=Yield recon)
# =========================
def run_one_mode(
    mode: str,
    df: pd.DataFrame,
    X: np.ndarray,
    y_res: np.ndarray,
    y_yld: np.ndarray,
    trend: np.ndarray,
    reg: np.ndarray,
    suffixes: List[str],
    device
) -> Dict:

    idx_all = np.arange(len(df))
    strat = df[COL_YEAR].values if USE_STRATIFY_YEAR else None

    idx_tr, idx_te = train_test_split(
        idx_all, test_size=TEST_SIZE, random_state=SEED, shuffle=True, stratify=strat
    )
    strat_tr = df.iloc[idx_tr][COL_YEAR].values if USE_STRATIFY_YEAR else None
    idx_train, idx_val = train_test_split(
        idx_tr, test_size=VAL_SIZE, random_state=SEED, shuffle=True, stratify=strat_tr
    )
    idx_test = idx_te

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    r_train, r_val, r_test = reg[idx_train], reg[idx_val], reg[idx_test]

    # targets
    res_train, res_val, res_test = y_res[idx_train], y_res[idx_val], y_res[idx_test]
    yld_train, yld_val, yld_test = y_yld[idx_train], y_yld[idx_val], y_yld[idx_test]
    trd_train, trd_val, trd_test = trend[idx_train], trend[idx_val], trend[idx_test]

    # scale X
    scalerX = StandardScaler()
    scalerX.fit(X_train.reshape(-1, X_train.shape[2]))
    X_train_s = scalerX.transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape).astype(np.float32)
    X_val_s   = scalerX.transform(X_val.reshape(-1, X_val.shape[2])).reshape(X_val.shape).astype(np.float32)
    X_test_s  = scalerX.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape).astype(np.float32)

    # scale y (Residual)
    scalerY = StandardScaler()
    scalerY.fit(res_train)
    res_train_s = scalerY.transform(res_train).astype(np.float32)
    res_val_s   = scalerY.transform(res_val).astype(np.float32)

    dl_train = DataLoader(SeqDataset(X_train_s, res_train_s, r_train), batch_size=BATCH_SIZE, shuffle=True)
    dl_val   = DataLoader(SeqDataset(X_val_s,   res_val_s,   r_val),   batch_size=BATCH_SIZE, shuffle=False)

    model = CNN_LSTM_TimeAtt(
        in_dim=X_train_s.shape[2],
        num_regions=int(reg.max()) + 1,
        time_steps=len(suffixes),
        mode=mode,
        dropout_p=0.5,
        tau_time=TAU_TIME,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    print(f"\n========== Start training (target=Residual) | MODE={mode} | tau={TAU_TIME} ==========")

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, dl_train, optim, loss_fn, device)
        va_loss = eval_loss(model, dl_val, loss_fn, device)

        if ep == 1 or ep % 20 == 0:
            print(f"Epoch {ep:03d} | Train MSE(scaled)={tr_loss:.4f} | Val MSE(scaled)={va_loss:.4f}")

        if va_loss < best_val - MIN_DELTA:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stop at epoch {ep}, best val MSE(scaled)={best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Predict Residual (raw) ----
    res_pred_train = scalerY.inverse_transform(predict(model, X_train_s, r_train, device))
    res_pred_val   = scalerY.inverse_transform(predict(model, X_val_s,   r_val,   device))
    res_pred_test  = scalerY.inverse_transform(predict(model, X_test_s,  r_test,  device))

    # ---- Reconstruct Yield: Yield_pred = Residual_pred + Trend(raw) ----
    yld_pred_train = res_pred_train + trd_train
    yld_pred_val   = res_pred_val   + trd_val
    yld_pred_test  = res_pred_test  + trd_test

    # ---- Metrics: Residual + Reconstructed Yield ----
    m_res_tr = metrics(res_train, res_pred_train)
    m_res_va = metrics(res_val,   res_pred_val)
    m_res_te = metrics(res_test,  res_pred_test)

    m_yld_tr = metrics(yld_train, yld_pred_train)
    m_yld_va = metrics(yld_val,   yld_pred_val)
    m_yld_te = metrics(yld_test,  yld_pred_test)

    print("\n===== Evaluation (Residual) =====")
    print(f"[TRAIN] RMSE={m_res_tr['RMSE']:.4f}, MAE={m_res_tr['MAE']:.4f}, R²={m_res_tr['R2']:.4f}")
    print(f"[VAL  ] RMSE={m_res_va['RMSE']:.4f}, MAE={m_res_va['MAE']:.4f}, R²={m_res_va['R2']:.4f}")
    print(f"[TEST ] RMSE={m_res_te['RMSE']:.4f}, MAE={m_res_te['MAE']:.4f}, R²={m_res_te['R2']:.4f}")

    print("\n===== Evaluation (Reconstructed Yield = Residual_pred + Trend) =====")
    print(f"[TRAIN] RMSE={m_yld_tr['RMSE']:.4f}, MAE={m_yld_tr['MAE']:.4f}, R²={m_yld_tr['R2']:.4f}")
    print(f"[VAL  ] RMSE={m_yld_va['RMSE']:.4f}, MAE={m_yld_va['MAE']:.4f}, R²={m_yld_va['R2']:.4f}")
    print(f"[TEST ] RMSE={m_yld_te['RMSE']:.4f}, MAE={m_yld_te['MAE']:.4f}, R²={m_yld_te['R2']:.4f}")

    # ---- Export per-sample predictions (train/test) ----
    if EXPORT_PRED_EXCEL:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(EXPORT_DIR, f"pred_samples_{mode}_tau{TAU_TIME}_{stamp}.xlsx")
        export_pred_excel_two_sheets(
            df=df,
            idx_train=idx_train,
            idx_test=idx_test,

            res_true_train=res_train,
            res_pred_train=res_pred_train,
            trd_train=trd_train,
            yld_true_train=yld_train,
            yld_pred_train=yld_pred_train,

            res_true_test=res_test,
            res_pred_test=res_pred_test,
            trd_test=trd_test,
            yld_true_test=yld_test,
            yld_pred_test=yld_pred_test,

            out_path=out_path,
        )

    # 返回：以“重构产量”的 test 指标为主用于消融排序
    return {
        "mode": mode,
        "tau_time": float(TAU_TIME),
        "seed": int(SEED),
        "stratify_year": bool(USE_STRATIFY_YEAR),
        "add_year_to_x": bool(ADD_YEAR_TO_X),
        "best_val_mse_scaled": float(best_val),

        # Residual metrics
        "test_res_r2": m_res_te["R2"],
        "test_res_rmse": m_res_te["RMSE"],
        "test_res_mae": m_res_te["MAE"],

        # Reconstructed yield metrics (MAIN)
        "train_yld_r2": m_yld_tr["R2"], "val_yld_r2": m_yld_va["R2"], "test_yld_r2": m_yld_te["R2"],
        "train_yld_rmse": m_yld_tr["RMSE"], "val_yld_rmse": m_yld_va["RMSE"], "test_yld_rmse": m_yld_te["RMSE"],
        "train_yld_mae": m_yld_tr["MAE"], "val_yld_mae": m_yld_va["MAE"], "test_yld_mae": m_yld_te["MAE"],
    }


# =========================
# 8) Main
# =========================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    need = [COL_ID, COL_YEAR, COL_RESID, COL_TREND, COL_YIELD]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # region index
    reg_ids = df[COL_ID].astype(int).to_numpy()
    uniq = sorted(list(set(reg_ids.tolist())))
    reg_map = {rid: i for i, rid in enumerate(uniq)}
    reg = np.array([reg_map[r] for r in reg_ids], dtype=np.int64)
    print("Regions:", len(uniq))

    X, y_res, y_yld, trend, used_vars, suffixes = build_X_and_targets(df)
    print("X shape:", X.shape, "| used_vars:", len(used_vars), "| T:", len(suffixes))
    print("ADD_YEAR_TO_X:", ADD_YEAR_TO_X, "| STRATIFY_YEAR:", USE_STRATIFY_YEAR)
    print("EXPORT_PRED_EXCEL:", EXPORT_PRED_EXCEL, "| EXPORT_DIR:", EXPORT_DIR)

    # modes: FULL / NO_TATT（PLAIN 合并）
    modes = ["FULL", "NO_TATT"] if RUN_ALL_MODES else [MODE.upper()]
    rows = []
    for m in modes:
        rows.append(run_one_mode(m, df, X, y_res, y_yld, trend, reg, suffixes, device))

    # summary: 按“重构 Yield”的 TEST R2 排序
    print("\n" + "=" * 96)
    print("Ablation Summary (sorted by TEST R2 of Reconstructed Yield desc)")
    print("=" * 96)
    out = pd.DataFrame(rows).sort_values(["test_yld_r2", "test_yld_rmse"], ascending=[False, True])

    show_cols = [
        "mode", "tau_time",
        "test_yld_r2", "test_yld_rmse", "test_yld_mae",
        "val_yld_r2", "val_yld_rmse",
        "train_yld_r2", "train_yld_rmse",
        "test_res_r2", "test_res_rmse"
    ]
    print(out[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()

# ablation_fit_yield_TimeAtt_mergePlain.py
# 直接拟合 yield（不涉及 Residual/Trend）
# 模型：CNN + LSTM + (Time-Att可选) + Region Emb + Stage Emb
# 变更点（本版）：
#   ✅ 导出训练集/测试集每样本结果到一个 Excel（两个Sheet：train/test）
#   ✅ 仍然打印各模式 Train/Val/Test 的 RMSE/MAE/R2，并汇总排序

import os
import random
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================
# 0) Seed & device
# =========================
def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================
# 1) Config
# =========================
DATA_PATH = r"D:\Shap_多维\分解时间序列\SIF_FPAR_EVI2_Tmpmin_Tmpmax_SM_汇总.xlsx"
SHEET_NAME = 0

COL_NAME = "NAME"
COL_ID   = "ID"
COL_YEAR = "YEAR"
COL_YIELD = "yield"

# Feature processing
ADD_YEAR_TO_X = True

# Split
SEED = 43
USE_STRATIFY_YEAR = True
TEST_SIZE = 0.20
VAL_SIZE  = 0.125  # from remaining -> about 7:1:2

# Train
BATCH_SIZE = 64
EPOCHS = 400
LR = 5e-4
WEIGHT_DECAY = 5e-3
PATIENCE = 60
MIN_DELTA = 1e-6

RUN_ALL_MODES = True
MODE = "FULL"  # FULL / NO_TATT

# Time-att sharpening
TAU_TIME = 1.0

# Export per-sample predictions
EXPORT_PRED_EXCEL = True
EXPORT_DIR = "outputs_noam"


# =========================
# 2) Utils
# =========================
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# =========================
# 3) Suffix / vars detection (6-digit numeric)
# =========================
def detect_suffixes_6digit_numeric(cols: List[str]) -> List[str]:
    suf = set()
    for c in cols:
        if "_" not in c:
            continue
        tail = c.split("_")[-1]
        if len(tail) == 6 and tail.isdigit():
            suf.add(tail)
    return sorted(list(suf))

def infer_used_vars(df: pd.DataFrame, suffixes: List[str]) -> List[str]:
    meta = {COL_NAME, COL_ID, COL_YEAR, COL_YIELD}
    vars_set = set()
    for c in df.columns:
        if c in meta:
            continue
        if "_" not in c:
            continue
        tail = c.split("_")[-1]
        if tail in suffixes:
            v = c.rsplit("_", 1)[0]
            vars_set.add(v)
    used = sorted(list(vars_set))
    if len(used) == 0:
        raise ValueError("No feature variables detected from suffixes.")
    return used

def build_X_and_y(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    X: (N,T,D) (optionally with YEAR as last channel)
    y: (N,1) yield
    """
    suffixes = detect_suffixes_6digit_numeric(list(df.columns))
    if len(suffixes) == 0:
        raise ValueError("No 6-digit numeric time suffixes detected (e.g., _030226).")

    used_vars = infer_used_vars(df, suffixes)

    N = len(df)
    T = len(suffixes)
    D = len(used_vars) + (1 if ADD_YEAR_TO_X else 0)

    X = np.zeros((N, T, D), dtype=np.float32)
    for ti, suf in enumerate(suffixes):
        cols_t = [f"{v}_{suf}" for v in used_vars]
        missing = [c for c in cols_t if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns at time={suf}: {missing[:10]}")
        X[:, ti, :len(used_vars)] = df[cols_t].to_numpy(dtype=np.float32)

    if ADD_YEAR_TO_X:
        year_vals = df[COL_YEAR].to_numpy(dtype=np.float32).reshape(-1, 1)
        X[:, :, -1] = np.repeat(year_vals, T, axis=1)

    y = df[COL_YIELD].to_numpy(dtype=np.float32).reshape(-1, 1)
    return X, y, used_vars, suffixes


# =========================
# 4) Dataset
# =========================
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, reg_idx: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.reg = torch.tensor(reg_idx, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.reg[i], self.y[i]


# =========================
# 5) Model: CNN + LSTM + (Time-Att optional) + Region/Stage Emb
# =========================
class CNN_LSTM_TimeAtt(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_regions: int,
        time_steps: int,
        mode: str = "FULL",
        conv_channels: int = 16,
        lstm_hidden: int = 32,
        region_emb_dim: int = 4,
        stage_emb_dim: int = 8,
        dropout_p: float = 0.5,
        tau_time: float = 1.0,
    ):
        super().__init__()
        self.mode = mode.upper()
        assert self.mode in {"FULL", "NO_TATT"}

        self.use_tatt = (self.mode == "FULL")
        self.time_steps = int(time_steps)
        self.tau_time = float(tau_time)

        self.region_emb = nn.Embedding(num_regions, region_emb_dim)
        self.stage_emb  = nn.Embedding(time_steps, stage_emb_dim)

        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=conv_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(conv_channels)

        self.lstm = nn.LSTM(
            input_size=conv_channels + stage_emb_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.att_fc = nn.Linear(lstm_hidden, 1) if self.use_tatt else None
        self.drop = nn.Dropout(dropout_p)

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + region_emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1)
        )

    def forward(self, x, reg_idx):
        # x: (B,T,D)
        B, T, D = x.shape

        x = x.permute(0, 2, 1)                   # (B,D,T)
        x = torch.relu(self.bn1(self.conv1(x)))  # (B,C,T)
        x = x.permute(0, 2, 1)                   # (B,T,C)

        stage_ids = torch.arange(self.time_steps, device=x.device).unsqueeze(0).repeat(B, 1)
        s_emb = self.stage_emb(stage_ids)        # (B,T,S)

        lstm_in = torch.cat([x, s_emb], dim=-1)  # (B,T,C+S)
        lstm_out, _ = self.lstm(lstm_in)         # (B,T,H)
        lstm_out = self.drop(lstm_out)

        if self.use_tatt:
            logits = self.att_fc(lstm_out).squeeze(-1)  # (B,T)
            tau = max(self.tau_time, 1e-6)
            att_w = torch.softmax(logits / tau, dim=1)  # (B,T)
            context = torch.sum(lstm_out * att_w.unsqueeze(-1), dim=1)  # (B,H)
        else:
            context = lstm_out[:, -1, :]  # last pooling

        context = self.drop(context)

        r_emb = self.region_emb(reg_idx)         # (B,R)
        fused = torch.cat([context, r_emb], dim=-1)
        out = self.head(fused)
        return out


# =========================
# 6) Train / predict
# =========================
def train_one_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total, n = 0.0, 0
    for xb, rb, yb in loader:
        xb, rb, yb = xb.to(device), rb.to(device), yb.to(device)
        optim.zero_grad(set_to_none=True)
        pred = model(xb, rb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optim.step()
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(1, n)

@torch.no_grad()
def eval_loss(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for xb, rb, yb in loader:
        xb, rb, yb = xb.to(device), rb.to(device), yb.to(device)
        pred = model(xb, rb)
        loss = loss_fn(pred, yb)
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(1, n)

@torch.no_grad()
def predict(model, X_s: np.ndarray, reg: np.ndarray, device, batch_size=256) -> np.ndarray:
    ds = SeqDataset(X_s, np.zeros((len(X_s), 1), dtype=np.float32), reg)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    out = []
    for xb, rb, _ in dl:
        xb, rb = xb.to(device), rb.to(device)
        pred = model(xb, rb)
        out.append(pred.detach().cpu().numpy())
    return np.concatenate(out, axis=0).reshape(-1, 1)


def export_pred_excel_two_sheets(
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    out_path: str,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _build_sheet(idx, y_t, y_p, split_name: str) -> pd.DataFrame:
        sub = df.iloc[idx].copy()
        if COL_NAME not in sub.columns:
            sub[COL_NAME] = ""

        out = pd.DataFrame({
            "split": split_name,
            COL_NAME: sub[COL_NAME].astype(str).values,
            COL_ID: sub[COL_ID].values,
            COL_YEAR: sub[COL_YEAR].values,

            "Yield_true": y_t.reshape(-1),
            "Yield_pred": y_p.reshape(-1),
            "Yield_err": (y_p - y_t).reshape(-1),
        })
        out["AbsErr_Yield"] = np.abs(out["Yield_err"].values)
        out = out.sort_values(["AbsErr_Yield"], ascending=False).reset_index(drop=True)
        return out

    df_train = _build_sheet(idx_train, y_true_train, y_pred_train, "train")
    df_test  = _build_sheet(idx_test,  y_true_test,  y_pred_test,  "test")

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df_train.to_excel(w, sheet_name="train", index=False)
        df_test.to_excel(w, sheet_name="test", index=False)

    print(f"[EXPORT] Per-sample predictions saved -> {out_path}")


# =========================
# 7) One run per mode (Target=Yield)
# =========================
def run_one_mode(mode: str, df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
                 reg: np.ndarray, suffixes: List[str], device) -> Dict:

    idx_all = np.arange(len(df))
    strat = df[COL_YEAR].values if USE_STRATIFY_YEAR else None

    idx_tr, idx_te = train_test_split(
        idx_all, test_size=TEST_SIZE, random_state=SEED, shuffle=True, stratify=strat
    )
    strat_tr = df.iloc[idx_tr][COL_YEAR].values if USE_STRATIFY_YEAR else None
    idx_train, idx_val = train_test_split(
        idx_tr, test_size=VAL_SIZE, random_state=SEED, shuffle=True, stratify=strat_tr
    )
    idx_test = idx_te

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    r_train, r_val, r_test = reg[idx_train], reg[idx_val], reg[idx_test]

    # scale X
    scalerX = StandardScaler()
    scalerX.fit(X_train.reshape(-1, X_train.shape[2]))
    X_train_s = scalerX.transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape).astype(np.float32)
    X_val_s   = scalerX.transform(X_val.reshape(-1, X_val.shape[2])).reshape(X_val.shape).astype(np.float32)
    X_test_s  = scalerX.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape).astype(np.float32)

    # scale y (yield)
    scalerY = StandardScaler()
    scalerY.fit(y_train)
    y_train_s = scalerY.transform(y_train).astype(np.float32)
    y_val_s   = scalerY.transform(y_val).astype(np.float32)

    dl_train = DataLoader(SeqDataset(X_train_s, y_train_s, r_train), batch_size=BATCH_SIZE, shuffle=True)
    dl_val   = DataLoader(SeqDataset(X_val_s,   y_val_s,   r_val),   batch_size=BATCH_SIZE, shuffle=False)

    model = CNN_LSTM_TimeAtt(
        in_dim=X_train_s.shape[2],
        num_regions=int(reg.max()) + 1,
        time_steps=len(suffixes),
        mode=mode,
        dropout_p=0.5,
        tau_time=TAU_TIME,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    print(f"\n========== Start training (target=Yield) | MODE={mode} | tau={TAU_TIME} ==========")

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, dl_train, optim, loss_fn, device)
        va_loss = eval_loss(model, dl_val, loss_fn, device)

        if ep == 1 or ep % 20 == 0:
            print(f"Epoch {ep:03d} | Train MSE(scaled)={tr_loss:.4f} | Val MSE(scaled)={va_loss:.4f}")

        if va_loss < best_val - MIN_DELTA:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stop at epoch {ep}, best val MSE(scaled)={best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # predict (inverse scale to raw yield)
    pred_y_train = scalerY.inverse_transform(predict(model, X_train_s, r_train, device))
    pred_y_val   = scalerY.inverse_transform(predict(model, X_val_s,   r_val,   device))
    pred_y_test  = scalerY.inverse_transform(predict(model, X_test_s,  r_test,  device))

    m_tr = metrics(y_train, pred_y_train)
    m_va = metrics(y_val,   pred_y_val)
    m_te = metrics(y_test,  pred_y_test)

    print("\n===== Evaluation (Yield) =====")
    print(f"[TRAIN] RMSE={m_tr['RMSE']:.4f}, MAE={m_tr['MAE']:.4f}, R²={m_tr['R2']:.4f}")
    print(f"[VAL  ] RMSE={m_va['RMSE']:.4f}, MAE={m_va['MAE']:.4f}, R²={m_va['R2']:.4f}")
    print(f"[TEST ] RMSE={m_te['RMSE']:.4f}, MAE={m_te['MAE']:.4f}, R²={m_te['R2']:.4f}")

    # ---- Export per-sample predictions (train/test) ----
    if EXPORT_PRED_EXCEL:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(EXPORT_DIR, f"pred_samples_yield_{mode}_tau{TAU_TIME}_{stamp}.xlsx")
        export_pred_excel_two_sheets(
            df=df,
            idx_train=idx_train,
            idx_test=idx_test,
            y_true_train=y_train,
            y_pred_train=pred_y_train,
            y_true_test=y_test,
            y_pred_test=pred_y_test,
            out_path=out_path,
        )

    return {
        "mode": mode,
        "tau_time": float(TAU_TIME),
        "seed": int(SEED),
        "stratify_year": bool(USE_STRATIFY_YEAR),
        "add_year_to_x": bool(ADD_YEAR_TO_X),
        "best_val_mse_scaled": float(best_val),

        "train_r2": m_tr["R2"], "val_r2": m_va["R2"], "test_r2": m_te["R2"],
        "train_rmse": m_tr["RMSE"], "val_rmse": m_va["RMSE"], "test_rmse": m_te["RMSE"],
        "train_mae": m_tr["MAE"], "val_mae": m_va["MAE"], "test_mae": m_te["MAE"],
    }


# =========================
# 8) Main
# =========================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    need = [COL_ID, COL_YEAR, COL_YIELD]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # region index
    reg_ids = df[COL_ID].astype(int).to_numpy()
    uniq = sorted(list(set(reg_ids.tolist())))
    reg_map = {rid: i for i, rid in enumerate(uniq)}
    reg = np.array([reg_map[r] for r in reg_ids], dtype=np.int64)
    print("Regions:", len(uniq))

    X, y, used_vars, suffixes = build_X_and_y(df)
    print("X shape:", X.shape, "| used_vars:", len(used_vars), "| T:", len(suffixes))
    print("ADD_YEAR_TO_X:", ADD_YEAR_TO_X, "| STRATIFY_YEAR:", USE_STRATIFY_YEAR)
    print("EXPORT_PRED_EXCEL:", EXPORT_PRED_EXCEL, "| EXPORT_DIR:", EXPORT_DIR)

    # modes: FULL / NO_TATT（PLAIN 合并）
    modes = ["FULL", "NO_TATT"] if RUN_ALL_MODES else [MODE.upper()]
    rows = []
    for m in modes:
        rows.append(run_one_mode(m, df, X, y, reg, suffixes, device))

    # summary
    print("\n" + "=" * 84)
    print("Ablation Summary (sorted by TEST R2 desc)")
    print("=" * 84)
    out = pd.DataFrame(rows).sort_values(["test_r2", "test_rmse"], ascending=[False, True])
    show_cols = ["mode", "tau_time", "test_r2", "test_rmse", "test_mae",
                 "val_r2", "val_rmse", "train_r2", "train_rmse"]
    print(out[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
