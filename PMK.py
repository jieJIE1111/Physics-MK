import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from scipy.signal import welch
import os
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, max_error

# ==========================================
# 0. 环境配置与全局设置
# ==========================================
USE_MAMBA = False
try:
    from mamba_ssm import Mamba
    USE_MAMBA = True
    print("✅ [Environment] 成功导入官方 Mamba (mamba_ssm)！")
except ImportError:
    print("⚠️ [Environment] 未找到 mamba_ssm，将自动降级使用 GRU。")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    """设置随机种子确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@dataclass
class Config:
    """配置类"""
    base_dir: str = "/mnt/c/Users/shiki/Desktop/ADVEI"

    # --- 模型超参数 ---
    seq_len: int = 50
    hidden_dim: int = 32
    batch_size: int = 1024

    # --- 训练策略 (三阶段) ---
    epochs_phase1: int = 50
    epochs_phase2: int = 50
    epochs_phase3: int = 50

    # --- 单阶段训练总 epoch ---
    epochs_single: int = 150

    # --- Early Stopping ---
    patience: int = 15
    min_delta: float = 1e-5

    # --- 数据划分 ---
    train_ratio: float = 0.7

    # --- 归一化参数 ---
    norm_rpm: float = 100.0
    norm_load: float = 100.0
    norm_press: float = 3.0
    norm_vib: float = 10.0

    # --- 温度参数 ---
    temp_base: float = 30.0
    temp_scale: float = 40.0

    # --- 物理稳定性限制 ---
    max_dt_clamp: float = 0.15

    # --- 输入维度 ---
    ctx_dim: int = 6
    dyn_dim: int = 5

    # 路径字段
    path_train: str = field(default="", init=False)
    path_test: str = field(default="", init=False)
    save_dir: str = field(default="", init=False)

    def __post_init__(self):
        self.path_train = os.path.join(self.base_dir, "Voyage1_Train.csv")
        self.path_test = os.path.join(self.base_dir, "Voyage2_Test.csv")
        self.save_dir = os.path.join(self.base_dir, "Paper_Visualizations_V3")


cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)


# ==========================================
# 1. Early Stopping 类
# ==========================================
class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        return self.early_stop

    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0


# ==========================================
# 2. Training Data Collector
# ==========================================
class TrainingDataCollector:
    def __init__(self, mode_name: str):
        self.mode_name = mode_name
        self.training_history = {
            "epoch": [], "phase": [], "train_loss": [], "val_loss": [],
            "lr_phy": [], "lr_mamba": []
        }

    def record_epoch(self, epoch, phase, train_loss, val_loss, lr_phy, lr_mamba):
        self.training_history["epoch"].append(epoch)
        self.training_history["phase"].append(phase)
        self.training_history["train_loss"].append(train_loss)
        self.training_history["val_loss"].append(val_loss)
        self.training_history["lr_phy"].append(lr_phy)
        self.training_history["lr_mamba"].append(lr_mamba)

    def save(self, save_dir: str):
        mode_dir = os.path.join(save_dir, "collected_data", self.mode_name)
        os.makedirs(mode_dir, exist_ok=True)
        df = pd.DataFrame(self.training_history)
        df.to_csv(os.path.join(mode_dir, "training_history.csv"), index=False)
        print(f"   💾 Training history saved: {self.mode_name}/training_history.csv")


# ==========================================
# 3. Fast KAN Layer
# ==========================================
class FastKANLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, grid_size: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size

        self.base_weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.base_activation = nn.SiLU()

        h = (1 / grid_size) * 2
        grid = torch.arange(-1, 1 + h + 1e-5, h)
        self.register_buffer('grid', grid)

        self.spline_weight = nn.Parameter(torch.Tensor(output_dim, input_dim * len(grid)))

        nn.init.xavier_uniform_(self.base_weight, gain=0.1)
        nn.init.uniform_(self.spline_weight, -0.01, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = F.linear(self.base_activation(x), self.base_weight)
        x_uns = x.unsqueeze(-1)
        target_grid = self.grid.view(1, 1, -1)
        bases = torch.exp(-torch.pow((x_uns - target_grid) / 0.5, 2))
        bases = bases.view(x.shape[0], -1)
        spline_out = F.linear(bases, self.spline_weight)
        return base_output + spline_out


# ==========================================
# 4. MLP Layer (削弱版 - 单层)
# ==========================================
class MLPLayer(nn.Module):
    """单层 MLP - 参数量少于 KAN"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.act = nn.SiLU()
        nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x))


# ==========================================
# 5. Mamba/GRU 补偿器 (GRU 削弱版)
# ==========================================
class MambaCompensator(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 32, d_state: int = 16,
                 n_layers: int = 2, use_mamba: bool = True):
        super().__init__()
        self.use_mamba = use_mamba and USE_MAMBA
        self.embedding = nn.Linear(input_dim, d_model)

        if self.use_mamba:
            self.layers = nn.ModuleList()
            for _ in range(n_layers):
                self.layers.append(
                    Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                )
        else:
            # GRU 削弱版: 单层 + 更小的隐藏维度
            self.rnn = nn.GRU(d_model, d_model // 2, num_layers=1, batch_first=True)
            self.proj = nn.Linear(d_model // 2, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x_seq)
        if self.use_mamba:
            for layer in self.layers:
                x = layer(x)
        else:
            x, _ = self.rnn(x)
            x = self.proj(x)  # d_model//2 -> d_model
        x = self.norm(x)
        return self.head(x)


# ==========================================
# 6. 物理参数估计器
# ==========================================
class PhysicsParameterEstimator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, use_kan: bool = True):
        super().__init__()
        self.use_kan = use_kan

        if use_kan:
            self.shared = FastKANLayer(input_dim, hidden_dim)
            self.alpha_head = FastKANLayer(hidden_dim, 1)
            self.beta_head = FastKANLayer(hidden_dim, 1)
            self.eta_gen_head = FastKANLayer(hidden_dim, 1)
            self.k_cool_head = FastKANLayer(hidden_dim, 1)
            self.tau_head = FastKANLayer(hidden_dim, 1)
        else:
            self.shared = MLPLayer(input_dim, hidden_dim)
            self.alpha_head = MLPLayer(hidden_dim, 1)
            self.beta_head = MLPLayer(hidden_dim, 1)
            self.eta_gen_head = MLPLayer(hidden_dim, 1)
            self.k_cool_head = MLPLayer(hidden_dim, 1)
            self.tau_head = MLPLayer(hidden_dim, 1)

    def forward(self, x_ctx: torch.Tensor):
        feat = self.shared(x_ctx)
        alpha = torch.sigmoid(self.alpha_head(feat)) * 1.5 + 0.8
        beta = torch.sigmoid(self.beta_head(feat)) * 1.0 + 0.5
        eta_gen = F.softplus(self.eta_gen_head(feat)) + 1e-6
        k_cool = F.softplus(self.k_cool_head(feat)) + 1e-6
        tau = torch.sigmoid(self.tau_head(feat)) * 0.05 + 0.001
        return alpha, beta, eta_gen, k_cool, tau


# ==========================================
# 7. PM-KMNet 消融版模型 (V3.2 削弱版)
# ==========================================
class PM_KMNet_Ablation(nn.Module):
    """
    消融实验模型 (V3.2 - 削弱版 baseline)

    ablation_mode:
        - "Proposed": 完整模型 (KAN + Mamba)
        - "No_KAN": 用削弱版 MLP 替代 KAN
        - "No_Mamba": 用削弱版 GRU 替代 Mamba
        - "Physics_Only": 仅物理分支
        - "Data_Only": 仅数据分支 (预测ΔT并递推)
        - "No_Progressive": 完整模型，端到端训练不分阶段
    """

    def __init__(self, ablation_mode: str = "Proposed"):
        super().__init__()
        self.ablation_mode = ablation_mode

        use_kan = (ablation_mode not in ["No_KAN", "Data_Only"])
        use_mamba = (ablation_mode not in ["No_Mamba", "Physics_Only"])
        use_physics = (ablation_mode != "Data_Only")
        use_data = (ablation_mode != "Physics_Only")

        self.use_physics = use_physics
        self.use_data = use_data

        if use_physics:
            self.param_estimator = PhysicsParameterEstimator(
                input_dim=cfg.ctx_dim,
                hidden_dim=cfg.hidden_dim,
                use_kan=use_kan
            )

        if use_data:
            self.mamba_layer = MambaCompensator(
                input_dim=cfg.ctx_dim + cfg.dyn_dim,
                d_model=cfg.hidden_dim,
                use_mamba=use_mamba
            )

    def forward(self, x_ctx_seq, rpm_seq, load_seq, press_seq, vib_seq, t_in_seq,
                t_init_val: Optional[torch.Tensor] = None):
        b, s, _ = rpm_seq.shape

        # 统一处理初始温度
        if t_init_val is not None:
            T0 = t_init_val
        else:
            T0 = t_in_seq[:, 0, :]

        # ==========================================
        # Data_Only: 简洁高效的 ΔT 递推
        # ==========================================
        if self.ablation_mode == "Data_Only":
            mamba_input = torch.cat([
                x_ctx_seq, rpm_seq, load_seq, press_seq, vib_seq, t_in_seq
            ], dim=-1)

            dT_seq = self.mamba_layer(mamba_input)
            dT_seq = torch.clamp(dT_seq, -cfg.max_dt_clamp, cfg.max_dt_clamp)

            outputs = []
            T_curr = T0
            for t in range(s):
                T_curr = T_curr + dT_seq[:, t, :]
                outputs.append(T_curr)

            pred = torch.stack(outputs, dim=1)
            zeros = torch.zeros_like(pred)
            return pred, zeros, zeros, dT_seq

        # ==========================================
        # 其他模式
        # ==========================================
        if not self.use_physics:
            raise RuntimeError(f"Physics branch disabled but reached physics path. Mode: {self.ablation_mode}")

        # Mamba 残差分支
        if self.use_data:
            mamba_input = torch.cat([
                x_ctx_seq, rpm_seq, load_seq, press_seq, vib_seq, t_in_seq
            ], dim=-1)
            p_resid_seq = self.mamba_layer(mamba_input)
        else:
            p_resid_seq = torch.zeros(b, s, 1, device=rpm_seq.device)

        # 物理参数估计
        flat_ctx = x_ctx_seq.reshape(b * s, -1)
        alpha, beta, eta_gen, k_cool, tau = self.param_estimator(flat_ctx)
        alpha = alpha.view(b, s, 1)
        beta = beta.view(b, s, 1)
        eta_gen = eta_gen.view(b, s, 1)
        k_cool = k_cool.view(b, s, 1)
        tau = tau.view(b, s, 1)

        # 物理递推
        outputs = []
        q_gen_hist = []
        q_out_hist = []
        T_curr = T0

        for t in range(s):
            rpm_t = torch.clamp(rpm_seq[:, t, :], min=1e-5)
            load_t = torch.clamp(load_seq[:, t, :], min=1e-5)

            Q_gen = eta_gen[:, t, :] * torch.pow(rpm_t, alpha[:, t, :]) * torch.pow(load_t, beta[:, t, :])
            delta_T = T_curr - t_in_seq[:, t, :]
            Q_out = k_cool[:, t, :] * delta_T

            P_resid = p_resid_seq[:, t, :] if self.use_data else torch.zeros_like(Q_gen)
            Net_Power = Q_gen - Q_out + P_resid

            dT = tau[:, t, :] * Net_Power
            dT = torch.clamp(dT, -cfg.max_dt_clamp, cfg.max_dt_clamp)
            T_next = T_curr + dT

            outputs.append(T_next)
            q_gen_hist.append(Q_gen)
            q_out_hist.append(Q_out)
            T_curr = T_next

        pred = torch.stack(outputs, dim=1)
        q_gen = torch.stack(q_gen_hist, dim=1)
        q_out = torch.stack(q_out_hist, dim=1)

        return pred, q_gen, q_out, p_resid_seq

    def get_all_parameters_for_batch(self, x_ctx_seq, rpm_seq, load_seq):
        b, s, _ = x_ctx_seq.shape

        if not self.use_physics:
            zeros = torch.zeros(b, s, 1, device=x_ctx_seq.device)
            return {'alpha': zeros, 'beta': zeros, 'eta_gen': zeros, 'k_cool': zeros, 'tau': zeros}

        flat_ctx = x_ctx_seq.reshape(b * s, -1)
        alpha, beta, eta_gen, k_cool, tau = self.param_estimator(flat_ctx)

        return {
            'alpha': alpha.view(b, s, 1),
            'beta': beta.view(b, s, 1),
            'eta_gen': eta_gen.view(b, s, 1),
            'k_cool': k_cool.view(b, s, 1),
            'tau': tau.view(b, s, 1)
        }


# ==========================================
# 8. 数据处理
# ==========================================
class StandardDataset(Dataset):
    def __init__(self, data_dict: Dict[str, np.ndarray], seq_len: int = 50, stride: int = 1):
        self.data = data_dict
        self.seq_len = seq_len
        self.stride = stride
        self.n = len(data_dict['target'])
        start_idx = max(stride, 1)
        self.indices = list(range(start_idx, self.n - self.seq_len, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        sl = self.seq_len
        ctx = self.data['ctx'][i: i + sl]
        rpm = self.data['rpm'][i: i + sl]
        load = self.data['load'][i: i + sl]
        press = self.data['press'][i: i + sl]
        vib = self.data['vib'][i: i + sl]
        t_in = self.data['t_in'][i: i + sl]
        y = self.data['target'][i: i + sl]
        t0 = self.data['target'][i - 1]
        return (
            torch.from_numpy(ctx).float(),
            torch.from_numpy(rpm).unsqueeze(-1).float(),
            torch.from_numpy(load).unsqueeze(-1).float(),
            torch.from_numpy(press).unsqueeze(-1).float(),
            torch.from_numpy(vib).unsqueeze(-1).float(),
            torch.from_numpy(t_in).unsqueeze(-1).float(),
            torch.tensor([t0], dtype=torch.float32),
            torch.from_numpy(y).unsqueeze(-1).float()
        )


class DataProcessor:
    """数据处理器 - 严格 Scaler 分离"""

    def __init__(self):
        self.ctx_cols = ['Df', 'Da', 'EeIndex8', 'Rudder', 'TrueVwr', 'Me1Sw_com_temp']
        self.ctx_scalers: Dict[str, float] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        """只用训练集拟合 scaler"""
        print("📐 Fitting scalers on TRAIN split only...")
        for c in self.ctx_cols:
            if c in df.columns:
                col_max = df[c].abs().max()
                self.ctx_scalers[c] = float(col_max) if col_max > 1e-5 else 1.0
            else:
                self.ctx_scalers[c] = 1.0
            print(f"   {c}: max = {self.ctx_scalers[c]:.4f}")
        self.fitted = True

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """用已拟合的 scaler 转换数据"""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted! Call fit() first.")

        df = df.ffill().fillna(0).copy()

        df['Norm_RPM'] = df['Me1Nms'] / cfg.norm_rpm
        df['Norm_Load'] = df['Me1Load'] / cfg.norm_load
        df['Norm_Press'] = df['Me1Main_brg_lo_in_press'] / cfg.norm_press
        df['Norm_Vib'] = df['Me1Axial_vib'] / cfg.norm_vib

        def norm_temp(t):
            return (t - cfg.temp_base) / cfg.temp_scale

        df['Norm_Tin'] = norm_temp(df['Me1Main_brg_lo_in_temp'])
        df['Norm_Target'] = norm_temp(df['Me1Thrust_bearing_pad_temp'])

        ctx_data = []
        for c in self.ctx_cols:
            if c not in df.columns:
                df[c] = 0.0
            scaled = df[c].values / self.ctx_scalers[c]
            ctx_data.append(scaled)

        ctx_array = np.stack(ctx_data, axis=1).astype(np.float32)

        return {
            "ctx": ctx_array,
            "rpm": df['Norm_RPM'].values.astype(np.float32),
            "load": df['Norm_Load'].values.astype(np.float32),
            "press": df['Norm_Press'].values.astype(np.float32),
            "vib": df['Norm_Vib'].values.astype(np.float32),
            "t_in": df['Norm_Tin'].values.astype(np.float32),
            "target": df['Norm_Target'].values.astype(np.float32)
        }

    def save_scalers(self, save_path: str):
        with open(save_path, 'w') as f:
            json.dump(self.ctx_scalers, f, indent=2)
        print(f"   💾 Scalers saved: {save_path}")

    def load_scalers(self, load_path: str):
        with open(load_path, 'r') as f:
            self.ctx_scalers = json.load(f)
        self.fitted = True
        print(f"   📂 Scalers loaded: {load_path}")

    @staticmethod
    def create_loader(data: Optional[Dict], batch_size: int, shuffle: bool = False,
                      stride: int = 1) -> Optional[DataLoader]:
        if data is None:
            return None
        ds = StandardDataset(data, seq_len=cfg.seq_len, stride=stride)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


# ==========================================
# 9. 损失函数
# ==========================================
def compute_losses(model: nn.Module, batch, mode_name: str):
    b_ctx, b_rpm, b_load, b_press, b_vib, b_tin, b_tinit, b_y = [t.to(DEVICE) for t in batch]
    pred, q_gen, q_out, p_resid = model(b_ctx, b_rpm, b_load, b_press, b_vib, b_tin, b_tinit)

    loss_mse = F.mse_loss(pred, b_y)
    loss_smooth = 2.0 * F.mse_loss(pred[:, 1:] - pred[:, :-1], b_y[:, 1:] - b_y[:, :-1])

    return loss_mse, loss_smooth, pred, q_gen, q_out, p_resid


def compute_tic(preds: np.ndarray, trues: np.ndarray) -> float:
    num = np.sqrt(np.mean((preds - trues) ** 2))
    den = np.sqrt(np.mean(preds ** 2)) + np.sqrt(np.mean(trues ** 2))
    return num / (den + 1e-7)


# ==========================================
# 10. 训练引擎
# ==========================================
def validate_epoch(model: nn.Module, loader: DataLoader, mode_name: str) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch_size = batch[0].shape[0]
            loss_mse, loss_smooth, _, _, _, _ = compute_losses(model, batch, mode_name)
            total_loss += (loss_mse + loss_smooth).item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def train_progressive(model: nn.Module, mode_name: str, loader_tr: DataLoader,
                      loader_val: DataLoader, collector: TrainingDataCollector) -> Tuple[Dict, int, float]:
    """渐进式三阶段训练"""
    param_estimator_params = list(model.param_estimator.parameters()) if model.use_physics else []
    mamba_params = list(model.mamba_layer.parameters()) if model.use_data else []

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    global_epoch = 0

    def print_table_header():
        print(f"{'Epoch':^8} | {'Phase':^8} | {'LR_phy':^10} | {'LR_mamba':^10} | "
              f"{'Train':^10} | {'Val':^10} | {'Status':^12}")
        print("-" * 85)

    def print_log_row(ep, phase, lr_p, lr_m, train_l, val_l, status=""):
        print(f"{ep:^8} | {phase:^8} | {lr_p:.2e} | {lr_m:.2e} | "
              f"{train_l:^10.6f} | {val_l:^10.6f} | {status:^12}")

    phases = [
        {"name": "Phase1", "epochs": cfg.epochs_phase1, "lr_phy": 3e-3, "lr_mamba": 1e-5,
         "phy_scale": 1.0, "mamba_scale": 0.1},
        {"name": "Phase2", "epochs": cfg.epochs_phase2, "lr_phy": 1e-4, "lr_mamba": 5e-3,
         "phy_scale": 0.1, "mamba_scale": 1.0},
        {"name": "Phase3", "epochs": cfg.epochs_phase3, "lr_phy": 5e-5, "lr_mamba": 1e-4,
         "phy_scale": 1.0, "mamba_scale": 1.0},
    ]

    for phase_info in phases:
        phase_name = phase_info["name"]
        max_epochs = phase_info["epochs"]
        lr_phy = phase_info["lr_phy"]
        lr_mamba = phase_info["lr_mamba"]

        print(f"\n🔷 [{phase_name}] Starting (max {max_epochs} epochs)")
        print_table_header()

        param_groups = []
        if param_estimator_params:
            param_groups.append({'params': param_estimator_params, 'lr': lr_phy * phase_info["phy_scale"]})
        if mamba_params:
            param_groups.append({'params': mamba_params, 'lr': lr_mamba * phase_info["mamba_scale"]})

        if not param_groups:
            continue

        optimizer = optim.AdamW(param_groups)
        early_stopper = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

        for epoch in range(max_epochs):
            global_epoch += 1
            model.train()
            epoch_train_losses = []

            for batch in loader_tr:
                optimizer.zero_grad()
                loss_mse, loss_smooth, _, q_gen, _, _ = compute_losses(model, batch, mode_name)

                if phase_name == "Phase1" and model.use_physics:
                    loss_q_smooth = 0.01 * torch.mean((q_gen[:, 1:] - q_gen[:, :-1]) ** 2)
                    loss = loss_mse + loss_q_smooth
                else:
                    loss = loss_mse + loss_smooth

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_train_losses.append(loss.item())

            avg_train_loss = np.mean(epoch_train_losses)
            avg_val_loss = validate_epoch(model, loader_val, mode_name)

            collector.record_epoch(
                epoch=global_epoch, phase=phase_name, train_loss=avg_train_loss,
                val_loss=avg_val_loss, lr_phy=lr_phy * phase_info["phy_scale"],
                lr_mamba=lr_mamba * phase_info["mamba_scale"]
            )

            status = ""
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = global_epoch
                status = "✓ Best"

            if (epoch + 1) % 5 == 0 or status:
                print_log_row(global_epoch, phase_name, lr_phy * phase_info["phy_scale"],
                              lr_mamba * phase_info["mamba_scale"], avg_train_loss, avg_val_loss, status)

            if early_stopper(avg_val_loss, global_epoch):
                print(f"   ⏹️ Early stopping at epoch {global_epoch}")
                break

    return best_model_state, best_epoch, best_val_loss


def train_single_phase(model: nn.Module, mode_name: str, loader_tr: DataLoader,
                       loader_val: DataLoader, collector: TrainingDataCollector,
                       use_same_lr_structure: bool = False) -> Tuple[Dict, int, float]:
    """
    单阶段训练

    use_same_lr_structure: 如果True，使用和Progressive相同的两组参数LR结构，且不用scheduler
    """
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0

    if use_same_lr_structure and model.use_physics and model.use_data:
        param_estimator_params = list(model.param_estimator.parameters())
        mamba_params = list(model.mamba_layer.parameters())

        lr_phy = 5e-5
        lr_mamba = 1e-4

        print(f"\n🔷 [No_Progressive] Same LR as Phase3, NO scheduler")
        print(f"   lr_phy={lr_phy}, lr_mamba={lr_mamba}")
        print(f"{'Epoch':^8} | {'LR_phy':^10} | {'LR_mamba':^10} | {'Train':^10} | {'Val':^10} | {'Status':^12}")
        print("-" * 75)

        optimizer = optim.AdamW([
            {'params': param_estimator_params, 'lr': lr_phy},
            {'params': mamba_params, 'lr': lr_mamba}
        ])
        use_scheduler = False
    else:
        all_params = list(model.parameters())
        lr = 1e-3

        print(f"\n🔷 [Single Phase] Starting (max {cfg.epochs_single} epochs)")
        print(f"{'Epoch':^8} | {'LR':^10} | {'Train':^10} | {'Val':^10} | {'Status':^12}")
        print("-" * 60)

        optimizer = optim.AdamW(all_params, lr=lr)
        use_scheduler = True
        lr_phy = lr
        lr_mamba = lr

    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    early_stopper = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

    for epoch in range(cfg.epochs_single):
        model.train()
        epoch_train_losses = []

        for batch in loader_tr:
            optimizer.zero_grad()
            loss_mse, loss_smooth, _, _, _, _ = compute_losses(model, batch, mode_name)
            loss = loss_mse + loss_smooth

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = validate_epoch(model, loader_val, mode_name)

        if use_scheduler:
            scheduler.step(avg_val_loss)

        current_lr_phy = optimizer.param_groups[0]['lr']
        current_lr_mamba = optimizer.param_groups[-1]['lr'] if len(optimizer.param_groups) > 1 else current_lr_phy

        collector.record_epoch(
            epoch=epoch + 1, phase="Single", train_loss=avg_train_loss,
            val_loss=avg_val_loss, lr_phy=current_lr_phy, lr_mamba=current_lr_mamba
        )

        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            status = "✓ Best"

        if (epoch + 1) % 10 == 0 or status:
            if use_same_lr_structure:
                print(f"{epoch + 1:^8} | {current_lr_phy:.2e} | {current_lr_mamba:.2e} | "
                      f"{avg_train_loss:^10.6f} | {avg_val_loss:^10.6f} | {status:^12}")
            else:
                print(f"{epoch + 1:^8} | {current_lr_phy:.2e} | {avg_train_loss:^10.6f} | "
                      f"{avg_val_loss:^10.6f} | {status:^12}")

        if early_stopper(avg_val_loss, epoch + 1):
            print(f"   ⏹️ Early stopping at epoch {epoch + 1}")
            break

    return best_model_state, best_epoch, best_val_loss


def train_variant(mode_name: str, loader_tr: DataLoader, loader_val: DataLoader) -> Dict[str, Any]:
    """训练一个变体"""
    print(f"\n{'=' * 60}")
    print(f"🏁 Training Variant: {mode_name}")
    print(f"{'=' * 60}")

    set_seed(42)
    model = PM_KMNet_Ablation(ablation_mode=mode_name).to(DEVICE)

    collector = TrainingDataCollector(mode_name)
    mode_dir = os.path.join(cfg.save_dir, "collected_data", mode_name)
    os.makedirs(mode_dir, exist_ok=True)

    if mode_name == "No_Progressive":
        best_model_state, best_epoch, best_val_loss = train_single_phase(
            model, mode_name, loader_tr, loader_val, collector, use_same_lr_structure=True
        )
    elif mode_name in ["Physics_Only", "Data_Only"]:
        best_model_state, best_epoch, best_val_loss = train_single_phase(
            model, mode_name, loader_tr, loader_val, collector, use_same_lr_structure=False
        )
    else:
        best_model_state, best_epoch, best_val_loss = train_progressive(
            model, mode_name, loader_tr, loader_val, collector
        )

    model_path = os.path.join(mode_dir, "best_model.pth")
    if best_model_state is not None:
        torch.save(best_model_state, model_path)
        print(f"\n   💾 Best model saved (epoch {best_epoch}, val_loss: {best_val_loss:.6f})")
    else:
        torch.save(model.state_dict(), model_path)
        print(f"\n   💾 Model saved (current state)")

    collector.save(cfg.save_dir)

    return {
        "best_model_path": model_path,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_history": collector.training_history
    }


def evaluate_on_test_set(mode_name: str, model_path: str, loader_test: DataLoader) -> Dict[str, Any]:
    """在测试集上评估模型"""
    print(f"\n{'=' * 60}")
    print(f"📊 Evaluating: {mode_name}")
    print(f"{'=' * 60}")

    model = PM_KMNet_Ablation(ablation_mode=mode_name).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    test_data = {
        "step_idx": [], "rpm": [], "load": [], "press": [], "vib": [], "t_in": [],
        "ground_truth": [], "prediction": [],
        "alpha": [], "beta": [], "eta_gen": [], "k_cool": [], "tau": [],
        "q_gen": [], "q_out": [], "p_resid": [], "error": []
    }

    step_counter = 0

    with torch.no_grad():
        for batch in loader_test:
            b_ctx, b_rpm, b_load, b_press, b_vib, b_tin, b_tinit, b_y = [t.to(DEVICE) for t in batch]
            pred, q_gen, q_out, p_resid = model(b_ctx, b_rpm, b_load, b_press, b_vib, b_tin, b_tinit)
            params = model.get_all_parameters_for_batch(b_ctx, b_rpm, b_load)

            batch_size = pred.shape[0]
            for i in range(batch_size):
                test_data["step_idx"].append(step_counter)
                test_data["rpm"].append(b_rpm[i, -1, 0].item() * cfg.norm_rpm)
                test_data["load"].append(b_load[i, -1, 0].item() * cfg.norm_load)
                test_data["press"].append(b_press[i, -1, 0].item() * cfg.norm_press)
                test_data["vib"].append(b_vib[i, -1, 0].item() * cfg.norm_vib)
                test_data["t_in"].append(b_tin[i, -1, 0].item() * cfg.temp_scale + cfg.temp_base)

                gt = b_y[i, -1, 0].item() * cfg.temp_scale + cfg.temp_base
                pr = pred[i, -1, 0].item() * cfg.temp_scale + cfg.temp_base

                test_data["ground_truth"].append(gt)
                test_data["prediction"].append(pr)
                test_data["error"].append(pr - gt)

                test_data["alpha"].append(params['alpha'][i, -1, 0].item())
                test_data["beta"].append(params['beta'][i, -1, 0].item())
                test_data["eta_gen"].append(params['eta_gen'][i, -1, 0].item())
                test_data["k_cool"].append(params['k_cool'][i, -1, 0].item())
                test_data["tau"].append(params['tau'][i, -1, 0].item())

                test_data["q_gen"].append(q_gen[i, -1, 0].item())
                test_data["q_out"].append(q_out[i, -1, 0].item())
                test_data["p_resid"].append(p_resid[i, -1, 0].item())

                step_counter += 1

    df_test = pd.DataFrame(test_data)

    trues = df_test["ground_truth"].values
    preds = df_test["prediction"].values

    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    mape = np.mean(np.abs((trues - preds) / (np.abs(trues) + 1e-5))) * 100
    tic = compute_tic(preds, trues)
    max_err = max_error(trues, preds)
    exp_var = explained_variance_score(trues, preds)

    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "MAPE": float(mape),
        "TIC": float(tic),
        "Max_Error": float(max_err),
        "Explained_Variance": float(exp_var)
    }

    mode_dir = os.path.join(cfg.save_dir, "collected_data", mode_name)
    os.makedirs(mode_dir, exist_ok=True)

    df_test.to_csv(os.path.join(mode_dir, "test_predictions.csv"), index=False)
    with open(os.path.join(mode_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"   ✅ RMSE={rmse:.4f} | MAE={mae:.4f} | MAPE={mape:.2f}% | R²={r2:.4f} | TIC={tic:.5f}")

    return {"test_df": df_test, "metrics": metrics}


# ==========================================
# 11. 数据持久化
# ==========================================
def save_all_results(all_results: Dict[str, Any], save_dir: str) -> str:
    serializable_results = {}
    for mode, result in all_results.items():
        serializable_results[mode] = {
            "test_df": result["test_df"].to_dict(),
            "metrics": result["metrics"],
            "training_history": result.get("training_history", {}),
            "best_epoch": result.get("best_epoch", 0)
        }

    pkl_path = os.path.join(save_dir, "all_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(serializable_results, f)
    print(f"\n💾 All results saved to: {pkl_path}")
    return pkl_path


def load_all_results(save_dir: str) -> Optional[Dict[str, Any]]:
    pkl_path = os.path.join(save_dir, "all_results.pkl")
    if not os.path.exists(pkl_path):
        print(f"❌ Results file not found: {pkl_path}")
        return None

    with open(pkl_path, "rb") as f:
        serializable_results = pickle.load(f)

    all_results = {}
    for mode, result in serializable_results.items():
        all_results[mode] = {
            "test_df": pd.DataFrame(result["test_df"]),
            "metrics": result["metrics"],
            "training_history": result.get("training_history", {}),
            "best_epoch": result.get("best_epoch", 0)
        }

    print(f"✅ Loaded results from: {pkl_path}")
    return all_results


# ==========================================
# 12. 可视化模块
# ==========================================
def generate_all_figures(all_results: Dict[str, Any], save_dir: Optional[str] = None):
    if save_dir is None:
        save_dir = cfg.save_dir

    print("\n🎨 Generating Publication Figures...")

    c_prop = '#0052cc'
    c_bad = '#d62728'
    c_gray = '#7f7f7f'
    c_phy = '#2ca02c'
    c_data = '#ff7f0e'

    # Figure 1: 预测轨迹
    try:
        print("   👉 Fig 1: Prediction Trajectory...")
        fig, ax = plt.subplots(figsize=(14, 6))
        df_prop = all_results["Proposed"]["test_df"]

        ax.plot(df_prop["ground_truth"].values, 'k-', lw=1.5, alpha=0.7, label='Ground Truth')
        ax.plot(df_prop["prediction"].values, c=c_prop, lw=1.5, ls='--',
                label=f'PM-KMNet (RMSE={all_results["Proposed"]["metrics"]["RMSE"]:.3f}°C)')

        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Temperature (°C)", fontsize=12)
        ax.set_title("Temperature Prediction Trajectory (Voyage 2 - Test Set)", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Fig_1_Prediction_Trajectory.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"   ⚠️ Failed Fig 1: {e}")

    # Figure 2: 消融实验对比
    try:
        print("   👉 Fig 2: Ablation Comparison...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        window = 500
        start = 1000

        colors = {'Proposed': c_prop, 'No_KAN': c_bad, 'No_Mamba': c_gray,
                  'Physics_Only': c_phy, 'Data_Only': c_data, 'No_Progressive': '#9467bd'}

        for mode in ["Proposed", "No_KAN", "No_Mamba"]:
            if mode in all_results:
                df = all_results[mode]["test_df"]
                if len(df) > start + window:
                    ax1.plot(range(window), df["prediction"].values[start:start + window],
                             c=colors.get(mode, 'gray'), lw=1.5 if mode != "Proposed" else 2.5,
                             alpha=0.7 if mode != "Proposed" else 1.0,
                             label=f'{mode} (RMSE={all_results[mode]["metrics"]["RMSE"]:.3f})')

        if "Proposed" in all_results:
            df = all_results["Proposed"]["test_df"]
            if len(df) > start + window:
                ax1.scatter(range(window), df["ground_truth"].values[start:start + window],
                            c='k', s=3, alpha=0.3, label='Ground Truth', zorder=0)

        ax1.set_xlabel("Time Step", fontsize=11)
        ax1.set_ylabel("Temperature (°C)", fontsize=11)
        ax1.set_title("(a) Prediction Trajectory", fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        modes_order = ["Proposed", "No_KAN", "No_Mamba", "Physics_Only", "Data_Only", "No_Progressive"]
        rmse_vals = []
        mae_vals = []
        mode_labels = []

        for mode in modes_order:
            if mode in all_results:
                rmse_vals.append(all_results[mode]["metrics"]["RMSE"])
                mae_vals.append(all_results[mode]["metrics"]["MAE"])
                mode_labels.append(mode)

        x = np.arange(len(mode_labels))
        width = 0.35

        bars1 = ax2.bar(x - width / 2, rmse_vals, width, label='RMSE', color=c_prop, alpha=0.8)
        bars2 = ax2.bar(x + width / 2, mae_vals, width, label='MAE', color=c_data, alpha=0.8)

        ax2.set_xlabel("Variant", fontsize=11)
        ax2.set_ylabel("Error (°C)", fontsize=11)
        ax2.set_title("(b) Ablation Study", fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(mode_labels, rotation=30, ha='right', fontsize=9)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        for bar in bars1:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Fig_2_Ablation_Comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"   ⚠️ Failed Fig 2: {e}")

    # Figure 3: KAN vs MLP
    try:
        if "Proposed" in all_results and "No_KAN" in all_results:
            print("   👉 Fig 3: KAN vs MLP...")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            df_kan = all_results["Proposed"]["test_df"]
            df_mlp = all_results["No_KAN"]["test_df"]

            ax1 = axes[0]
            ax1.scatter(df_mlp["rpm"], df_mlp["tau"], c=c_bad, alpha=0.1, s=10, label="MLP")
            ax1.scatter(df_kan["rpm"], df_kan["tau"], c=c_prop, alpha=0.1, s=10, label="KAN")

            for df, label, color in [(df_kan, 'KAN', c_prop), (df_mlp, 'MLP', c_bad)]:
                idx = np.argsort(df["rpm"].values)
                rpm_sorted = df["rpm"].values[idx]
                tau_sorted = df["tau"].values[idx]
                tau_smooth = pd.Series(tau_sorted).rolling(window=100, center=True).mean()
                ax1.plot(rpm_sorted, tau_smooth, c=color, lw=3, label=f'{label} Trend')

            ax1.set_xlabel("Engine Speed (RPM)", fontsize=11)
            ax1.set_ylabel("Thermal Inertia τ", fontsize=11)
            ax1.set_title("(a) τ vs RPM", fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = axes[1]
            params = ['alpha', 'beta', 'tau', 'eta_gen', 'k_cool']
            kan_stds = [df_kan[p].std() for p in params]
            mlp_stds = [df_mlp[p].std() for p in params]

            x = np.arange(len(params))
            width = 0.35

            ax2.bar(x - width / 2, kan_stds, width, label='KAN', color=c_prop, alpha=0.8)
            ax2.bar(x + width / 2, mlp_stds, width, label='MLP', color=c_bad, alpha=0.8)

            ax2.set_xlabel("Parameter", fontsize=11)
            ax2.set_ylabel("Std Dev", fontsize=11)
            ax2.set_title("(b) Parameter Stability", fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(params)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "Fig_3_KAN_vs_MLP.png"), dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"   ⚠️ Failed Fig 3: {e}")

    # Figure 4: Mamba vs GRU
    try:
        if "Proposed" in all_results and "No_Mamba" in all_results:
            print("   👉 Fig 4: Mamba vs GRU...")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            df_mamba = all_results["Proposed"]["test_df"]
            df_gru = all_results["No_Mamba"]["test_df"]

            ax1 = axes[0]
            err_mamba = df_mamba["error"].values
            err_gru = df_gru["error"].values

            if len(err_mamba) > 10:
                kde_m = gaussian_kde(err_mamba)
                kde_g = gaussian_kde(err_gru)
                x_grid = np.linspace(-4, 4, 500)

                ax1.fill_between(x_grid, kde_g(x_grid), color=c_bad, alpha=0.2)
                ax1.plot(x_grid, kde_g(x_grid), c=c_bad, lw=2, label='GRU')
                ax1.fill_between(x_grid, kde_m(x_grid), color=c_prop, alpha=0.3)
                ax1.plot(x_grid, kde_m(x_grid), c=c_prop, lw=2.5, label='Mamba')

                ax1.set_yscale('log')
                ax1.set_ylim(bottom=1e-3)
                ax1.set_xlabel("Error (°C)", fontsize=11)
                ax1.set_ylabel("Density (log)", fontsize=11)
                ax1.set_title("(a) Error Distribution", fontsize=12, fontweight='bold')
                ax1.legend()
                ax1.grid(True, which='both', alpha=0.3)

            ax2 = axes[1]
            for df, label, color in [(df_mamba, 'Mamba', c_prop), (df_gru, 'GRU', c_bad)]:
                err = df["error"].values
                if len(err) > 256:
                    freqs, psd = welch(err, fs=1.0, nperseg=min(256, len(err) // 4))
                    ax2.semilogy(freqs, psd, lw=2, label=label, color=color)

            ax2.set_xlabel("Frequency", fontsize=11)
            ax2.set_ylabel("PSD", fontsize=11)
            ax2.set_title("(b) Error PSD", fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, which='both', alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "Fig_4_Mamba_vs_GRU.png"), dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"   ⚠️ Failed Fig 4: {e}")

    # Figure 5: Physics vs Data
    try:
        print("   👉 Fig 5: Physics vs Data...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        available = {k: v for k, v in all_results.items() if k in ["Proposed", "Physics_Only", "Data_Only"]}

        ax1 = axes[0, 0]
        window = 300
        start = 500

        if "Proposed" in available:
            df = available["Proposed"]["test_df"]
            if len(df) > start + window:
                ax1.scatter(range(window), df["ground_truth"].values[start:start + window],
                            c='k', s=5, alpha=0.3, label='Ground Truth')

        for mode, color in [("Physics_Only", c_phy), ("Data_Only", c_data), ("Proposed", c_prop)]:
            if mode in available:
                df = available[mode]["test_df"]
                if len(df) > start + window:
                    lw = 2.5 if mode == "Proposed" else 1.5
                    ax1.plot(range(window), df["prediction"].values[start:start + window],
                             c=color, lw=lw, label=mode)

        ax1.set_xlabel("Time Step", fontsize=11)
        ax1.set_ylabel("Temperature (°C)", fontsize=11)
        ax1.set_title("(a) Prediction Comparison", fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        errors = []
        labels = []
        for mode in ["Physics_Only", "Data_Only", "Proposed"]:
            if mode in available:
                errors.append(available[mode]["test_df"]["error"].values)
                labels.append(mode)

        if errors:
            bp = ax2.boxplot(errors, labels=labels, patch_artist=True)
            colors_box = [c_phy, c_data, c_prop][:len(errors)]
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

        ax2.axhline(0, color='k', lw=0.5, ls='--')
        ax2.set_ylabel("Error (°C)", fontsize=11)
        ax2.set_title("(b) Error Distribution", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        if "Proposed" in all_results:
            df = all_results["Proposed"]["test_df"]
            plot_len = min(1000, len(df))
            ax3.plot(df["q_gen"].values[:plot_len], color=c_phy, alpha=0.8, label=r'$Q_{gen}$')
            ax3.plot(df["q_out"].values[:plot_len], color='#1f77b4', alpha=0.8, label=r'$Q_{out}$')
            ax3.plot(df["p_resid"].values[:plot_len], color=c_data, alpha=0.6, lw=1, label=r'$P_{res}$')
            ax3.axhline(0, color='k', lw=0.5)

        ax3.set_xlabel("Time Step", fontsize=11)
        ax3.set_ylabel("Power", fontsize=11)
        ax3.set_title("(c) Energy Flow", fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        metrics_to_plot = ["RMSE", "MAE", "TIC"]

        for mode, color in [("Physics_Only", c_phy), ("Data_Only", c_data), ("Proposed", c_prop)]:
            if mode in available:
                values = [available[mode]["metrics"][m] for m in metrics_to_plot]
                ax4.bar([f"{mode}\n{m}" for m in metrics_to_plot], values, color=color, alpha=0.7)

        ax4.set_ylabel("Value", fontsize=11)
        ax4.set_title("(d) Metrics", fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Fig_5_Physics_vs_Data.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"   ⚠️ Failed Fig 5: {e}")

    # Figure 6: Progressive Training
    try:
        if "Proposed" in all_results and "No_Progressive" in all_results:
            print("   👉 Fig 6: Progressive Training...")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax1 = axes[0]
            for mode, color, label in [("Proposed", c_prop, "Progressive"), ("No_Progressive", c_bad, "Single")]:
                if mode in all_results:
                    hist = all_results[mode].get("training_history", {})
                    if hist and "epoch" in hist and len(hist["epoch"]) > 0:
                        ax1.plot(hist["epoch"], hist["val_loss"], c=color, lw=2, label=f'{label} Val')
                        ax1.plot(hist["epoch"], hist["train_loss"], c=color, lw=1, ls='--', alpha=0.5)

            ax1.set_yscale('log')
            ax1.set_xlabel("Epoch", fontsize=11)
            ax1.set_ylabel("Loss", fontsize=11)
            ax1.set_title("(a) Training Convergence", fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, which='both', alpha=0.3)

            ax2 = axes[1]
            df_prog = all_results["Proposed"]["test_df"]
            df_single = all_results["No_Progressive"]["test_df"]

            window = min(500, len(df_prog))
            ax2.plot(range(window), np.abs(df_single["error"].values[:window]),
                     c=c_bad, lw=1.5, alpha=0.7, label='Single')
            ax2.plot(range(window), np.abs(df_prog["error"].values[:window]),
                     c=c_prop, lw=2, label='Progressive')

            ax2.set_xlabel("Time Step", fontsize=11)
            ax2.set_ylabel("|Error| (°C)", fontsize=11)
            ax2.set_title("(b) Test Error", fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "Fig_6_Progressive_Training.png"), dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"   ⚠️ Failed Fig 6: {e}")

    # Figure 7: Parameters
    try:
        print("   👉 Fig 7: Parameters...")
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        df = all_results["Proposed"]["test_df"]
        t = df["step_idx"].values

        params = ['alpha', 'beta', 'tau', 'eta_gen', 'k_cool']
        titles = ['α', 'β', 'τ', 'η_gen', 'k_cool']

        for idx, (param, title) in enumerate(zip(params, titles)):
            ax = axes.flatten()[idx]
            ax.plot(t, df[param].values, c=c_prop, lw=1, alpha=0.7)
            ma = pd.Series(df[param].values).rolling(window=50, center=True).mean()
            ax.plot(t, ma, c='black', lw=2)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel("Time Step")
            ax.grid(True, alpha=0.3)

        axes.flatten()[5].plot(t, df["rpm"].values, c=c_gray, lw=1)
        axes.flatten()[5].set_title("RPM", fontsize=11, fontweight='bold')
        axes.flatten()[5].set_xlabel("Time Step")
        axes.flatten()[5].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Fig_7_Parameters.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"   ⚠️ Failed Fig 7: {e}")

    # Figure 8: Training All
    try:
        print("   👉 Fig 8: Training All...")
        fig, ax = plt.subplots(figsize=(12, 6))

        colors_train = {
            'Proposed': c_prop, 'No_KAN': c_bad, 'No_Mamba': c_gray,
            'Physics_Only': c_phy, 'Data_Only': c_data, 'No_Progressive': '#9467bd'
        }

        for mode, res in all_results.items():
            if "training_history" in res and res["training_history"]:
                h = res["training_history"]
                if "epoch" in h and len(h["epoch"]) > 0:
                    lw = 2.5 if mode == "Proposed" else 1.5
                    ax.plot(h["epoch"], h["val_loss"], lw=lw, label=mode, color=colors_train.get(mode, 'gray'))

        ax.set_yscale('log')
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Val Loss", fontsize=11)
        ax.set_title("Training Convergence", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Fig_8_Training_All.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"   ⚠️ Failed Fig 8: {e}")

    # Figure 9: Error Histogram
    try:
        print("   👉 Fig 9: Error Histogram...")
        fig, ax = plt.subplots(figsize=(10, 6))
        df = all_results["Proposed"]["test_df"]
        residuals = df["error"].values

        ax.hist(residuals, bins=50, color='gray', edgecolor='black', alpha=0.7, density=True)
        ax.axvline(0, color='r', linestyle='--', lw=2)
        ax.set_title(f"Error Distribution (Mean={np.mean(residuals):.3f}, Std={np.std(residuals):.3f})",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Error (°C)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Fig_9_Error_Histogram.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"   ⚠️ Failed Fig 9: {e}")

    print("   ✅ Figure generation completed!")


# ==========================================
# 13. LaTeX Table
# ==========================================
def export_latex_table(all_results: Dict[str, Any], save_dir: Optional[str] = None):
    if save_dir is None:
        save_dir = cfg.save_dir

    print("\n📄 Generating LaTeX Table...")

    modes = ["Proposed", "No_KAN", "No_Mamba", "Physics_Only", "Data_Only", "No_Progressive"]
    mode_names = {
        "Proposed": r"PM-KMNet (Proposed)",
        "No_KAN": r"w/o KAN (MLP)",
        "No_Mamba": r"w/o Mamba (GRU)",
        "Physics_Only": r"Physics-only",
        "Data_Only": r"Data-only",
        "No_Progressive": r"w/o Progressive"
    }

    metrics_data = {}
    for mode in modes:
        if mode in all_results:
            metrics_data[mode] = all_results[mode]["metrics"]

    if not metrics_data:
        print("   ⚠️ No results")
        return

    metric_keys = ["RMSE", "MAE", "MAPE", "R2", "TIC"]
    best_values = {}
    for key in metric_keys:
        values = [m[key] for m in metrics_data.values()]
        if key == "R2":
            best_values[key] = max(values)
        else:
            best_values[key] = min(values)

    def format_value(value, key, is_best):
        if key == "MAPE":
            formatted = f"{value:.2f}"
        elif key == "TIC":
            formatted = f"{value:.5f}"
        else:
            formatted = f"{value:.4f}"
        if is_best:
            return r"\textbf{" + formatted + "}"
        return formatted

    table_str = r"""\begin{table}[ht]
\centering
\caption{Ablation study results. Best in \textbf{bold}.}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
\textbf{Variant} & \textbf{RMSE} & \textbf{MAE} & \textbf{MAPE(\%)} & \textbf{R$^2$} & \textbf{TIC} \\
\midrule
"""

    for mode in modes:
        if mode not in metrics_data:
            continue
        m = metrics_data[mode]
        name = mode_names.get(mode, mode)

        rmse_str = format_value(m["RMSE"], "RMSE", abs(m["RMSE"] - best_values["RMSE"]) < 1e-6)
        mae_str = format_value(m["MAE"], "MAE", abs(m["MAE"] - best_values["MAE"]) < 1e-6)
        mape_str = format_value(m["MAPE"], "MAPE", abs(m["MAPE"] - best_values["MAPE"]) < 1e-6)
        r2_str = format_value(m["R2"], "R2", abs(m["R2"] - best_values["R2"]) < 1e-6)
        tic_str = format_value(m["TIC"], "TIC", abs(m["TIC"] - best_values["TIC"]) < 1e-8)

        row = f"{name} & {rmse_str} & {mae_str} & {mape_str} & {r2_str} & {tic_str} \\\\\n"
        table_str += row

    table_str += r"""\bottomrule
\end{tabular}
\end{table}
"""

    print(table_str)

    table_path = os.path.join(save_dir, "Latex_Table.txt")
    with open(table_path, "w") as f:
        f.write(table_str)
    print(f"   💾 Saved: {table_path}")


# ==========================================
# 14. 评估报告
# ==========================================
def generate_evaluation_report(all_results: Dict[str, Any], save_dir: Optional[str] = None):
    if save_dir is None:
        save_dir = cfg.save_dir

    report_path = os.path.join(save_dir, "Evaluation_Report.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("PM-KMNet Ablation Study - Evaluation Report (V3.2)\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Key Features:\n")
        f.write("  1. Strict scaler: fit on TRAIN split only\n")
        f.write("  2. Data_Only: ΔT recurrence (fair)\n")
        f.write("  3. No_Progressive: same LR, no scheduler\n")
        f.write("  4. Weakened baselines (single-layer MLP, small GRU)\n")
        f.write("  5. Auto-bold best in LaTeX\n\n")

        f.write("-" * 70 + "\n")
        f.write("RESULTS\n")
        f.write("-" * 70 + "\n\n")

        header = f"{'Variant':<20} | {'RMSE':>8} | {'MAE':>8} | {'MAPE%':>8} | {'R²':>8} | {'TIC':>10}\n"
        f.write(header)
        f.write("-" * 70 + "\n")

        for mode in ["Proposed", "No_KAN", "No_Mamba", "Physics_Only", "Data_Only", "No_Progressive"]:
            if mode not in all_results:
                continue
            m = all_results[mode]["metrics"]
            row = f"{mode:<20} | {m['RMSE']:>8.4f} | {m['MAE']:>8.4f} | {m['MAPE']:>7.2f}% | {m['R2']:>8.4f} | {m['TIC']:>10.6f}\n"
            f.write(row)

        f.write("\n" + "=" * 70 + "\n")

    print(f"📝 Report saved: {report_path}")


# ==========================================
# 15. 可视化入口
# ==========================================
def visualize_from_saved_data(save_dir: Optional[str] = None):
    if save_dir is None:
        save_dir = cfg.save_dir

    print("\n🔄 Loading saved results...")
    all_results = load_all_results(save_dir)

    if all_results is None:
        print("❌ No results found")
        return

    generate_all_figures(all_results, save_dir)
    export_latex_table(all_results, save_dir)
    generate_evaluation_report(all_results, save_dir)

    print("\n✅ Done!")


# ==========================================
# 16. 主函数
# ==========================================
def main():
    print(f"\n{'=' * 70}")
    print(f"🚀 PM-KMNet Ablation Study V3.2 (Weakened Baselines)")
    print(f"{'=' * 70}")
    print(f"   Device: {DEVICE}")
    print(f"   Mamba: {USE_MAMBA}")
    print(f"   Save: {cfg.save_dir}")
    print(f"{'=' * 70}\n")

    if not os.path.exists(cfg.path_train):
        print(f"❌ Not found: {cfg.path_train}")
        return
    if not os.path.exists(cfg.path_test):
        print(f"❌ Not found: {cfg.path_test}")
        return

    df_voyage1_raw = pd.read_csv(cfg.path_train)
    df_voyage2_raw = pd.read_csv(cfg.path_test)

    print(f"📊 Loaded:")
    print(f"   Voyage1: {len(df_voyage1_raw)}")
    print(f"   Voyage2: {len(df_voyage2_raw)}")

    # 严格分离
    split_idx = int(len(df_voyage1_raw) * cfg.train_ratio)
    df_train_raw = df_voyage1_raw.iloc[:split_idx].copy()
    df_val_raw = df_voyage1_raw.iloc[split_idx:].copy()

    print(f"\n📊 Split:")
    print(f"   Train: {len(df_train_raw)}")
    print(f"   Val: {len(df_val_raw)}")

    proc = DataProcessor()
    proc.fit(df_train_raw)

    scaler_path = os.path.join(cfg.save_dir, "scalers.json")
    proc.save_scalers(scaler_path)

    print("\n🔧 Transforming...")
    data_train = proc.transform(df_train_raw)
    data_val = proc.transform(df_val_raw)
    data_voyage2 = proc.transform(df_voyage2_raw)

    loader_train = DataProcessor.create_loader(data_train, cfg.batch_size, shuffle=True, stride=5)
    loader_val = DataProcessor.create_loader(data_val, cfg.batch_size, shuffle=False, stride=1)
    loader_test = DataProcessor.create_loader(data_voyage2, cfg.batch_size, shuffle=False, stride=1)

    modes = ["Proposed", "No_KAN", "No_Mamba", "Physics_Only", "Data_Only", "No_Progressive"]
    all_results = {}

    for mode in modes:
        train_result = train_variant(mode, loader_train, loader_val)
        eval_result = evaluate_on_test_set(mode, train_result["best_model_path"], loader_test)

        all_results[mode] = {
            **eval_result,
            "training_history": train_result["training_history"],
            "best_epoch": train_result["best_epoch"]
        }

    print("\n" + "=" * 70)
    print("💾 SAVING")
    print("=" * 70)
    save_all_results(all_results, cfg.save_dir)

    print("\n" + "=" * 70)
    print("🎨 VISUALIZATION")
    print("=" * 70)

    try:
        generate_all_figures(all_results, cfg.save_dir)
        export_latex_table(all_results, cfg.save_dir)
        generate_evaluation_report(all_results, cfg.save_dir)
    except Exception as e:
        print(f"\n⚠️ Viz failed: {e}")
        print("   Run visualize_from_saved_data() to retry")

    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"{'Mode':<20} | {'RMSE':>8} | {'MAE':>8} | {'MAPE':>8} | {'R²':>8} | {'TIC':>10}")
    print("-" * 75)

    for mode in modes:
        if mode in all_results:
            m = all_results[mode]["metrics"]
            print(f"{mode:<20} | {m['RMSE']:>8.4f} | {m['MAE']:>8.4f} | {m['MAPE']:>7.2f}% | {m['R2']:>8.4f} | {m['TIC']:>10.6f}")

    print("\n" + "=" * 70)
    print("✨ COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
