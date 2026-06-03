from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


if nn is not None:

    class PairwiseRiskRankingLoss(nn.Module):
        def __init__(self, margin: float = 0.0, scale: float = 10.0) -> None:
            super().__init__()
            self.margin = margin
            self.scale = scale

        def forward(self, risk_scores, true_severity):
            n = risk_scores.shape[0]
            if n < 2:
                return torch.tensor(0.0, device=risk_scores.device)
            sev_diff = true_severity.unsqueeze(1) - true_severity.unsqueeze(0)
            risk_diff = risk_scores.unsqueeze(1) - risk_scores.unsqueeze(0)
            positive_pairs = sev_diff > 0
            if positive_pairs.sum() == 0:
                return torch.tensor(0.0, device=risk_scores.device)
            pair_loss = torch.log1p(torch.exp(self.scale * (self.margin - risk_diff)))
            return pair_loss[positive_pairs].mean()


    class JointRiskTransformer(nn.Module):
        def __init__(self, input_dim: int, config: dict):
            super().__init__()
            d_model = config.get("d_model", 64)
            nhead = config.get("nhead", 4)
            n_layers = config.get("n_layers", 2)
            dim_ff = config.get("dim_feedforward", 128)
            dropout = config.get("dropout", 0.1)
            max_seq_len = config.get("max_seq_len", 16)

            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.occ_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            self.sev_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.ReLU(),
            )
            self.occ_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
            self.sev_loss_fn = nn.MSELoss(reduction="none")
            self.rank_loss_fn = PairwiseRiskRankingLoss(
                margin=config.get("rank_margin", 0.0),
                scale=config.get("rank_scale", 10.0),
            )
            self.lambda_sev = config.get("lambda_sev", 0.5)
            self.lambda_rank = config.get("lambda_rank", 0.1)

        def forward(self, x, seq_lengths=None):
            batch_size, seq_len, _ = x.shape
            h = self.input_proj(x)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            h = h + self.pos_embedding(positions)
            padding_mask = None
            if seq_lengths is not None:
                seq_lengths = seq_lengths.to(x.device).long().clamp(min=1, max=seq_len)
                pad_starts = (seq_len - seq_lengths).unsqueeze(1)
                padding_mask = positions < pad_starts
            h = self.encoder(h, src_key_padding_mask=padding_mask)
            h_pooled = h[:, -1, :]
            occ_logit = self.occ_head(h_pooled).squeeze(-1)
            sev_pred = self.sev_head(h_pooled).squeeze(-1)
            return occ_logit, sev_pred

        def compute_loss(
            self,
            occ_logit,
            sev_pred,
            y_occ,
            y_sev,
            sev_available,
            propensity_scores=None,
        ):
            occ_loss = self.occ_loss_fn(occ_logit, y_occ.float())
            sev_mask = sev_available.bool()
            if sev_mask.sum() > 0:
                sev_errors = self.sev_loss_fn(sev_pred[sev_mask], y_sev[sev_mask])
                if propensity_scores is not None:
                    weights = 1.0 / propensity_scores[sev_mask].clamp(min=0.05)
                    weights = weights * weights.shape[0] / weights.sum()
                    sev_loss = (sev_errors * weights).mean()
                else:
                    sev_loss = sev_errors.mean()
            else:
                sev_loss = torch.tensor(0.0, device=occ_logit.device)

            if sev_mask.sum() > 1:
                p_fire = torch.sigmoid(occ_logit[sev_mask])
                risk_scores = p_fire * sev_pred[sev_mask]
                rank_loss = self.rank_loss_fn(risk_scores, y_sev[sev_mask])
            else:
                rank_loss = torch.tensor(0.0, device=occ_logit.device)

            total = occ_loss + self.lambda_sev * sev_loss + self.lambda_rank * rank_loss
            return total, {
                "occ_loss": float(occ_loss.item()),
                "sev_loss": float(sev_loss.item()),
                "rank_loss": float(rank_loss.item()),
                "total_loss": float(total.item()),
            }


    class IndexedTensorDataset(TensorDataset):
        def __getitem__(self, index):
            batch = super().__getitem__(index)
            return (*batch, torch.tensor(index, dtype=torch.long))


    class JointRiskTrainer:
        def __init__(self, model: JointRiskTransformer, config: dict):
            self.model = model
            self.config = config
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.get("lr", 3e-4),
                weight_decay=config.get("weight_decay", 1e-5),
            )
            self.best_val_metric = -float("inf")
            self.patience_counter = 0

        def train_epoch(self, train_loader, propensity_scores=None):
            self.model.train()
            epoch_losses = []
            for batch in train_loader:
                if len(batch) == 6:
                    x, seq_lengths, y_occ, y_sev, sev_avail, indices = batch
                else:
                    x, seq_lengths, y_occ, y_sev, sev_avail = batch
                    indices = None
                p_scores = None
                if propensity_scores is not None and indices is not None:
                    p_scores = propensity_scores[indices]

                self.optimizer.zero_grad()
                occ_logit, sev_pred = self.model(x, seq_lengths=seq_lengths)
                loss, components = self.model.compute_loss(
                    occ_logit,
                    sev_pred,
                    y_occ,
                    y_sev,
                    sev_avail,
                    p_scores,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_losses.append(components)
            return epoch_losses

        def evaluate(self, val_loader):
            self.model.eval()
            all_risk = []
            all_sev = []
            with torch.no_grad():
                for batch in val_loader:
                    x, seq_lengths, y_occ, y_sev, sev_avail = batch[:5]
                    occ_logit, sev_pred = self.model(x, seq_lengths=seq_lengths)
                    p_fire = torch.sigmoid(occ_logit)
                    risk = p_fire * sev_pred
                    mask = sev_avail.bool()
                    if mask.sum() > 0:
                        all_risk.extend(risk[mask].cpu().numpy())
                        all_sev.extend(y_sev[mask].cpu().numpy())
            if len(all_risk) < 3:
                return 0.0
            from scipy.stats import spearmanr

            rho, _ = spearmanr(all_risk, all_sev)
            return 0.0 if np.isnan(rho) else float(rho)


    def build_joint_risk_loader(
        x,
        seq_lengths,
        y_occ,
        y_sev,
        sev_available,
        batch_size: int = 64,
        shuffle: bool = True,
        include_indices: bool = False,
    ):
        if include_indices:
            dataset = IndexedTensorDataset(x, seq_lengths, y_occ, y_sev, sev_available)
        else:
            dataset = TensorDataset(x, seq_lengths, y_occ, y_sev, sev_available)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

else:  # pragma: no cover

    class PairwiseRiskRankingLoss:  # type: ignore[no-redef]
        def __init__(self, margin: float = 0.0, scale: float = 10.0) -> None:
            raise ImportError("torch is required for JointRisk components")


    class JointRiskTransformer:  # type: ignore[no-redef]
        def __init__(self, input_dim: int, config: dict):
            raise ImportError("torch is required for JointRisk components")


    class JointRiskTrainer:  # type: ignore[no-redef]
        def __init__(self, model, config: dict):
            raise ImportError("torch is required for JointRisk components")


    def build_joint_risk_loader(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError("torch is required for JointRisk components")
