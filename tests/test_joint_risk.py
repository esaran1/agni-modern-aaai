from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from agni.experiment_utils import _build_temporal_sequences, _frame_to_joint_tensors
from agni.models.joint_risk_model import JointRiskTransformer, PairwiseRiskRankingLoss

pytest.importorskip("torch")


def test_pairwise_ranking_loss_correct_ranking() -> None:
    loss_fn = PairwiseRiskRankingLoss()
    risk = torch.tensor([0.9, 0.5, 0.1])
    sev = torch.tensor([0.8, 0.4, 0.05])
    loss = loss_fn(risk, sev)
    assert loss.item() < 0.1


def test_pairwise_ranking_loss_wrong_ranking() -> None:
    loss_fn = PairwiseRiskRankingLoss()
    risk = torch.tensor([0.1, 0.5, 0.9])
    sev = torch.tensor([0.8, 0.4, 0.05])
    loss = loss_fn(risk, sev)
    assert loss.item() > 0.5


def test_joint_loss_components() -> None:
    model = JointRiskTransformer(input_dim=18, config={"d_model": 32, "nhead": 2, "n_layers": 1})
    x = torch.randn(8, 4, 18)
    occ_logit, sev_pred = model(x)
    y_occ = torch.randint(0, 2, (8,))
    y_sev = torch.rand(8)
    sev_avail = torch.randint(0, 2, (8,))

    total, components = model.compute_loss(occ_logit, sev_pred, y_occ, y_sev, sev_avail)
    assert total.item() >= 0
    assert components["occ_loss"] >= 0
    assert components["sev_loss"] >= 0
    assert components["rank_loss"] >= 0


def test_temporal_sequences_use_multiple_timesteps() -> None:
    df = pd.DataFrame(
        {
            "patch_id": ["0_0", "0_0", "0_0"],
            "reference_date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
            "weather_vpd_mean_l7d": [1.0, 2.0, 3.0],
            "terrain_twi_mean": [0.1, 0.2, 0.3],
        }
    )
    sequences, lengths = _build_temporal_sequences(
        df,
        ["weather_vpd_mean_l7d", "terrain_twi_mean"],
        sequence_length=3,
    )
    assert sequences.shape == (3, 3, 2)
    assert lengths.tolist() == [1, 2, 3]
    assert np.allclose(sequences[0], [[0.0, 0.0], [0.0, 0.0], [1.0, 0.1]])
    assert np.allclose(sequences[1], [[0.0, 0.0], [1.0, 0.1], [2.0, 0.2]])


def test_joint_tensors_use_full_history_for_validation_rows() -> None:
    df = pd.DataFrame(
        {
            "patch_id": ["0_0", "0_0", "0_0", "0_0"],
            "reference_date": pd.to_datetime(
                ["2020-01-01", "2020-01-08", "2020-01-15", "2020-01-22"]
            ),
            "split": ["train", "train", "val", "test"],
            "weather_vpd_mean_l7d": [1.0, 2.0, 3.0, 4.0],
            "terrain_twi_mean": [0.1, 0.2, 0.3, 0.4],
            "y_occ_30d": [0, 1, 1, 0],
            "y_sev_dnbr": [0.0, 0.2, 0.4, 0.1],
            "y_sev_available": [1, 1, 1, 1],
        }
    )
    x_val, seq_lengths, _, _, _ = _frame_to_joint_tensors(
        df,
        ["weather_vpd_mean_l7d", "terrain_twi_mean"],
        sequence_length=3,
        subset_index=df.index[df["split"] == "val"],
    )
    assert tuple(x_val.shape) == (1, 3, 2)
    assert seq_lengths.tolist() == [3]
    assert np.allclose(x_val[0].tolist(), [[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]])
