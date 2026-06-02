from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from agni.models.joint_risk_model import JointRiskTransformer, PairwiseRiskRankingLoss


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
