import torch

from imf_video import adaptive_loss_metrics


def test_raw_metric_remains_informative_when_adaptive_objective_saturates():
    """Prevent the dashboard from mistaking the bounded objective for raw MSE."""
    loss_u = torch.tensor([1000.0, 2000.0])
    loss_v = torch.tensor([3000.0, 4000.0])

    objective, raw_metric = adaptive_loss_metrics(
        loss_u, loss_v, norm_eps=0.01, norm_p=1.0,
        loss_v_weight=1.0, elements_per_sample=100,
    )

    assert torch.isclose(raw_metric, torch.tensor(50.0))
    assert 1.999 < objective.item() < 2.0
