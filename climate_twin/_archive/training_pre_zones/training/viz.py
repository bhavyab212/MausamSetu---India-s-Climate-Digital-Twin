"""Visualization helpers for post-round diagnostics.

Generates confusion matrix, reliability diagram, skill-by-region,
skill-by-month, and sample prediction comparison figures.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def confusion_matrix_figure(
    pred: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.01,
) -> go.Figure:
    """Rain occurrence confusion matrix (2×2).

    pred, truth: (N, H, W) or (H, W) — normalized [0,1]
    mask: (H, W)
    threshold: in normalized space
    """
    valid = mask > 0.5
    if pred.ndim == 3:
        p_flat = np.array([p[valid] for p in pred]).flatten()
        t_flat = np.array([t[valid] for t in truth]).flatten()
    else:
        p_flat = pred[valid]
        t_flat = truth[valid]

    p_occ = p_flat > threshold
    t_occ = t_flat > threshold

    tp = int(np.sum(p_occ & t_occ))
    fp = int(np.sum(p_occ & ~t_occ))
    fn = int(np.sum(~p_occ & t_occ))
    tn = int(np.sum(~p_occ & ~t_occ))

    cm = np.array([[tn, fp], [fn, tp]])
    labels = ["No rain", "Rain"]

    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
    ))
    fig.update_layout(
        title=f"Confusion Matrix (threshold={threshold:.3f})",
        xaxis_title="Predicted", yaxis_title="Observed",
        height=320, margin=dict(l=50, r=20, t=40, b=40),
    )

    # Add metrics as annotation
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0
    pod_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    far_val = fp / (tp + fp) if (tp + fp) > 0 else 0
    csi_val = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0

    fig.add_annotation(
        text=f"POD={pod_val:.3f} | FAR={far_val:.3f} | CSI={csi_val:.3f} | Acc={accuracy:.3f}",
        xref="paper", yref="paper", x=0.5, y=-0.15,
        showarrow=False, font=dict(size=11),
    )
    return fig


def reliability_diagram(
    pred_probs: np.ndarray,
    observed_binary: np.ndarray,
    n_bins: int = 10,
) -> go.Figure:
    """Reliability diagram: predicted probability vs observed frequency.

    pred_probs: continuous predictions (used as probability proxies)
    observed_binary: 0/1 ground truth
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    observed_freq = np.zeros(n_bins)
    mean_pred = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (pred_probs >= bin_edges[i]) & (pred_probs < bin_edges[i + 1])
        if in_bin.sum() > 0:
            observed_freq[i] = observed_binary[in_bin].mean()
            mean_pred[i] = pred_probs[in_bin].mean()
            counts[i] = in_bin.sum()

    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True,
                        vertical_spacing=0.05)

    # Reliability curve
    fig.add_trace(go.Scatter(x=mean_pred, y=observed_freq, mode="lines+markers",
                             name="Model", line=dict(color="#0F4C81", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Perfect", line=dict(color="gray", dash="dash")), row=1, col=1)

    # Histogram of predictions
    fig.add_trace(go.Bar(x=bin_centers, y=counts, name="Count",
                         marker_color="rgba(15,76,129,0.3)"), row=2, col=1)

    fig.update_layout(
        height=400, margin=dict(l=50, r=20, t=40, b=40),
        title="Reliability Diagram",
        showlegend=True,
    )
    fig.update_yaxes(title_text="Observed frequency", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Predicted probability", row=2, col=1)
    return fig


def skill_by_lead_time(
    rmse_per_lead: list[float],
    baseline_rmse_per_lead: list[float] | None = None,
    unit: str = "",
) -> go.Figure:
    """Skill vs lead time (if multiple steps predicted)."""
    leads = list(range(1, len(rmse_per_lead) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=leads, y=rmse_per_lead, name="Model",
                             mode="lines+markers", line=dict(color="#0F4C81", width=2)))
    if baseline_rmse_per_lead:
        fig.add_trace(go.Scatter(x=leads, y=baseline_rmse_per_lead, name="Persistence",
                                 mode="lines+markers", line=dict(color="#F57602", dash="dash")))

    fig.update_layout(
        height=300, margin=dict(l=50, r=20, t=40, b=40),
        title="Skill vs Lead Time",
        xaxis_title="Lead (steps)", yaxis_title=f"RMSE ({unit})",
    )
    return fig


def sample_predictions_figure(
    observed: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    title: str = "Sample Prediction",
) -> go.Figure:
    """Side-by-side: observed, predicted, error maps.

    observed, predicted: (H, W) — single channel, normalized [0,1]
    mask: (H, W)
    """
    error = (predicted - observed) * mask

    fig = make_subplots(rows=1, cols=3, subplot_titles=["Observed", "Predicted", "Error (Pred−Obs)"])

    # Apply mask for display
    obs_display = np.where(mask > 0.5, observed, np.nan)
    pred_display = np.where(mask > 0.5, predicted, np.nan)
    err_display = np.where(mask > 0.5, error, np.nan)

    common_kw = dict(showscale=False)
    fig.add_trace(go.Heatmap(z=obs_display, colorscale="viridis", **common_kw), row=1, col=1)
    fig.add_trace(go.Heatmap(z=pred_display, colorscale="viridis", **common_kw), row=1, col=2)
    fig.add_trace(go.Heatmap(z=err_display, colorscale="RdBu", zmid=0, **common_kw), row=1, col=3)

    fig.update_layout(
        title=title, height=300, margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig


def training_curves_figure(
    train_losses: list[float],
    val_losses: list[float],
    grad_norms: list[float] | None = None,
    lrs: list[float] | None = None,
) -> go.Figure:
    """Multi-panel training diagnostics."""
    n_panels = 2 + (1 if grad_norms else 0) + (1 if lrs else 0)
    titles = ["Loss", "Val Loss"]
    if grad_norms:
        titles.append("Grad Norm")
    if lrs:
        titles.append("Learning Rate")

    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=titles)

    fig.add_trace(go.Scatter(y=train_losses, name="Train", line=dict(color="#0F4C81")), row=1, col=1)
    fig.add_trace(go.Scatter(y=val_losses, name="Val", line=dict(color="#F57602")), row=1, col=2)

    col_idx = 3
    if grad_norms:
        fig.add_trace(go.Scatter(y=grad_norms, name="Grad", line=dict(color="#07945B")), row=1, col=col_idx)
        col_idx += 1
    if lrs:
        fig.add_trace(go.Scatter(y=lrs, name="LR", line=dict(color="#8B5CF6")), row=1, col=col_idx)

    fig.update_layout(height=250, margin=dict(l=40, r=20, t=40, b=30), showlegend=False)
    return fig


def metric_comparison_bars(
    model_metrics: dict[str, float],
    persistence_metrics: dict[str, float],
    climatology_metrics: dict[str, float],
    highlight_keys: list[str] | None = None,
) -> go.Figure:
    """Bar chart comparing model vs baselines across metrics."""
    keys = highlight_keys or ["rmse", "mae", "pod", "csi", "far"]
    keys = [k for k in keys if k in model_metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=keys, y=[model_metrics.get(k, 0) for k in keys],
        name="Model", marker_color="#0F4C81",
    ))
    fig.add_trace(go.Bar(
        x=keys, y=[persistence_metrics.get(k, 0) for k in keys],
        name="Persistence", marker_color="#F57602",
    ))
    fig.add_trace(go.Bar(
        x=keys, y=[climatology_metrics.get(k, 0) for k in keys],
        name="Climatology", marker_color="#94A3B8",
    ))

    fig.update_layout(
        barmode="group", height=320, margin=dict(l=40, r=20, t=40, b=40),
        title="Model vs Baselines",
        yaxis_title="Metric value",
    )
    return fig


def walkforward_progress_figure(rounds: list[dict]) -> go.Figure:
    """The big cumulative learning graph across walk-forward rounds.

    rounds: list (oldest→newest) of dicts with keys:
        round_num, val_label (str), rmse, mae, csi, bias,
        pers_rmse, clim_rmse, train_loss, val_loss
    Four stacked panels sharing the round axis:
        1) Error (RMSE) — model vs persistence vs climatology
        2) Skill (CSI, higher=better)
        3) Final train / val loss per round
        4) Bias (systematic over/under-prediction)
    """
    x = [r["round_num"] for r in rounds]

    def col(key):
        return [r.get(key) if isinstance(r.get(key), (int, float)) else None for r in rounds]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=[
            "Error — Val RMSE vs baselines (↓ better)",
            "Skill — CSI (↑ better)",
            "Optimization — final train / val loss per round (↓ better)",
            "Bias — mean(pred−obs), 0 is unbiased",
        ],
    )
    # Panel 1: RMSE + baselines
    fig.add_trace(go.Scatter(x=x, y=col("rmse"), name="Model RMSE", mode="lines+markers",
                             line=dict(color="#0F4C81", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=col("pers_rmse"), name="Persistence", mode="lines",
                             line=dict(color="#F57602", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=col("clim_rmse"), name="Climatology", mode="lines",
                             line=dict(color="#94A3B8", dash="dot")), row=1, col=1)
    # Panel 2: CSI
    fig.add_trace(go.Scatter(x=x, y=col("csi"), name="CSI", mode="lines+markers",
                             line=dict(color="#22C55E", width=2), showlegend=False), row=2, col=1)
    # Panel 3: losses
    fig.add_trace(go.Scatter(x=x, y=col("train_loss"), name="Train loss", mode="lines+markers",
                             line=dict(color="#0F4C81")), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=col("val_loss"), name="Val loss", mode="lines+markers",
                             line=dict(color="#F57602")), row=3, col=1)
    # Panel 4: bias
    fig.add_trace(go.Scatter(x=x, y=col("bias"), name="Bias", mode="lines+markers",
                             line=dict(color="#A855F7"), showlegend=False), row=4, col=1)
    fig.add_hline(y=0.0, line=dict(color="#556", dash="dot"), row=4, col=1)

    fig.update_layout(
        height=760, margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", y=1.06, x=0),
        title="Walk-Forward Learning Progression (x-axis = round over time)",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Walk-forward round →", row=4, col=1)
    return fig


_DASH_PALETTE = ["#F4A34A", "#22D3EE", "#22C55E", "#A855F7", "#EF4444",
                 "#EAB308", "#3B82F6", "#EC4899"]


def training_dashboard_figure(series_by_model: dict) -> go.Figure:
    """DataRobot-style 2×2 training dashboard comparing multiple models.

    series_by_model: {name: {"train":[...], "val":[...], "lr":[...], "grad":[...]}}
    x-axis = iteration (epoch index concatenated across that model's rounds).
    Panels: Loss (train solid / val dashed), Skill (1−val loss), Learning Rate, Grad Norm.
    """
    fig = make_subplots(
        rows=2, cols=2, vertical_spacing=0.13, horizontal_spacing=0.09,
        subplot_titles=["Loss  (train ─  ·  val ┄)", "Skill  (1 − val loss, ↑ better)",
                        "Learning Rate", "Gradient Norm"],
    )
    for i, (name, s) in enumerate(series_by_model.items()):
        c = _DASH_PALETTE[i % len(_DASH_PALETTE)]
        tr = s.get("train") or []
        vl = s.get("val") or []
        lr = s.get("lr") or []
        gd = s.get("grad") or []
        xt = list(range(len(tr)))
        # Panel 1 — loss (train solid = legend entry, val dashed)
        fig.add_trace(go.Scatter(x=xt, y=tr, name=name, legendgroup=name,
                                 line=dict(color=c, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(vl))), y=vl, name=name, legendgroup=name,
                                 showlegend=False, line=dict(color=c, width=1.5, dash="dot")),
                      row=1, col=1)
        # Panel 2 — skill = 1 - val loss
        sk = [1.0 - v for v in vl]
        fig.add_trace(go.Scatter(x=list(range(len(sk))), y=sk, name=name, legendgroup=name,
                                 showlegend=False, line=dict(color=c, width=2)), row=1, col=2)
        # Panel 3 — LR
        fig.add_trace(go.Scatter(x=list(range(len(lr))), y=lr, name=name, legendgroup=name,
                                 showlegend=False, line=dict(color=c, width=2)), row=2, col=1)
        # Panel 4 — grad norm
        fig.add_trace(go.Scatter(x=list(range(len(gd))), y=gd, name=name, legendgroup=name,
                                 showlegend=False, line=dict(color=c, width=2)), row=2, col=2)

    for r, c in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.update_xaxes(title_text="Iterations", row=r, col=c, gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", row=r, col=c)
    fig.update_layout(
        height=620, margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", y=1.08, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,16,28,0.6)",
        hovermode="x unified",
    )
    return fig
