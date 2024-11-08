import torch
import copy
import logging
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score


# get test anomaly score
def get_anomaly_score(iterator, model, device, cof):
    model.eval()
    score_list = []
    correlation_list = []
    anomaly_score = []

    with torch.no_grad():
        for x in iterator:
            x = x.to(device)  # Put into GPU
            x = x.permute(0, 2, 1)
            test_label = torch.zeros((x.shape[0], 1)).to(device)  # all set as 0

            x_recon, recon_embed, embed, _, _, A = model(x, test_label)
            score = torch.sqrt(F.mse_loss(x, x_recon)).cpu().numpy() + cof * torch.norm(recon_embed - embed,
                                                                                        dim=1).cpu().numpy()

            score_metrics = np.abs((x_recon - x).sum(axis=2).cpu().numpy())

            anomaly_score.extend(score_metrics)
            score_list.extend(score)
            correlation_list.extend(A)

    return score_list, correlation_list, score_metrics


def adjust_pred(pred, label):
    """
    Borrow from https://github.com/NetManAIOps/OmniAnomaly/blob/master/omni_anomaly/eval_methods.py
    """
    adjusted_pred = copy.deepcopy(pred)

    anomaly_state = False
    anomaly_count = 0
    latency = 0
    for i in range(len(adjusted_pred)):
        if label[i] and adjusted_pred[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not label[j]:
                    break
                else:
                    if not adjusted_pred[j]:
                        adjusted_pred[j] = True
                        latency += 1
        elif not label[i]:
            anomaly_state = False
        if anomaly_state:
            adjusted_pred[i] = True
    return adjusted_pred


def compute_binary_metrics(anomaly_pred, anomaly_label, adjustment=True):
    if not adjustment:
        eval_anomaly_pred = anomaly_pred
        metrics = {
            "f1": f1_score(eval_anomaly_pred, anomaly_label),
            "pc": precision_score(eval_anomaly_pred, anomaly_label),
            "rc": recall_score(eval_anomaly_pred, anomaly_label),
        }
    else:
        eval_anomaly_pred = adjust_pred(anomaly_pred, anomaly_label)
        metrics = {
            "f1_adjusted": f1_score(eval_anomaly_pred, anomaly_label),
            "pc_adjusted": precision_score(eval_anomaly_pred, anomaly_label),
            "rc_adjusted": recall_score(eval_anomaly_pred, anomaly_label),
        }
    return metrics


def best_th(
        anomaly_score,
        anomaly_label,
        target_metric="f1",
        target_direction="max",
        point_adjustment=True,
):
    logging.info("Searching for the best threshold..")
    search_range = np.linspace(0, 1, 100)
    search_history = []
    if point_adjustment:
        target_metric = target_metric + "_adjusted"

    for anomaly_percent in search_range:
        theta = np.percentile(anomaly_score, 100 * (1 - anomaly_percent))
        pred = (anomaly_score >= theta).astype(int)

        metric_dict = compute_binary_metrics(pred, anomaly_label, point_adjustment)
        current_value = metric_dict[target_metric]

        logging.debug(f"th={theta}, {target_metric}={current_value}")

        search_history.append(
            {
                "best_value": current_value,
                "best_theta": theta,
                "target_metric": target_metric,
                "target_direction": target_direction,
            }
        )

    result = (
        max(search_history, key=lambda x: x["best_value"])
        if target_direction == "max"
        else min(search_history, key=lambda x: x["best_value"])
    )
    return result["best_theta"]


def compute_prediction(anomaly_score, anomaly_label, point_adjustment=True):
    th = best_th(
        anomaly_score,
        anomaly_label,
        point_adjustment=point_adjustment
    )

    anomaly_pred = (anomaly_score >= th).astype(int)
    pred_results = {"anomaly_pred": anomaly_pred, "anomaly_pred_adjusted": None, "th": th}

    if point_adjustment:
        pred_results["anomaly_pred_adjusted"] = adjust_pred(
            anomaly_pred, anomaly_label
        )

    return pred_results
