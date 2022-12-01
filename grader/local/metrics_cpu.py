import numpy as np
import scipy.stats as stats


def compute_metrics(total_losses, all_score_predictions, all_score_targets):
    """ Computes Pearson correlation and accuracy within 0.5 and 1 of target score and adds each to total_losses dict. """
    total_losses['rmse'] = np.sqrt(np.mean((all_score_predictions - all_score_targets)**2))
    total_losses['pearson'] = stats.pearsonr(all_score_predictions, all_score_targets)[0]
    total_losses['within_0.5'] = _accuracy_within_margin(all_score_predictions, all_score_targets, 0.5)
    total_losses['within_1'] = _accuracy_within_margin(all_score_predictions, all_score_targets, 1)
    total_losses['within_1.5'] = _accuracy_within_margin(all_score_predictions, all_score_targets, 1.5)
    total_losses['within_2'] = _accuracy_within_margin(all_score_predictions, all_score_targets, 2)

def _accuracy_within_margin(score_predictions, score_target, margin):
    """ Returns the percentage of predicted scores that are within the provided margin from the target score. """
    return np.sum(
        np.where(
            np.abs(score_predictions - score_target) <= margin,
            np.ones(len(score_predictions)),
            np.zeros(len(score_predictions)))).item() / len(score_predictions) * 100
