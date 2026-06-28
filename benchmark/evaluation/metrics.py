def character_error_rate(prediction: str, ground_truth: str) -> float:
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    return _levenshtein(prediction, ground_truth) / len(ground_truth)


def word_error_rate(prediction: str, ground_truth: str) -> float:
    pred_words = prediction.split()
    gt_words = ground_truth.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return _levenshtein(pred_words, gt_words) / len(gt_words)


def exact_match(prediction: str, ground_truth: str) -> bool:
    return prediction == ground_truth


def _levenshtein(s1, s2) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]
