from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.chrf_score import sentence_chrf
from nltk.metrics import f_measure


def compute_metrics(references: list[list[str]], hypothesis: list[str]) -> dict[str, float]:
    """
    Compute BLEU, chrF, and F1 scores for translation quality.

    Args:
        references: List of reference tokenized sentences (list of list of tokens).
        hypothesis: Hypothesis tokenized sentence (list of tokens).

    Returns:
        Dictionary containing BLEU, chrF, and F1 scores.
    """
    return {
        'bleu': sentence_bleu(references, hypothesis),
        'chrf': sentence_chrf(' '.join(references[0]), ' '.join(hypothesis)),
        'f1': f_measure(set(references[0]), set(hypothesis))
    }
