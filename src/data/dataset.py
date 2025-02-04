from datasets import load_dataset


DATASET_NAMES = [
    "matejklemen/clc_fce",  # src_tokens, tgt_tokens, corrections
    "jhu-clsp/jfleg", # sentence, correction
    "agentlans/high-quality-english-sentences", # only text, number of rows about 1.5 million
]

# train_dataset = load_dataset(DATASET_NAME, split="train")
# valid_dataset = load_dataset(DATASET_NAME, split="validation")
# test_dataset = load_dataset(DATASET_NAME, split="test")
