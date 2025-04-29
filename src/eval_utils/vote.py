# cp from https://github.com/openreasoner/openr/blob/main/reason/reranking/vote_utils.py

from collections import Counter, defaultdict
from typing import List


COR_ALL = "pass"
MAJORITY_VOTE = "majority_vote"
ORM_VOTE = "orm_vote"
ORM_MAX = "orm_max"
PRM_MIN_MAX = "prm_min_max"
PRM_MIN_VOTE = "prm_min_vote"
PRM_LAST_MAX = "prm_last_max"
PRM_LAST_VOTE = "prm_last_vote"


def _agg_pass(samples):
    for sample in samples:
        if sample["correct"]:
            return sample
    return samples[0]


def _agg_majority_vote(samples):
    ans_list = [sample["ans"] for sample in samples]
    counts = Counter(ans_list)
    most_common = max(counts, key=counts.get)
    chosen_list = [sample for sample in samples if sample["ans"] == most_common]
    return chosen_list[0]


def _agg_orm_vote(x_list: List[str], v_list: List[float]):
    assert len(x_list) == len(v_list)
    x_dict = defaultdict(lambda: 0.0)
    for x, v in zip(x_list, v_list):
        x_dict[x] += v

    highest_x = max(x_dict, key=x_dict.get)
    return highest_x


def _agg_orm_max(x_list: List[str], v_list: List[float]):
    text_max = x_list[v_list.index(max(v_list))]
    return text_max


def _agg_prm_min_max(samples):
    score_list = [sample["scores"] for sample in samples]
    score_list = [min(v) if v else -1.0 for v in score_list]
    chosen_sample = samples[score_list.index(max(score_list))]
    return chosen_sample


def _agg_prm_last_max(samples):
    score_list = [sample["scores"] for sample in samples]
    score_list = [v[-1] if v else -1.0 for v in score_list]
    chosen_sample = samples[score_list.index(max(score_list))]
    return chosen_sample


def _agg_prm_min_vote(samples):
    ans_dict = defaultdict(lambda: 0.0)
    for sample in samples:
        ans_dict[sample["ans"]] += min(sample["scores"])

    highest_answer = max(ans_dict, key=ans_dict.get)
    chosen_list = [sample for sample in samples if sample["ans"] == highest_answer]
    return chosen_list[0]


def _agg_prm_last_vote(samples):
    ans_dict = defaultdict(lambda: 0.0)
    for sample in samples:
        ans_dict[sample["ans"]] += sample["scores"][-1]

    highest_answer = max(ans_dict, key=ans_dict.get)
    chosen_list = [sample for sample in samples if sample["ans"] == highest_answer]
    return chosen_list[0]


AGG_FN_MAP = {
    COR_ALL: _agg_pass,
    MAJORITY_VOTE: _agg_majority_vote,
    # ORM_VOTE: _agg_orm_vote,
    # ORM_MAX: _agg_orm_max,
    PRM_MIN_MAX: _agg_prm_min_max,
    PRM_MIN_VOTE: _agg_prm_min_vote,
    PRM_LAST_MAX: _agg_prm_last_max,
    PRM_LAST_VOTE: _agg_prm_last_vote,
}
