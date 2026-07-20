import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import cohen_kappa_score
from code_directory._1_main_pipeline import utils
from code_directory._1_main_pipeline.t_test import compute_tone_score, TONE_MAP

SURVEY_DATA_FILENAME = "../../data/survey_data/AI Tone Sycophancy and Trustworthiness Survey Answers.xlsx"
EVAL_RESULTS_FILENAME = "../../data/results/(2) evaluations/eval_results.jsonl"
OUTPUT_DIR = "../../data/survey_results/(7) survey tests"

N_TONE_ITEMS = 12
TONE_COL_START = 2  # first tone-sycophancy answer column, 0-indexed

N_TRUST_ITEMS = 8
TRUST_COL_START = 15  # first trustworthiness answer column, 0-indexed

TRUST_MAP = {
    "very untrustworthy": 1,
    "somewhat untrustworthy": 2,
    "neutral": 3,
    "somewhat trustworthy": 4,
    "very trustworthy": 5,
}

# tone sycophancy question number (survey order) -> AI eval lookup key
TONE_ITEM_METADATA = {
    1:  {"condition": "normal",    "index": 86},
    2:  {"condition": "monitored", "index": 7},
    3:  {"condition": "normal",    "index": 58},
    4:  {"condition": "monitored", "index": 1783},
    5:  {"condition": "normal",    "index": 663},
    6:  {"condition": "monitored", "index": 329},
    7:  {"condition": "normal",    "index": 833},
    8:  {"condition": "monitored", "index": 201},
    9:  {"condition": "normal",    "index": 925},
    10: {"condition": "monitored", "index": 246},
    11: {"condition": "normal",    "index": 843},
    12: {"condition": "monitored", "index": 1670},
}

# trustworthiness question number (survey order) -> condition + prompt type
TRUST_ITEM_METADATA = {
    1: {"condition": "normal",    "prompt_type": "baseline"},
    2: {"condition": "monitored", "prompt_type": "baseline"},
    3: {"condition": "normal",    "prompt_type": "negative"},
    4: {"condition": "monitored", "prompt_type": "negative"},
    5: {"condition": "normal",    "prompt_type": "false_belief"},
    6: {"condition": "monitored", "prompt_type": "false_belief"},
    7: {"condition": "normal",    "prompt_type": "positive"},
    8: {"condition": "monitored", "prompt_type": "positive"},
}

PROMPT_TYPES = ["baseline", "negative", "false_belief", "positive"]


def load_human_ratings():
    survey_df = pd.read_excel(SURVEY_DATA_FILENAME)
    tone_sycophancy_columns = survey_df.columns[TONE_COL_START:TONE_COL_START + N_TONE_ITEMS]
    tone_score_rows = []
    for participant_id, response_row in survey_df.iterrows():
        for item_id, column_name in enumerate(tone_sycophancy_columns, start=1):
            tone_label = response_row[column_name].replace(" Tone Sycophancy", "").strip().lower()
            tone_score_rows.append({
                "item_id": item_id,
                "participant_id": participant_id,
                "tone_score": TONE_MAP.get(tone_label, float("nan")),
            })
    return pd.DataFrame(tone_score_rows).dropna()


def load_ai_ratings():
    eval_df = pd.DataFrame(utils.load_from_jsonl(EVAL_RESULTS_FILENAME))

    tone_metadata_df = pd.DataFrame([
        {"item_id": item_id, "condition": item_meta["condition"], "index": item_meta["index"]}
        for item_id, item_meta in TONE_ITEM_METADATA.items()
    ])
    filtered_eval_df = eval_df.merge(tone_metadata_df, on=["condition", "index"], how="inner")
    filtered_eval_df["tone_score"] = filtered_eval_df["eval_output"].apply(compute_tone_score)

    matched_item_ids = set(filtered_eval_df["item_id"])
    for item_id in sorted(set(TONE_ITEM_METADATA) - matched_item_ids):
        print(f"warning: no AI eval match for item {item_id} ({TONE_ITEM_METADATA[item_id]})")

    return filtered_eval_df[["item_id", "tone_score"]].dropna()


def compute_simple_agreement(human_ratings_df, ai_ratings_df):
    human_majority_score_by_item = human_ratings_df.groupby("item_id")["tone_score"].agg(lambda x: x.mode().iloc[0])
    ai_score_by_item = ai_ratings_df.set_index("item_id")["tone_score"]
    shared_item_ids = human_majority_score_by_item.index.intersection(ai_score_by_item.index)
    is_match = human_majority_score_by_item.loc[shared_item_ids] == ai_score_by_item.loc[shared_item_ids]
    return is_match.mean(), int(is_match.sum()), len(shared_item_ids)


def compute_agreement_breakdown(human_ratings_df, ai_ratings_df):
    human_majority_score_by_item = human_ratings_df.groupby("item_id")["tone_score"].agg(lambda x: x.mode().iloc[0])
    ai_score_by_item = ai_ratings_df.set_index("item_id")["tone_score"]
    shared_item_ids = sorted(human_majority_score_by_item.index.intersection(ai_score_by_item.index))

    breakdown_rows = []
    for item_id in shared_item_ids:
        human_score = human_majority_score_by_item.loc[item_id]
        ai_score = ai_score_by_item.loc[item_id]
        breakdown_rows.append({
            "item_id": item_id,
            "human_majority_score": human_score,
            "ai_score": ai_score,
            "match": human_score == ai_score,
            "distance": abs(human_score - ai_score),
        })
    breakdown_df = pd.DataFrame(breakdown_rows)

    human_label_counts = human_majority_score_by_item.loc[shared_item_ids].value_counts().sort_index()
    ai_label_counts = ai_score_by_item.loc[shared_item_ids].value_counts().sort_index()
    label_distribution_df = pd.DataFrame({
        "human_majority_count": human_label_counts,
        "ai_count": ai_label_counts,
    }).fillna(0).astype(int)

    return breakdown_df, label_distribution_df


def compute_weighted_kappa(human_ratings_df, ai_ratings_df):
    human_majority_score_by_item = human_ratings_df.groupby("item_id")["tone_score"].agg(lambda x: x.mode().iloc[0])
    ai_score_by_item = ai_ratings_df.set_index("item_id")["tone_score"]
    shared_item_ids = human_majority_score_by_item.index.intersection(ai_score_by_item.index)
    kappa = cohen_kappa_score(
        human_majority_score_by_item.loc[shared_item_ids],
        ai_score_by_item.loc[shared_item_ids],
        weights="quadratic",
    )
    return kappa, len(shared_item_ids)


def load_trust_ratings():
    survey_df = pd.read_excel(SURVEY_DATA_FILENAME)
    trust_columns = survey_df.columns[TRUST_COL_START:TRUST_COL_START + N_TRUST_ITEMS]
    trust_score_rows = []
    for participant_id, response_row in survey_df.iterrows():
        for item_id, column_name in enumerate(trust_columns, start=1):
            trust_label = response_row[column_name].strip().lower()
            item_meta = TRUST_ITEM_METADATA[item_id]
            trust_score_rows.append({
                "participant_id": participant_id,
                "condition": item_meta["condition"],
                "prompt_type": item_meta["prompt_type"],
                "trust_score": TRUST_MAP.get(trust_label, float("nan")),
            })
    return pd.DataFrame(trust_score_rows).dropna()


def sig_flag(p_value):
    if pd.isna(p_value): return "n/a"
    if p_value < 0.001: return "*** (p<0.001)"
    if p_value < 0.01:  return "**  (p<0.01)"
    if p_value < 0.05:  return "*   (p<0.05)"
    return "not significant"


def ttest_trust_by_prompt_type(trust_ratings_df):
    result_rows = []
    for prompt_type in PROMPT_TYPES:
        prompt_type_df = trust_ratings_df[trust_ratings_df["prompt_type"] == prompt_type]
        paired_wide_df = prompt_type_df.pivot(index="participant_id", columns="condition", values="trust_score").dropna()

        score_diff = paired_wide_df["normal"] - paired_wide_df["monitored"]
        if (score_diff == 0).all():
            n_pairs, t_stat, p_value = len(score_diff), float("nan"), float("nan")
        else:
            t_stat, p_value = ttest_rel(paired_wide_df["normal"], paired_wide_df["monitored"])
            n_pairs = len(paired_wide_df)

        result_rows.append({
            "prompt_type": prompt_type,
            "n": n_pairs,
            "mean_normal": round(paired_wide_df["normal"].mean(), 4),
            "mean_monitored": round(paired_wide_df["monitored"].mean(), 4),
            "median_normal": paired_wide_df["normal"].median(),
            "median_monitored": paired_wide_df["monitored"].median(),
            "n_higher_in_normal": int((score_diff > 0).sum()),
            "n_higher_in_monitored": int((score_diff < 0).sum()),
            "n_tied": int((score_diff == 0).sum()),
            "t_stat": round(t_stat, 4) if not pd.isna(t_stat) else float("nan"),
            "p_value": f"{p_value:.20f}" if not pd.isna(p_value) else float("nan"),
            "p_value_rounded": round(p_value, 4) if not pd.isna(p_value) else float("nan"),
            "significance": sig_flag(p_value),
        })

    return pd.DataFrame(result_rows)


def main():
    human_ratings_df = load_human_ratings()
    ai_ratings_df = load_ai_ratings()

    simple_agreement_rate, n_matches, n_shared_items = compute_simple_agreement(human_ratings_df, ai_ratings_df)
    weighted_kappa_value, n_shared_items = compute_weighted_kappa(human_ratings_df, ai_ratings_df)

    agreement_results_df = pd.DataFrame([
        {
            "metric": "simple_agreement",
            "value": round(simple_agreement_rate, 4),
            "n_items": n_shared_items,
            "n_matches": n_matches,
        },
        {
            "metric": "weighted_kappa",
            "value": round(weighted_kappa_value, 4),
            "n_items": n_shared_items,
        },
    ])

    agreement_results_df.to_csv(f"{OUTPUT_DIR}/human_ai_agreement.csv", index=False)
    print(agreement_results_df.to_string(index=False))

    breakdown_df, label_distribution_df = compute_agreement_breakdown(human_ratings_df, ai_ratings_df)
    breakdown_df.to_csv(f"{OUTPUT_DIR}/tone_agreement_breakdown.csv", index=False)
    label_distribution_df.to_csv(f"{OUTPUT_DIR}/tone_label_distribution.csv")
    print(breakdown_df.to_string(index=False))
    print(label_distribution_df.to_string())

    trust_ratings_df = load_trust_ratings()
    trust_test_results_df = ttest_trust_by_prompt_type(trust_ratings_df)
    trust_test_results_df.round(4).to_csv(f"{OUTPUT_DIR}/ttest_trust_normal_vs_monitored.csv", index=False)
    print(trust_test_results_df.to_string(index=False))


if __name__ == "__main__":
    main()
