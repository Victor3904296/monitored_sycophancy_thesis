from code_directory._1_main_pipeline import utils
from collections import Counter
import pandas as pd

inference_normal_filename = "../../data/results/(1) inferences/answers_normal.jsonl"
inference_monitored_filename = "../../data/results/(1) inferences/answers_reasoning_monitored.jsonl"
nulls_normal_filename = "../../data/results/(0) nulls/answers_normal_nulls.jsonl"
nulls_monitored_filename = "../../data/results/(0) nulls/answers_monitored_nulls.jsonl"

eval_normal_filename = "../../data/results/(2) evaluations/eval_normal.jsonl"
eval_monitored_filename = "../../data/results/(2) evaluations/eval_reasoning_monitored.jsonl"
eval_results_filename = "../../data/results/(2) evaluations/eval_results.jsonl"
eval_normal_nulls_filename = "../../data/results/(0) nulls/eval_normal_nulls.jsonl"
eval_monitored_nulls_filename = "../../data/results/(0) nulls/eval_monitored_nulls.jsonl"

# FIRST PART:
# Checks whether all indices that were run appear in the inference and null files
# this is to make sure there are no missing indices or duplicates
file_pairs = [
    (inference_normal_filename, nulls_normal_filename),
    (inference_monitored_filename, nulls_monitored_filename),
    (eval_normal_filename, eval_normal_nulls_filename, nulls_normal_filename),
    (eval_monitored_filename, eval_monitored_nulls_filename, nulls_monitored_filename),
]

for pair in file_pairs:
    inference_file, null_file = pair[0], pair[1]
    extra_null_files = list(pair[2:])  # inference nulls, only present for eval pairs

    inference_responses = utils.load_from_jsonl(inference_file)
    null_responses = utils.load_from_jsonl(null_file)

    inference_indices = [row["index"] for row in inference_responses]
    null_indices = [row["index"] for row in null_responses]

    extra_null_indices = []
    for extra_file in extra_null_files:
        extra_rows = utils.load_from_jsonl(extra_file)
        extra_null_indices += [row["index"] for row in extra_rows]

    inference_duplicates = {
        index: count
        for index, count in Counter(inference_indices).items()
        if count > 1
    }
    null_duplicates = {
        index: count
        for index, count in Counter(null_indices).items()
        if count > 1
    }

    cross_duplicates = sorted(set(inference_indices) & set(null_indices))

    all_indices = inference_indices + null_indices + extra_null_indices
    all_counts = Counter(all_indices)

    missing = [
        i
        for i in range(min(all_counts), max(all_counts) + 1)
        if i not in all_counts
    ]

    print(f"{inference_file} + {null_file}" + (f" (+ {extra_null_files})" if extra_null_files else "") + ":")
    print(f"  Inference rows: {len(inference_responses)}")
    print(f"  Null rows: {len(null_responses)}")
    if extra_null_indices:
        print(f"  Extra null rows (inference nulls): {len(extra_null_indices)}")
    print(f"  Total rows: {len(all_indices)}")
    print(f"  Unique indices across all files: {len(all_counts)}")

    if inference_duplicates:
        print(f"  Duplicates within inference file: {inference_duplicates}")
    else:
        print("  No duplicates within inference file")

    if null_duplicates:
        print(f"  Duplicates within null file: {null_duplicates}")
    else:
        print("  No duplicates within null file")

    if cross_duplicates:
        print(f"  Indices appearing in both files: {cross_duplicates}")
    else:
        print("  No indices appear in both files")

    if missing:
        print(f"  Missing indices across all files: {missing}")
    else:
        print("  No missing indices across all files")

    print()

df = pd.DataFrame(utils.load_from_jsonl(eval_results_filename))
df["question_id"] = df["index"] // 4  # same as in metrics.py

# filter to baseline only
baseline_df = df[df["metadata"].apply(lambda m: m["prompt_template"]) == "{question}"]

normal_ids = set(baseline_df[baseline_df["condition"] == "normal"]["question_id"])
monitored_ids = set(baseline_df[baseline_df["condition"] == "monitored"]["question_id"])
shared_ids = normal_ids & monitored_ids

baseline_df = baseline_df[baseline_df["question_id"].isin(shared_ids)]


# SECOND PART:
# This checks for which exact questions the 2 baselines differed
# This was checked because the baselines had the exact same accuracy
def get_grade(eval_output):
    if eval_output is None:
        return "INCORRECT"
    return "CORRECT" if "FACTUAL GRADE: CORRECT" in eval_output else "INCORRECT"

normal_grades = {row["question_id"]: get_grade(row["eval_output"])
                 for _, row in baseline_df[baseline_df["condition"] == "normal"].iterrows()}
monitored_grades = {row["question_id"]: get_grade(row["eval_output"])
                    for _, row in baseline_df[baseline_df["condition"] == "monitored"].iterrows()}

differing = [qid for qid in shared_ids if normal_grades[qid] != monitored_grades[qid]]

print(f"Shared question_ids (both conditions present): {len(shared_ids)}")
print(f"Question_ids with differing grades: {len(differing)}")
if differing:
    for qid in sorted(differing):
        print(f"  [question_id={qid}] normal={normal_grades[qid]}, monitored={monitored_grades[qid]}")
else:
    print("All baseline grades match across both conditions.")
