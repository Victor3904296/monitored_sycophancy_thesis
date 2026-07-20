import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

survey_filename = "../../data/survey_data/AI Tone Sycophancy and Trustworthiness Survey Answers.xlsx"
output_dir = "../../data/survey_results/(6) survey plots"

tone_cols = list(range(2, 14))
trust_cols = list(range(15, 23))

tone_category_map = {
    "Low Tone Sycophancy": "low",
    "Medium Tone Sycophancy": "medium",
    "High Tone Sycophancy": "high",
}

trust_category_order = [
    "Very untrustworthy",
    "Somewhat untrustworthy",
    "Neutral",
    "Somewhat trustworthy",
    "Very trustworthy",
]

tone_prompt_query_labels = ["negative", "false_belief", "positive"]
trust_prompt_query_labels = ["baseline", "negative", "false_belief", "positive"]

prompt_display_labels = [
    "Don't think\n[correct\nanswer]",
    "Think\n[incorrect\nanswer]",
    "Think\n[correct\nanswer]"
]

tone_prompt_display_labels = prompt_display_labels
trust_prompt_display_labels = ["Baseline"] + prompt_display_labels

tone_distribution_colors = {"low": "#008000", "medium": "#F5C518", "high": "#D7191C"}
trust_distribution_colors = {
    "Very untrustworthy": "#D7191C",
    "Somewhat untrustworthy": "#F5A623",
    "Neutral": "#CCCCCC",
    "Somewhat trustworthy": "#8FD694",
    "Very trustworthy": "#008000",
}

bar_width = 0.25
x3 = np.array([0.1, 0.4, 0.7])
x4 = np.array([0, 0.3, 0.6, 0.9])

# question metadata, in column order (question 1 -> tone_cols[0] / trust_cols[0], etc.)
tone_questions = [
    {"condition": "normal", "prompt_type": "false_belief", "factuality": "CORRECT", "ai": "medium"},
    {"condition": "monitored", "prompt_type": "positive", "factuality": "CORRECT", "ai": "low"},
    {"condition": "normal", "prompt_type": "false_belief", "factuality": "INCORRECT", "ai": "high"},
    {"condition": "monitored", "prompt_type": "positive", "factuality": "INCORRECT", "ai": "low"},
    {"condition": "normal", "prompt_type": "positive", "factuality": "INCORRECT", "ai": "low"},
    {"condition": "monitored", "prompt_type": "negative", "factuality": "INCORRECT", "ai": "medium"},
    {"condition": "normal", "prompt_type": "negative", "factuality": "CORRECT", "ai": "low"},
    {"condition": "monitored", "prompt_type": "negative", "factuality": "CORRECT", "ai": "high"},
    {"condition": "normal", "prompt_type": "negative", "factuality": "INCORRECT", "ai": "medium"},
    {"condition": "monitored", "prompt_type": "false_belief", "factuality": "CORRECT", "ai": "medium"},
    {"condition": "normal", "prompt_type": "positive", "factuality": "CORRECT", "ai": "high"},
    {"condition": "monitored", "prompt_type": "false_belief", "factuality": "INCORRECT", "ai": "high"},
]

trust_questions = [
    {"condition": "normal", "prompt_type": "baseline", "factuality": "CORRECT"},
    {"condition": "monitored", "prompt_type": "baseline", "factuality": "INCORRECT"},
    {"condition": "normal", "prompt_type": "negative", "factuality": "INCORRECT"},
    {"condition": "monitored", "prompt_type": "negative", "factuality": "CORRECT"},
    {"condition": "normal", "prompt_type": "false_belief", "factuality": "INCORRECT"},
    {"condition": "monitored", "prompt_type": "false_belief", "factuality": "CORRECT"},
    {"condition": "normal", "prompt_type": "positive", "factuality": "CORRECT"},
    {"condition": "monitored", "prompt_type": "positive", "factuality": "INCORRECT"},
]


def load_survey_df():
    return pd.read_excel(survey_filename, header=0)


def add_legend(fig, patches, title):
    fig.legend(
        handles=patches,
        title=title,
        loc="center left",
        bbox_to_anchor=(0.8, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        edgecolor="lightgrey",
    )


def make_tone_distribution_plot(df, condition, ax, show_ylabel=True):
    lows, mediums, highs = [], [], []
    for pt in tone_prompt_query_labels:
        cols = [tone_cols[i] for i, question in enumerate(tone_questions)
                if question["condition"] == condition and question["prompt_type"] == pt]
        ratings = df.iloc[:, cols].values.flatten()
        ratings = [tone_category_map[r] for r in ratings]
        total = len(ratings)
        lows.append(ratings.count("low") / total)
        mediums.append(ratings.count("medium") / total)
        highs.append(ratings.count("high") / total)

    ax.bar(x3, lows, width=bar_width, color=tone_distribution_colors["low"], label="low", zorder=3)
    ax.bar(x3, mediums, width=bar_width, color=tone_distribution_colors["medium"], label="medium", bottom=lows, zorder=3)
    ax.bar(x3, highs, width=bar_width, color=tone_distribution_colors["high"], label="high",
           bottom=[l + m for l, m in zip(lows, mediums)], zorder=3)

    ax.set_ylim(0, 1.1)
    if show_ylabel:
        ax.set_ylabel("Proportion", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.set_title(f"{condition.capitalize()} condition", fontsize=10, fontweight="bold")
    ax.set_xticks(x3)
    ax.set_xticklabels(tone_prompt_display_labels, fontsize=6)
    ax.set_xlabel("Human ratings", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)


def plot_tone_distribution():
    df = load_survey_df()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

    for i, (ax, condition) in enumerate(zip(axes, ["normal", "monitored"])):
        make_tone_distribution_plot(df, condition, ax, show_ylabel=(i == 0))

    patches = [mpatches.Patch(color=tone_distribution_colors[score], label=score.capitalize()) for score in ["high", "medium", "low"]]
    add_legend(fig, patches, "Tone score")

    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    path = f"{output_dir}/tone_survey_distribution_per_condition.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {path}")
    plt.show()


def make_trust_distribution_plot(df, condition, ax, show_ylabel=True):
    proportions = {score: [] for score in trust_category_order}
    factualities = []
    for pt in trust_prompt_query_labels:
        matching_questions = [question for question in trust_questions
                               if question["condition"] == condition and question["prompt_type"] == pt]
        cols = [trust_cols[i] for i, question in enumerate(trust_questions)
                if question["condition"] == condition and question["prompt_type"] == pt]
        ratings = df.iloc[:, cols].values.flatten()
        total = len(ratings)
        for score in trust_category_order:
            proportions[score].append(list(ratings).count(score) / total)
        factualities.append(matching_questions[0]["factuality"])

    bottoms = np.zeros(len(trust_prompt_query_labels))
    for score in trust_category_order:
        heights = proportions[score]
        for j, (xi, height) in enumerate(zip(x4, heights)):
            hatch = "///" if factualities[j] != "CORRECT" else None
            edge = "black" if hatch else "none"
            ax.bar(xi, height, width=bar_width, bottom=bottoms[j], color=trust_distribution_colors[score],
                   hatch=hatch, edgecolor=edge, linewidth=0.6, zorder=3)
        bottoms += np.array(heights)

    ax.set_ylim(0, 1.1)
    if show_ylabel:
        ax.set_ylabel("Proportion", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.set_title(f"{condition.capitalize()} condition", fontsize=10, fontweight="bold")
    ax.set_xticks(x4)
    ax.set_xticklabels(trust_prompt_display_labels, fontsize=6)
    ax.set_xlabel("Human ratings", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)


def plot_trust_distribution():
    df = load_survey_df()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

    for i, (ax, condition) in enumerate(zip(axes, ["normal", "monitored"])):
        make_trust_distribution_plot(df, condition, ax, show_ylabel=(i == 0))

    patches = [mpatches.Patch(color=trust_distribution_colors[score], label=score)
               for score in reversed(trust_category_order)]
    factuality_patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Incorrect answer")
    add_legend(fig, patches + [factuality_patch], "Trustworthiness")

    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    path = f"{output_dir}/trust_survey_distribution_per_condition.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {path}")
    plt.show()


def make_tone_question_comparison_plot(df, condition, ax, show_ylabel=True):
    question_indices = [i for i, question in enumerate(tone_questions) if question["condition"] == condition]
    question_indices.sort(key=lambda i: (tone_prompt_query_labels.index(tone_questions[i]["prompt_type"]),
                                      tone_questions[i]["factuality"] != "CORRECT"))

    # group the correct/incorrect pair for each prompt type close together, with a wider gap between groups
    pair_width = 1.0
    group_gap = 0.6
    x = []
    pos = 0.0
    for idx in range(len(question_indices)):
        x.append(pos)
        pos += pair_width
        if idx % 2 == 1:
            pos += group_gap
    x = np.array(x)

    lows, mediums, highs, ai_cats, factualities = [], [], [], [], []
    for i in question_indices:
        question = tone_questions[i]
        col = tone_cols[i]
        ratings = [tone_category_map[r] for r in df.iloc[:, col].values]
        total = len(ratings)
        lows.append(ratings.count("low") / total)
        mediums.append(ratings.count("medium") / total)
        highs.append(ratings.count("high") / total)
        ai_cats.append(question["ai"])
        factualities.append(question["factuality"])

    question_bar_width = 0.7
    for xi, low, medium, high, factuality in zip(x, lows, mediums, highs, factualities):
        hatch = "///" if factuality != "CORRECT" else None
        edge = "black" if hatch else "none"
        ax.bar(xi, low, width=question_bar_width, color=tone_distribution_colors["low"],
               hatch=hatch, edgecolor=edge, linewidth=0.6, zorder=3)
        ax.bar(xi, medium, width=question_bar_width, color=tone_distribution_colors["medium"], bottom=low,
               hatch=hatch, edgecolor=edge, linewidth=0.6, zorder=3)
        ax.bar(xi, high, width=question_bar_width, color=tone_distribution_colors["high"], bottom=low + medium,
               hatch=hatch, edgecolor=edge, linewidth=0.6, zorder=3)

    for xi, ai_cat in zip(x, ai_cats):
        ax.scatter(xi, 1.08, marker="*", s=140, color=tone_distribution_colors[ai_cat],
                   edgecolor="black", linewidth=0.7, zorder=4, clip_on=False)

    group_x = [(x[2 * g] + x[2 * g + 1]) / 2 for g in range(len(x) // 2)]

    ax.set_ylim(0, 1.15)
    if show_ylabel:
        ax.set_ylabel("Proportion", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%" if v <= 1 else ""))
    ax.set_title(f"{condition.capitalize()} condition", fontsize=10, fontweight="bold")
    ax.set_xticks(group_x)
    ax.set_xticklabels(tone_prompt_display_labels, fontsize=7)
    ax.set_xlabel("Human ratings", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)


def plot_tone_question_comparison():
    df = load_survey_df()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)

    for i, (ax, condition) in enumerate(zip(axes, ["normal", "monitored"])):
        make_tone_question_comparison_plot(df, condition, ax, show_ylabel=(i == 0))

    patches = [mpatches.Patch(color=tone_distribution_colors[score], label=score.capitalize()) for score in ["high", "medium", "low"]]
    factuality_patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Incorrect answer")
    ai_marker = Line2D([0], [0], marker="*", color="w", markerfacecolor="grey", markeredgecolor="black",
                       markersize=11, label="AI rating (colour = score)")
    add_legend(fig, patches + [factuality_patch, ai_marker], "Tone score")

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    path = f"{output_dir}/tone_survey_question_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {path}")
    plt.show()


if __name__ == "__main__":
    plot_tone_distribution()
    plot_tone_question_comparison()
    plot_trust_distribution()
