from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mne.viz import plot_topomap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from csp import MyCSP


def train(epochs_train, epochs_data, labels):
    # Custom CSP (Common Spatial Pattern)
    csp = MyCSP(n_components=16)
    # Classifier l1_ratio
    logr = LogisticRegression(solver='liblinear', verbose=False)
    # Pipeline
    model = Pipeline([("CSP", csp), ("LogisticRegression", logr)], memory=None, verbose=False)

    # monte-carlo cross-validation generator:
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    score = cross_val_score(model, epochs_train.get_data(), labels, cv=cv, verbose=False)

    model.fit(epochs_data, labels)
    accuracy = model.score(epochs_data, labels)

    # if plot:
    #     plot_results()

    return model, accuracy, score


def plot_results(subject, raw, w_times, mean_scores, csp, epochs, stats, raw_filtered):
    """
    Combine all visualizations into a single final figure.
    Adds textual statistics to the right of the temporal plot.
    """
    x_label = "Time (s)"

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    # Grid: 3 rows, 4 columns
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1])

    # 1. Raw Signals
    times = raw.times
    data_raw = raw.get_data(picks="eeg", units="uV")
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(times, data_raw.T, color='k', linewidth=0.2, alpha=0.5)
    ax1.set_title("Raw Data")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Amplitude (µV)")

    # 2. Filtered Signals
    data_filtered = raw_filtered.get_data(picks="eeg", units="uV")
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.plot(times, data_filtered.T, color='k', linewidth=0.2, alpha=0.5)
    ax2.set_title("Filtered Data (7-30 Hz)")
    ax2.set_xlabel(x_label)

    # 3. Temporal Classification Score
    ax3 = fig.add_subplot(gs[1, :3])
    ax3.plot(w_times, mean_scores, label="Mean Score", color='#1f77b4', linewidth=1.5)
    ax3.axvline(0, linestyle="--", color="k", label="Onset")
    ax3.axhline(0.5, linestyle="-", color="r", label="Chance (50%)")
    ax3.set_xlabel(x_label)
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Performance Over Time", fontweight='bold')
    ax3.legend(loc="lower right", framealpha=0.9)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # 4. Textual Statistics
    ax_text = fig.add_subplot(gs[1, 3])
    ax_text.axis("off")

    stats_content = (
        f"Mean (CV)     : {stats['mean']:.1%}\n"
        f"Std Dev       : ±{stats['std']:.1%}\n"
        f"Chance Level  : {stats['balance']:.1%}\n"
        f"Max Score     : {stats['max']:.1%}"
    )
    text_str = f"RESULTS\n{'─' * 18}\n{stats_content}"

    # Style box to prettify the text
    props = dict(boxstyle='round,pad=1', facecolor='#f0f0f0', edgecolor='#bdbdbd', alpha=0.9)
    ax_text.text(0.05, 0.5, text_str, transform=ax_text.transAxes,
                 va='center', ha='left', fontsize=10, family='monospace',
                 bbox=props, linespacing=1.6)

    # 5. CSP Patterns (Top 4 components) - full bottom row
    for i in range(min(4, len(csp.patterns_))):
        ax_pattern = fig.add_subplot(gs[2, i])
        plot_topomap(csp.patterns_[i], epochs.info, axes=ax_pattern, show=False)
        ax_pattern.set_title(f"CSP Pattern {i+1}")

    plt.suptitle(f"EEG Analysis Summary - Subject(s): {subject}", fontsize=12)
    plt.show()