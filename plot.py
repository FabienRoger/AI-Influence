# %%
from matplotlib import pyplot as plt
import json
from matplotlib.ticker import MaxNLocator

data = json.load(open("results/summary_stats.json"))

models = ["gpt-3.5-turbo", "gpt-4"]
methods = ["direct", "sentiment"]
templates = {"fabien": "Long template", "minimal": "Minimal template"}

# print male-female bias in a errorbar horizontal
fig, axs = plt.subplots(1, 2, dpi=300, figsize=(6, 2.5), sharey=True)

for j, ax, method in zip(range(len(axs)), axs, methods):
    ax.set_title(f"Method: {method}")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=2))

    y_labels = []
    for i, model in enumerate(models):
        this_data = data[f"mf-{model}-{method}"]
        x = [d["mean"] for d in this_data.values()]
        x_err = [d["2-sigma-u"] for d in this_data.values()]
        y_labels += [k + " name" for k in this_data.keys()]
        y = range(i * len(x), (i + 1) * len(x))
        ax.errorbar(x, y, xerr=x_err, fmt="o", label=model)

    for i in range(1, len(y_labels) - 2, 2):
        ax.axhline(i + 0.5, color="black", linestyle="--", alpha=0.5)

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Average score")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    if j == 0:
        ax.legend()
# %%
# print male-female bias in a errorbar horizontal
fig, axs = plt.subplots(1, 2, dpi=300, figsize=(6, 2.5), sharey=True)

for j, ax, method in zip(range(len(axs)), axs, methods):
    ax.set_title(f"Method: {method}")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=2))

    y_labels = []
    for i, model in enumerate(models):
        this_data = data[f"mfant-{model}-{method}"]
        x = [d["mean"] for d in this_data.values()]
        x_err = [d["2-sigma-u"] for d in this_data.values()]
        y_labels += [k + " name" for k in this_data.keys()]
        y = range(i * len(x), (i + 1) * len(x))
        ax.errorbar(x, y, xerr=x_err, fmt="o", label=model)

    for i in range(1, len(y_labels) - 2, 2):
        ax.axhline(i + 0.5, color="black", linestyle="--", alpha=0.5)

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Average score")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    if j == 0:
        ax.legend()
# %%
# print normal-alternative bias in a errorbar horizontal
normal_alternative_keys = ["normal", "alternative"]

for normal_alternative in [True, False]:
    figsize = (6, 3) if normal_alternative else (8, 4)
    line_every = 2 if normal_alternative else 4
    fig, axs = plt.subplots(1, 2, dpi=300, figsize=figsize, sharey=True)

    for j, ax, method in zip(range(len(axs)), axs, methods):
        ax.set_title(f"Method: {method}")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2))

        y_labels = []
        for i, model in enumerate(models):
            this_data = {}
            for template_name, template in templates.items():
                this_data.update(
                    {
                        f"{k} ({template})": v
                        for k, v in data[f"{model}-{template_name}-{method}"].items()
                        if not (k in normal_alternative_keys) ^ normal_alternative
                    }
                )

            x = [d["mean"] for d in this_data.values()]
            x_err = [d["2-sigma-u"] for d in this_data.values()]
            y_labels += [k for k in this_data.keys()]
            y = range(i * len(x), (i + 1) * len(x))
            ax.errorbar(x, y, xerr=x_err, fmt="o", label=model)

        for i in range(line_every - 1, len(y_labels) - line_every, line_every):
            ax.axhline(i + 0.5, color="black", linestyle="--", alpha=0.5)

        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Average score")
        ax.set_ylabel("Normal or alternative?" if normal_alternative else "Excluded publication category")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        if j == 0:
            ax.legend()
    plt.plot()
# %%
# print normal-alternative bias in a errorbar horizontal
normal_alternative_keys = ["Pro-AI", "Anti-AI"]

figsize = (6, 3)
line_every = 2
fig, axs = plt.subplots(1, 2, dpi=300, figsize=figsize, sharey=True)

for j, ax, method in zip(range(len(axs)), axs, methods):
    ax.set_title(f"Method: {method}")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=2))

    y_labels = []
    for i, model in enumerate(models):
        this_data = {}
        for template_name, template in templates.items():
            this_data.update(
                {
                    f"{k} ({template})": v
                    for k, v in data[f"{model}-{template_name}-{method}"].items()
                    if k in normal_alternative_keys
                }
            )

        x = [d["mean"] for d in this_data.values()]
        x_err = [d["2-sigma-u"] for d in this_data.values()]
        y_labels += [k for k in this_data.keys()]
        y = range(i * len(x), (i + 1) * len(x))
        ax.errorbar(x, y, xerr=x_err, fmt="o", label=model)

    for i in range(line_every - 1, len(y_labels) - line_every, line_every):
        ax.axhline(i + 0.5, color="black", linestyle="--", alpha=0.5)

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Average score")
    ax.set_ylabel("Explicitely Pro or Anti AI?")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    if j == 0:
        ax.legend()
# %%
