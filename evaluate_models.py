"""
Automatic evaluation of the base model and every trained LoRA adapter on the held-out test sets.

Statistics (all computed the same way for every model, whatever it was trained on):

    Likelihood of the real reply (teacher forcing, last assistant reply only)
        test loss, perplexity, next-token accuracy, paired bootstrap difference vs the base model
    Style fidelity of the generated replies vs the real replies
        reply length, lines per reply, lowercase line starts, trailing periods, emotes,
        assistant-style markup, unigram Jensen-Shannon divergence, length KS distance
    Failure modes
        replies cut off at max_new_tokens, in-reply repetition, echoing the user's message,
        lines copied verbatim from the training data
    Sanity checks on the comparison itself
        generalisation gap (train-sample loss vs test loss),
        test-set leakage (test replies that also occur in the model's training data),
        eval-loss curves logged during training

Output (in output_dir):
    results.json   -- raw per-sample losses and every generated reply
    summary.json / summary.csv / summary.md
    plots/*.png

Run from the repository root:
    uv run python evaluate_models.py                # evaluate (GPU) and write the report
    uv run python evaluate_models.py --plots-only   # rebuild summary and plots from results.json
"""

import csv
import gc
import glob
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter

import numpy as np

from parsing.chat_analytics import tokenize

# ========================================
# CONFIG
# ========================================

SAMPLES_PER_PROMPT = 3  # sampled replies per test prompt; averaged, so one unlucky sample matters less
GENERATION_BATCH_SIZE = 2 #anything bigger that 5 triggers unsloth error for 4 bit quantisation
TRAIN_PROBE_SIZE = 25  # training conversations re-scored for the generalisation gap
BOOTSTRAP_RESAMPLES = 5000
COPIED_LINE_MIN_WORDS = 4  # shorter lines ("no", "xd") repeat naturally and say nothing about memorisation

BASE_MODEL_KEY = "base"
REAL_KEY = "real"
END_OF_TURN = "<|im_end|>"

XD = re.compile(r"xd+")
MARKUP = re.compile(r"[*#`]|^\s*[-•]\s", re.MULTILINE)


# ========================================
# TEXT METRICS
# ========================================

def rouge_l(hypothesis, reference):
    """ROUGE-L F1 (0-100) on word tokens: longest common subsequence."""
    hyp, ref = tokenize(hypothesis), tokenize(reference)
    if not hyp or not ref:
        return 0.0
    previous = [0] * (len(ref) + 1)
    for h in hyp:
        current = [0]
        for j, r in enumerate(ref, 1):
            current.append(previous[j - 1] + 1 if h == r else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    return 200 * lcs / (len(hyp) + len(ref)) if lcs else 0.0


def repetition(tokens, n=3):
    """Share of word n-grams in one reply that repeat an earlier n-gram of the same reply."""
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def jensen_shannon(tokens_a, tokens_b):
    """Jensen-Shannon divergence (base 2, 0 = same word distribution, 1 = disjoint vocabularies)."""
    a, b = Counter(tokens_a), Counter(tokens_b)
    total_a, total_b = sum(a.values()), sum(b.values())
    if not total_a or not total_b:
        return 1.0
    divergence = 0.0
    for token in a.keys() | b.keys():
        p, q = a[token] / total_a, b[token] / total_b
        m = (p + q) / 2
        divergence += (p * math.log2(p / m) if p else 0) / 2 + (q * math.log2(q / m) if q else 0) / 2
    return divergence


def ks_distance(values_a, values_b):
    """Two-sample Kolmogorov-Smirnov statistic: largest gap between the two cumulative distributions."""
    a, b = np.sort(values_a), np.sort(values_b)
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                               - np.searchsorted(b, grid, side="right") / len(b))))


def is_emote(token):
    return not token[0].isalnum() or XD.fullmatch(token) is not None


def style_profile(texts):
    """Corpus-level style statistics of a list of replies."""
    tokens = [tokenize(text) for text in texts]
    lines = [line.strip() for text in texts for line in text.splitlines() if line.strip()]
    lettered = [line for line in lines if line[0].isalpha()]
    all_tokens = [token for reply in tokens for token in reply]
    return {
        "median_words": statistics.median(len(reply) for reply in tokens),
        "mean_lines": len(lines) / len(texts),
        "lowercase_start_pct": 100 * sum(line[0].islower() for line in lettered) / max(len(lettered), 1),
        "trailing_period_pct": 100 * sum(line.endswith(".") for line in lines) / max(len(lines), 1),
        "emotes_per_100_words": 100 * sum(map(is_emote, all_tokens)) / max(len(all_tokens), 1),
        "markup_pct": 100 * sum(bool(MARKUP.search(text)) for text in texts) / len(texts),
        "repetition_pct": 100 * statistics.mean(repetition(reply) for reply in tokens),
    }


def bootstrap(values, rng):
    """Mean and 95% percentile bootstrap interval over samples."""
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(0, len(values), (BOOTSTRAP_RESAMPLES, len(values)))].mean(axis=1)
    return [float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# ========================================
# MODEL EVALUATION (GPU)
# ========================================

def _conversations(dataset):
    return [row["conversations"] for row in dataset]


def _normalise(text):
    return " ".join(text.split()).casefold()


def score_conversations(model, tokenizer, conversations):
    """Cross-entropy of the last assistant reply (including its end-of-turn token), one entry per conversation."""
    import torch

    scores = []
    for conversation in conversations:
        prefix = tokenizer.apply_chat_template(conversation[:-1], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        assert full.startswith(prefix), "The chat template does not keep the prompt as a prefix of the full text."
        end = full.rfind(END_OF_TURN)
        end = end + len(END_OF_TURN) if end >= len(prefix) else len(full)

        encoded = tokenizer(full, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
        starts = encoded["offset_mapping"][0, 1:, 0]
        input_ids = encoded["input_ids"].to(model.device)
        logits = model(input_ids=input_ids).logits[0, :-1].float()
        targets = input_ids[0, 1:]
        # Only tokens that start inside the reply count; the prompt is given, not predicted.
        in_reply = ((starts >= len(prefix)) & (starts < end)).to(model.device)

        token_losses = torch.nn.functional.cross_entropy(logits[in_reply], targets[in_reply], reduction="none")
        scores.append({
            "loss": token_losses.mean().item(),
            "tokens": int(in_reply.sum()),
            "correct": int((logits[in_reply].argmax(dim=-1) == targets[in_reply]).sum()),
        })
    return scores


def generate_replies(model, tokenizer, conversations, generation):
    """SAMPLES_PER_PROMPT sampled replies to every conversation with its last assistant reply removed."""
    replies = []
    for start in range(0, len(conversations), GENERATION_BATCH_SIZE):
        batch = conversations[start:start + GENERATION_BATCH_SIZE]
        texts = [tokenizer.apply_chat_template(conversation[:-1], tokenize=False, add_generation_prompt=True)
                 for conversation in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=generation["max_new_tokens"],
            do_sample=True,
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            num_return_sequences=SAMPLES_PER_PROMPT,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        finished = (new_tokens == tokenizer.eos_token_id).any(dim=1).tolist()
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for i in range(len(batch)):
            rows = range(i * SAMPLES_PER_PROMPT, (i + 1) * SAMPLES_PER_PROMPT)
            replies.append([{"text": decoded[row].strip(), "truncated": not finished[row]} for row in rows])
        print(f"    {min(start + GENERATION_BATCH_SIZE, len(conversations))}/{len(conversations)} prompts answered")
    return replies


def supervised_texts(conversations, mask_prompt):
    """Messages the loss was computed on: the last reply when the prompt was masked, otherwise everything."""
    if mask_prompt:
        return [conversation[-1]["content"] for conversation in conversations]
    return [message["content"] for conversation in conversations for message in conversation[1:]]


def leakage(test_conversations, train_conversations, mask_prompt):
    """Share (%) of test replies found verbatim in the training data of a model."""
    supervised = "\x00".join(_normalise(text) for text in supervised_texts(train_conversations, mask_prompt))
    anywhere = "\x00".join(_normalise(message["content"])
                           for conversation in train_conversations for message in conversation[1:])
    targets = [_normalise(conversation[-1]["content"]) for conversation in test_conversations]
    return {
        "trained_on_pct": 100 * sum(target in supervised for target in targets) / len(targets),
        "in_training_text_pct": 100 * sum(target in anywhere for target in targets) / len(targets),
    }


def training_curve(adapter):
    """Loss history logged by the trainer, read from the latest checkpoint next to the adapter."""
    states = glob.glob(os.path.join(os.path.dirname(os.path.normpath(adapter)), "checkpoint-*", "trainer_state.json"))
    if not states:
        return None
    latest = max(states, key=lambda path: int(os.path.basename(os.path.dirname(path)).split("-")[-1]))
    with open(latest, encoding="utf-8") as f:
        history = json.load(f)["log_history"]
    return {
        "eval": [[entry["step"], entry["eval_loss"]] for entry in history if "eval_loss" in entry],
        "train": [[entry["step"], entry["loss"]] for entry in history if "loss" in entry],
    }


def evaluate_model(base_model_name, adapter, test_sets, train_probe, generation, seed):
    """
    Load one model (base + adapter, unless None), score and sample it on every test set.
    Returns (likelihood, generations, train_likelihood); train_likelihood is None without a train_probe.
    A separate function so every reference to the model dies when it returns and the GPU can be freed.
    """
    # Unsloth must be imported before transformers/torch to apply its patches.
    from unsloth import FastLanguageModel  # noqa: I001
    import torch
    from peft import PeftModel

    from generate_human_tests import MAX_SEQ_LENGTH

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    FastLanguageModel.for_inference(model)
    # Decoder-only models need left padding for batched generation.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    likelihood, generations, train_likelihood = {}, {}, None
    with torch.no_grad():
        for key, conversations in test_sets.items():
            print(f"  test set: {key}")
            likelihood[key] = score_conversations(model, tokenizer, conversations)
            # Same seed for every model, so the differences come from the weights, not the sampling.
            torch.manual_seed(seed)
            generations[key] = generate_replies(model, tokenizer, conversations, generation)
        if train_probe is not None:
            train_likelihood = score_conversations(model, tokenizer, train_probe)

    # free the GPU before the next one is loaded.
    # unsloth caches a view of each 4-bit weight on its quant_state; gc can't see the view's
    # C++ link back to the weight, so that cycle never frees unless the cache is dropped.
    # uncomment is batch-size is bigger than 5
    #for param in model.parameters():
    #    if getattr(param, "quant_state", None) is not None:
    #        param.quant_state._unsloth_nf4_plan = None

    del model, tokenizer
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  after {base_model_name} {adapter}: allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
        f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB")
    live = [o for o in gc.get_objects() if isinstance(o, torch.Tensor) and o.is_cuda]
    print(f"  live CUDA tensors: {len(live)}, {sum(t.numel() * t.element_size() for t in live) / 2**30:.2f} GiB")
    return likelihood, generations, train_likelihood


def evaluate_models(base_model_name, experiments, test_sets, train_sets, seed=2137):
    """Score and sample every model; returns the raw results dict (see run_automatic_evaluations)."""
    # Unsloth must be imported before transformers/torch to apply its patches.
    import unsloth  # noqa: F401, I001
    import torch

    from generate_human_tests import MAX_NEW_TOKENS, TEMPERATURE, TOP_P

    test_sets = {key: _conversations(dataset) for key, dataset in test_sets.items()}
    train_sets = {key: _conversations(dataset) for key, dataset in train_sets.items()}
    models = [{"name": BASE_MODEL_KEY, "adapter": None, "dataset": None, "mask_prompt": None}] + list(experiments)
    generation = {"max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE, "top_p": TOP_P,
                  "samples_per_prompt": SAMPLES_PER_PROMPT, "seed": seed}

    results = {
        "base_model": base_model_name,
        "generation": generation,
        "models": {model["name"]: {key: model[key] for key in ("adapter", "dataset", "mask_prompt")}
                   for model in models},
        "samples": {
            key: [{"last_user": conversation[-2]["content"], "reference": conversation[-1]["content"]}
                  for conversation in conversations]
            for key, conversations in test_sets.items()
        },
        "likelihood": {}, "train_likelihood": {}, "generations": {}, "leakage": {}, "training_curves": {},
    }

    for model_spec in models:
        name, adapter = model_spec["name"], model_spec["adapter"]
        print(f"Evaluating: {name}")
        probe = None
        if adapter is not None:
            train = train_sets[model_spec["dataset"]]
            probe = random.Random(seed).sample(train, min(TRAIN_PROBE_SIZE, len(train)))
        likelihood, generations, train_likelihood = evaluate_model(
            base_model_name, adapter, test_sets, probe, generation, seed)

        results["likelihood"][name] = likelihood
        results["generations"][name] = generations

        if adapter is not None:
            results["train_likelihood"][name] = train_likelihood
            results["leakage"][name] = {key: leakage(conversations, train, model_spec["mask_prompt"])
                                        for key, conversations in test_sets.items()}
            results["training_curves"][name] = training_curve(adapter)

            # Lines of the generated replies that the model was trained to produce, word for word.
            trained_lines = {_normalise(line)
                             for text in supervised_texts(train, model_spec["mask_prompt"])
                             for line in text.splitlines()}
            for replies in results["generations"][name].values():
                for reply in (reply for prompt_replies in replies for reply in prompt_replies):
                    long_lines = [_normalise(line) for line in reply["text"].splitlines()
                                  if len(tokenize(line)) >= COPIED_LINE_MIN_WORDS]
                    reply["long_lines"] = len(long_lines)
                    reply["copied_lines"] = sum(line in trained_lines for line in long_lines)

    return results


# ========================================
# SUMMARY
# ========================================

# (key, label, number format, what "best" means: "lower", "higher", or "real" = closest to the real replies)
METRICS = [
    ("loss", "Test loss", "{:.3f}", "lower"),
    ("perplexity", "Perplexity", "{:.1f}", "lower"),
    ("token_accuracy_pct", "Next-token accuracy %", "{:.1f}", "higher"),
    ("median_words", "Median words per reply", "{:.0f}", "real"),
    ("mean_lines", "Lines per reply", "{:.1f}", "real"),
    ("lowercase_start_pct", "Lines starting lowercase %", "{:.0f}", "real"),
    ("trailing_period_pct", "Lines ending with a period %", "{:.0f}", "real"),
    ("emotes_per_100_words", "Emotes per 100 words", "{:.1f}", "real"),
    ("markup_pct", "Replies with markup (* # `) %", "{:.0f}", "real"),
    ("unigram_jsd", "Word distribution distance (JSD)", "{:.2f}", "lower"),
    ("length_ks", "Length distribution distance (KS)", "{:.2f}", "lower"),
    ("truncated_pct", "Replies cut at token limit %", "{:.0f}", "lower"),
    ("repetition_pct", "Repeated 3-grams in a reply %", "{:.1f}", "real"),
    ("echo_rouge_l", "Echo of the user's message (ROUGE-L)", "{:.1f}", "real"),
    ("copied_lines_pct", "Lines copied from training %", "{:.1f}", "lower"),
]
METRIC_INFO = {key: (label, fmt, best) for key, label, fmt, best in METRICS}
LIKELIHOOD_METRICS = ["loss", "token_accuracy_pct"]
STYLE_METRICS = [key for key, *_ in METRICS[3:]]


def summarise(results, seed=2137):
    """
    Aggregate the raw results into {test set: {model or "real": {metric: value}}}.
    Metrics with a bootstrap interval also get "<metric>_ci": [low, high]; "loss_vs_base" is the
    paired per-sample difference to the base model (negative = better than base).
    """
    rng = np.random.default_rng(seed)
    summary = {}
    for test, samples in results["samples"].items():
        references = [sample["reference"] for sample in samples]
        users = [sample["last_user"] for sample in samples]
        reference_tokens = [tokenize(reference) for reference in references]
        real = style_profile(references)
        real["echo_rouge_l"] = statistics.mean(map(rouge_l, references, users))
        real["n"] = len(samples)
        summary[test] = {REAL_KEY: real}
        base_losses = np.array([score["loss"] for score in results["likelihood"][BASE_MODEL_KEY][test]])

        for model in results["models"]:
            scores = results["likelihood"][model][test]
            losses = np.array([score["loss"] for score in scores])
            tokens = np.array([score["tokens"] for score in scores])
            correct = np.array([score["correct"] for score in scores])
            replies = results["generations"][model][test]
            flat = [reply for prompt_replies in replies for reply in prompt_replies]
            texts = [reply["text"] for reply in flat]

            row = style_profile(texts)
            for key, values in {
                "loss": losses,
                "token_accuracy_pct": 100 * correct / tokens,
                "echo_rouge_l": [statistics.mean(rouge_l(reply["text"], user) for reply in prompt_replies)
                                 for prompt_replies, user in zip(replies, users)],
                "truncated_pct": [100 * statistics.mean(reply["truncated"] for reply in prompt_replies)
                                  for prompt_replies in replies],
            }.items():
                row[key], *row[key + "_ci"] = bootstrap(values, rng)
            if model != BASE_MODEL_KEY:
                row["loss_vs_base"], *row["loss_vs_base_ci"] = bootstrap(losses - base_losses, rng)
                row["wins_vs_base_pct"] = 100 * float(np.mean(losses < base_losses))
                long_lines = sum(reply["long_lines"] for reply in flat)
                row["copied_lines_pct"] = 100 * sum(reply["copied_lines"] for reply in flat) / max(long_lines, 1)
            # Token-weighted, so one very short reply does not dominate the perplexity.
            row["perplexity"] = float(np.exp((losses * tokens).sum() / tokens.sum()))
            row["unigram_jsd"] = jensen_shannon([token for reply in map(tokenize, texts) for token in reply],
                                                [token for reply in reference_tokens for token in reply])
            row["length_ks"] = ks_distance([len(tokenize(text)) for text in texts],
                                           [len(reply) for reply in reference_tokens])
            row["trained_on"] = results["models"][model]["dataset"]
            summary[test][model] = row

    for model, scores in results["train_likelihood"].items():
        test = results["models"][model]["dataset"]
        train_loss, *train_ci = bootstrap([score["loss"] for score in scores], rng)
        summary[test][model].update(train_loss=train_loss, train_loss_ci=train_ci,
                                    generalisation_gap=summary[test][model]["loss"] - train_loss)
    return summary


def best_model(rows, metric):
    """Model with the best value of a metric among `rows` ({model: row}, may include "real")."""
    best = METRIC_INFO[metric][2]
    values = {model: row[metric] for model, row in rows.items() if model != REAL_KEY and metric in row}
    if not values or (best == "real" and metric not in rows.get(REAL_KEY, {})):
        return None
    if best == "real":
        return min(values, key=lambda model: abs(values[model] - rows[REAL_KEY][metric]))
    return (min if best == "lower" else max)(values, key=values.get)


def display_name(model):
    return {REAL_KEY: "real replies"}.get(model, model.removesuffix("_model").replace("_", " "))


def write_summary_files(summary, results, output_dir):
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    extra = ["loss_vs_base", "wins_vs_base_pct", "train_loss", "generalisation_gap"]
    keys = [key for key, *_ in METRICS]
    with open(os.path.join(output_dir, "summary.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_set", "model", "trained_on"]
                        + [column for key in keys + extra for column in (key, key + "_ci_low", key + "_ci_high")])
        for test, rows in summary.items():
            for model, row in rows.items():
                cells = []
                for key in keys + extra:
                    cells += [row.get(key, ""), *row.get(key + "_ci", ["", ""])]
                writer.writerow([test, model, row.get("trained_on") or ""] + cells)

    lines = ["# Automatic model evaluation", ""]
    for test, rows in summary.items():
        lines += [f"## Test set: {test} ({rows[REAL_KEY]['n']} conversations)", "",
                  "| Metric | " + " | ".join(display_name(model) for model in rows) + " | Best is |",
                  "|---|" + "---:|" * len(rows) + "---|"]
        for key, label, fmt, best in METRICS:
            winner = best_model(rows, key)
            cells = []
            for model, row in rows.items():
                cell = fmt.format(row[key]) if key in row else "–"
                if key + "_ci" in row:
                    cell += " ({} – {})".format(*(fmt.format(bound) for bound in row[key + "_ci"]))
                cells.append(f"**{cell}**" if model == winner else cell)
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {'closest to real' if best == 'real' else best} |")
        lines += ["", "Loss difference to the base model (paired, per conversation; negative = better):", ""]
        for model, row in rows.items():
            if "loss_vs_base" in row:
                low, high = row["loss_vs_base_ci"]
                verdict = "significant" if low > 0 or high < 0 else "not significant"
                lines.append(f"- {display_name(model)}: {row['loss_vs_base']:+.3f} ({low:+.3f} – {high:+.3f}), "
                             f"better on {row['wins_vs_base_pct']:.0f}% of conversations, {verdict} at 95%")
        lines.append("")

    lines += ["## Test-set leakage", "",
              "Share of test replies that occur verbatim in the model's own training data. "
              "\"Trained on\" = inside text the loss was computed on; \"in training text\" also counts prompts.", "",
              "| Model | Test set | Trained on % | In training text % |", "|---|---|---:|---:|"]
    for model, per_test in results["leakage"].items():
        for test, leak in per_test.items():
            lines.append(f"| {display_name(model)} | {test} | {leak['trained_on_pct']:.0f} "
                         f"| {leak['in_training_text_pct']:.0f} |")
    lines += ["", "Intervals are 95% percentile bootstrap intervals over test conversations "
                  f"({BOOTSTRAP_RESAMPLES} resamples). Generation metrics average "
                  f"{results['generation']['samples_per_prompt']} sampled replies per prompt.", ""]
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ========================================
# PLOTS
# ========================================

SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Fixed order, one colour per adapter; the base model is the grey baseline, the real replies are ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def save_plots(summary, results, plots_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    os.makedirs(plots_dir, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": SECONDARY, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": SECONDARY, "ytick.labelcolor": INK,
    })

    models = list(results["models"])
    adapters = [model for model in models if model != BASE_MODEL_KEY]
    colors = {BASE_MODEL_KEY: MUTED, REAL_KEY: INK,
              **{model: SERIES[i % len(SERIES)] for i, model in enumerate(adapters)}}
    tests = list(summary)

    def clean(ax, grid="x"):
        ax.spines[["top", "right", "left" if grid == "x" else "top"]].set_visible(False)
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    def finish(fig, title, note, filename):
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
        fig.supxlabel("\n".join(_wrap(note, int(fig.get_figwidth() * 15))), x=0.01, ha="left", fontsize=8.5,
                      color=SECONDARY)
        fig.savefig(os.path.join(plots_dir, filename), dpi=160)
        plt.close(fig)

    def model_axis(ax, names, labelled):
        ax.set_yticks(range(len(names)), [display_name(name) for name in names] if labelled else [""] * len(names))
        ax.set_ylim(len(names) - 0.4, -0.6)

    def trained_note(test):
        return f"{test} test set · {summary[test][REAL_KEY]['n']} conversations"

    def interval_panels(metrics, title, note, filename):
        """Rows = metrics, columns = test sets; a dot with its 95% interval per model."""
        fig, axes = plt.subplots(len(metrics), len(tests), figsize=(5.2 * len(tests) + 1.2, 2.6 * len(metrics) + 1),
                                 squeeze=False, layout="constrained")
        for r, metric in enumerate(metrics):
            label, fmt, best = METRIC_INFO[metric]
            for c, test in enumerate(tests):
                ax, rows = axes[r][c], summary[test]
                winner = best_model(rows, metric)
                for y, model in enumerate(models):
                    value, (low, high) = rows[model][metric], rows[model][metric + "_ci"]
                    ax.plot([low, high], [y, y], color=colors[model], linewidth=2, solid_capstyle="round")
                    ax.plot(value, y, "o", color=colors[model], markersize=8,
                            markeredgecolor=SURFACE, markeredgewidth=2)
                    ax.annotate(fmt.format(value), (high, y), xytext=(6, 0), textcoords="offset points",
                                va="center", color=INK, fontweight="bold" if model == winner else "normal")
                model_axis(ax, models, labelled=c == 0)
                clean(ax)
                ax.margins(x=0.12)
                ax.set_title(f"{label} · {test} test set")
                ax.set_xlabel(f"{label} ({best} is better)")
        finish(fig, title, note, filename)

    # ---- 1. likelihood of the real reply
    interval_panels(
        LIKELIHOOD_METRICS, "How well does each model predict the real replies?",
        "Teacher-forced on the held-out conversations; only the last reply is scored, for every model. "
        "Lines are 95% bootstrap intervals; bold = best.",
        "01_test_loss.png")

    # ---- 3. paired loss difference vs base
    fig, axes = plt.subplots(1, len(tests), figsize=(5.2 * len(tests) + 1.2, 0.5 * len(adapters) + 2.2),
                             squeeze=False, layout="constrained")
    for c, test in enumerate(tests):
        ax = axes[0][c]
        for y, model in enumerate(adapters):
            row = summary[test][model]
            low, high = row["loss_vs_base_ci"]
            ax.plot([low, high], [y, y], color=colors[model], linewidth=2, solid_capstyle="round")
            ax.plot(row["loss_vs_base"], y, "o", color=colors[model], markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            ax.annotate(f"{row['loss_vs_base']:+.2f} · wins {row['wins_vs_base_pct']:.0f}%", (high, y),
                        xytext=(6, 0), textcoords="offset points", va="center", color=INK)
        ax.axvline(0, color=AXIS, linewidth=1)
        model_axis(ax, adapters, labelled=c == 0)
        clean(ax)
        ax.margins(x=0.25)
        ax.set_title(trained_note(test))
        ax.set_xlabel("Test loss minus base-model loss (left of 0 = better than base)")
    finish(fig, "Is each adapter really better than the base model?",
           "Paired difference per conversation with a 95% bootstrap interval; an interval that crosses 0 is not a "
           "significant difference. \"wins\" = share of conversations with a lower loss than base.",
           "03_loss_vs_base.png")

    # ---- 4. style fidelity and failure modes, one figure per test set
    for test in tests:
        rows = summary[test]
        fig, axes = plt.subplots(3, 4, figsize=(15, 8.2), layout="constrained")
        for i, (ax, metric) in enumerate(zip(axes.flat, STYLE_METRICS)):
            label, fmt, best = METRIC_INFO[metric]
            shown = [model for model in models if metric in rows[model]]
            winner = best_model(rows, metric)
            values = [rows[model][metric] for model in shown]
            ax.barh([models.index(model) for model in shown], values, height=0.55,
                    color=[colors[model] for model in shown])
            for model, value in zip(shown, values):
                ax.annotate(fmt.format(value), (value, models.index(model)), xytext=(4, 0),
                            textcoords="offset points", va="center", color=INK, zorder=4,
                            fontweight="bold" if model == winner else "normal",
                            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
            if metric in rows[REAL_KEY]:
                ax.axvline(rows[REAL_KEY][metric], color=INK, linewidth=1.5)
            model_axis(ax, models, labelled=i % 4 == 0)
            clean(ax)
            ax.margins(x=0.18)
            ax.set_xlim(left=0)
            ax.set_title(label, fontsize=9.5)
            ax.set_xlabel("closest to the real replies is better" if best == "real" else f"{best} is better",
                          fontsize=8)
        fig.legend(handles=[Line2D([], [], color=INK, linewidth=1.5, label="real replies in the test set")],
                   loc="outside upper right", frameon=False)
        finish(fig, f"Do the generated replies look like the real ones? — {trained_note(test)}",
               f"Computed over all sampled replies ({results['generation']['samples_per_prompt']} per prompt). "
               "Bold = best model. Lines copied from training: lines of at least "
               f"{COPIED_LINE_MIN_WORDS} words that occur verbatim in the adapter's training targets.",
               f"04_style_and_failures_{test}.png")

    # ---- 5. reply length distribution
    fig, axes = plt.subplots(1, len(tests), figsize=(5.6 * len(tests) + 1, 4.6), squeeze=False,
                             layout="constrained", sharey=True)
    for c, test in enumerate(tests):
        ax = axes[0][c]
        series = {model: [len(tokenize(reply["text"])) for prompt_replies in results["generations"][model][test]
                          for reply in prompt_replies] for model in models}
        series[REAL_KEY] = [len(tokenize(sample["reference"])) for sample in results["samples"][test]]
        for model, lengths in series.items():
            lengths = np.sort(lengths)
            ax.step(np.concatenate([[0], lengths]), np.arange(len(lengths) + 1) / len(lengths), where="post",
                    color=colors[model], linewidth=2.6 if model == REAL_KEY else 2, label=display_name(model))
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 20, 50, 100, 200], ["0", "2", "4", "6", "8", "10", "20", "50", "100", "200"])
        ax.minorticks_off()
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.02)
        clean(ax, grid="y")
        ax.set_title(trained_note(test))
        ax.set_xlabel("Words per reply (linear to 10, then log scale)")
        if c == 0:
            ax.set_ylabel("Share of replies at most this long")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside right upper", frameon=False)
    finish(fig, "Reply length: generated vs real",
           "Cumulative distributions; a model whose curve follows the black one writes replies of realistic length.",
           "05_reply_length.png")

    # ---- 6. generalisation gap
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(adapters) + 2), layout="constrained")
    for y, model in enumerate(adapters):
        row = summary[results["models"][model]["dataset"]][model]
        ax.plot([row["train_loss"], row["loss"]], [y, y], color=colors[model], linewidth=2)
        ax.plot(row["train_loss"], y, "o", color=SURFACE, markersize=8, markeredgecolor=colors[model],
                markeredgewidth=2)
        ax.plot(row["loss"], y, "o", color=colors[model], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"gap {row['generalisation_gap']:+.2f}", (max(row["train_loss"], row["loss"]), y),
                    xytext=(8, 0), textcoords="offset points", va="center", color=INK)
    model_axis(ax, adapters, labelled=True)
    clean(ax)
    ax.margins(x=0.2)
    ax.set_xlabel("Loss on the last reply")
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=SURFACE, markeredgecolor=SECONDARY, markeredgewidth=2,
               markersize=8, label=f"training sample ({TRAIN_PROBE_SIZE} conversations)"),
        Line2D([], [], marker="o", linestyle="", color=SECONDARY, markersize=8, label="held-out test set"),
    ], loc="outside upper right", frameon=False, ncols=2)
    finish(fig, "Generalisation gap", "Each adapter on its own kind of data. A test loss far to the right of the "
           "training loss means memorisation rather than learning.", "06_generalisation_gap.png")

    # ---- 7. leakage
    columns = [(test, kind) for test in tests for kind in ("trained_on_pct", "in_training_text_pct")]
    matrix = np.array([[results["leakage"][model][test][kind] for test, kind in columns] for model in adapters])
    fig, ax = plt.subplots(figsize=(2.1 * len(columns) + 2.4, 0.6 * len(adapters) + 2.2), layout="constrained")
    ax.imshow(matrix, cmap=LinearSegmentedColormap.from_list("blue", SEQUENTIAL), vmin=0, vmax=100, aspect="auto")
    for (r, c), value in np.ndenumerate(matrix):
        ax.text(c, r, f"{value:.0f}%", ha="center", va="center", color="#ffffff" if value > 55 else INK)
    ax.set_xticks(range(len(columns)), [f"{test} test\n{'reply was a training target' if kind == 'trained_on_pct' else 'reply anywhere in training text'}"
                                        for test, kind in columns])
    ax.set_yticks(range(len(adapters)), [display_name(model) for model in adapters])
    ax.set_xticks(np.arange(len(columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(adapters) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.spines[:].set_visible(False)
    finish(fig, "Has the model already seen the test replies?",
           "Share of test replies found verbatim in the adapter's training data. High values mean the test scores "
           "of that model on that test set are optimistic.", "07_test_set_leakage.png")

    # ---- 8. training curves
    curves = {model: curve for model, curve in results["training_curves"].items() if curve}
    if curves:
        fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
        for model, curve in curves.items():
            steps, losses = zip(*curve["eval"])
            ax.plot(steps, losses, color=colors[model], linewidth=2, label=display_name(model),
                    linestyle="-" if results["models"][model]["mask_prompt"] else (0, (4, 2)))
        clean(ax, grid="y")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Eval loss logged by the trainer")
        fig.legend(loc="outside right upper", frameon=False)
        finish(fig, "Eval loss during training",
               "Dashed = loss over all tokens (prompt not masked), solid = loss on the last reply only; the two "
               "kinds are not comparable with each other. Use 01_test_loss.png to compare models.",
               "08_training_curves.png")

    # ---- 9. summary table
    table_metrics = ["loss", "perplexity", "token_accuracy_pct", "median_words",
                     "lowercase_start_pct", "emotes_per_100_words", "markup_pct", "unigram_jsd",
                     "truncated_pct", "copied_lines_pct"]
    row_height, column_width, name_width = 0.36, 1.12, 2.5
    table_rows = sum(len(summary[test]) + 3.7 for test in tests)
    fig = plt.figure(figsize=(name_width + column_width * len(table_metrics) + 0.3, row_height * table_rows + 1.3))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, fig.get_figwidth())
    ax.set_ylim(fig.get_figheight(), 0)
    ax.axis("off")
    ax.text(0.2, 0.4, "Automatic evaluation summary", fontsize=13, fontweight="bold", va="center")
    y = 1.0
    for test in tests:
        rows = summary[test]
        ax.text(0.2, y, trained_note(test), fontweight="bold", va="center")
        for i, metric in enumerate(table_metrics):
            label, _, best = METRIC_INFO[metric]
            arrow = {"lower": "↓", "higher": "↑", "real": "≈ real"}[best]
            ax.text(name_width + column_width * (i + 1) - 0.1, y + row_height * 2.1,
                    "\n".join(_wrap(label, 15)) + f"\n{arrow}", ha="right", va="bottom", fontsize=7.5, color=SECONDARY)
        y += row_height * 2.8
        for model in [*models, REAL_KEY]:
            ax.plot([0.2, fig.get_figwidth() - 0.2], [y - row_height / 2] * 2, color=GRID, linewidth=0.8)
            ax.plot(0.32, y, "o", color=colors[model], markersize=7)
            trained = rows[model].get("trained_on")
            ax.text(0.5, y, display_name(model) + (f"  ({trained})" if trained else ""), va="center")
            for i, metric in enumerate(table_metrics):
                if metric in rows[model]:
                    ax.text(name_width + column_width * (i + 1) - 0.1, y, METRIC_INFO[metric][1].format(rows[model][metric]),
                            ha="right", va="center",
                            fontweight="bold" if model == best_model(rows, metric) else "normal")
            y += row_height
        y += row_height * 0.9
    ax.text(0.2, y - row_height * 0.3, "Bold = best model in the column. ↓ lower is better, ↑ higher is better, "
            "≈ real: closest to the real replies. In brackets: the data the adapter was trained on.",
            fontsize=8, color=SECONDARY, va="center")
    fig.savefig(os.path.join(plots_dir, "00_summary_table.png"), dpi=160)
    plt.close(fig)


def _wrap(text, width):
    lines, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    return lines + [current]


# ========================================
# MAIN
# ========================================

def write_report(results, output_dir, seed=2137):
    """Summary files and plots from the raw results; needs no GPU."""
    summary = summarise(results, seed=seed)
    write_summary_files(summary, results, output_dir)
    save_plots(summary, results, os.path.join(output_dir, "plots"))
    print(f"Report written to {output_dir}")
    return summary


def run_automatic_evaluations(base_model_name, experiments, test_sets, train_sets, output_dir, seed=2137):
    """
    base_model_name -- Hugging Face base model; evaluated without an adapter as "base"
    experiments     -- list of {"name", "adapter" (directory), "dataset" (key into train_sets),
                       "mask_prompt" (as used in training)}
    test_sets       -- {key: held-out dataset with a "conversations" column}; every model is scored on all of them
    train_sets      -- {key: training dataset}; used for the generalisation gap, leakage and copied lines
    output_dir      -- results.json, summary.* and plots/ are written here

    Returns the summary dict (see summarise).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = evaluate_models(base_model_name, experiments, test_sets, train_sets, seed=seed)
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return write_report(results, output_dir, seed=seed)


if __name__ == "__main__":
    output_dir = "./output/statistics/model_evaluation"
    seed = 2137

    if "--plots-only" in sys.argv:
        with open(os.path.join(output_dir, "results.json"), encoding="utf-8") as f:
            write_report(json.load(f), output_dir, seed=seed)
        sys.exit()

    from datasets import load_dataset

    # Same splits as main.py: 25 test, then 25 eval, the rest is training data.
    test_sets, train_sets = {}, {}
    for key, path in {"chat": "./data/final_chat_ania.json", "synthetic": "./data/synthetic_chat_ania_merged.json"}.items():
        split = load_dataset("json", data_files=path, split="train").train_test_split(test_size=25, seed=seed)
        test_sets[key] = split["test"]
        train_sets[key] = split["train"].train_test_split(test_size=25, seed=seed)["train"]

    run_automatic_evaluations(
        base_model_name="speakleash/Bielik-4.5B-v3.0-Instruct",
        experiments=[
            {"name": "full_chat_dump_model", "adapter": "./output/full_chat_dump_model/adapter",
             "dataset": "chat", "mask_prompt": False},
            {"name": "masked_chat_dump_q_k_model", "adapter": "./output/masked_chat_dump_q_k_model/adapter",
             "dataset": "chat", "mask_prompt": True},
            {"name": "masked_chat_dump_q_k_v_o_model", "adapter": "./output/masked_chat_dump_q_k_v_o_model/adapter",
             "dataset": "chat", "mask_prompt": True},
            {"name": "synthetic_chat_model", "adapter": "./output/synthetic_chat_model/adapter",
             "dataset": "synthetic", "mask_prompt": True},
        ],
        test_sets=test_sets,
        train_sets=train_sets,
        output_dir=output_dir,
        seed=seed,
    )
