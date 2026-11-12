#!/usr/bin/env python3
"""Analyze Polish chat .txt exports (Python 3.10+, standard library).

Examples:
  python whatsapp_stats.py chat.txt
  python whatsapp_stats.py chat1.txt chat2.txt --speaker-1-names "Alice" "Ala" --speaker-2-names "Bob" "Robert"
  python whatsapp_stats.py chat.txt --exclude i a to --turn-gap-minutes 5
  python whatsapp_stats.py chat.txt --exclude-file stopwords.txt --mattr-window 50 \
      --json stats.json --csv speakers.csv --plots plots
  python whatsapp_stats.py --self-test

Optional plots require matplotlib: python -m pip install matplotlib
Exclusion files contain one token per line; blank lines and # comments are ignored.
Exclusions affect ONLY the top-word ranking, not lengths or lexical diversity.
Multiple files are combined in input order without deduplication. Turns and gaps
never cross file boundaries; MATTR uses the combined speaker stream. Supply files
before CLI options. Both alias lists are required when either is used; exact,
case-sensitive aliases map to speaker_1 and speaker_2, and other names are excluded.
Python API: analyze_files(["chat1.txt", "chat2.txt"],
    speaker_1_names=["Alice", "Ala"], speaker_2_names=["Bob", "Robert"])
Lengths are token counts. All tokens are case-folded (XD and xd are equivalent).
The top-word ranking merges every X followed by D's (xd, xdd, ...) into one XDDD.
Unicode emoji sequences, ASCII faces, words and numbers each count as one token.
Punctuation alone does not count. Emoji handling covers common sequences, flags,
skin tones, keycaps and ZWJ families; it is not a full Unicode grapheme segmenter.

A turn is a maximal consecutive same-speaker run with each adjacent gap <= the
configured threshold. System notices are ignored. Timestamp reversals break turns
and are reported; messages are never silently sorted. Gaps include overnight
pauses. MATTR uses overlapping windows within each speaker's concatenated token
stream (crossing message boundaries); it is null if fewer than W tokens exist.
TTR is unique tokens / tokens and depends strongly on sample size.

Parser: a 'Sender — DD/MM/YYYY HH:MM' header line, followed by the message body on
the next lines. Supports UTF-8 BOM and multiline bodies. Every line up to the next
header belongs to one message, since the export has no other message boundary.
Header-shaped lines in message bodies are inherently ambiguous in plain exports.
Messages containing only 'Image' placeholders are dropped and counted in diagnostics;
placeholders alongside other text are retained as text.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics
import sys
import textwrap
import unicodedata


HEADER = re.compile(
    r"^[\u200e\u200f\ufeff]*(.+?) \u2014 (\d{2}/\d{2}/\d{4} \d{2}:\d{2})\s*$"
)
EMOJI_BASE = r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u2300-\u23FF\u2B00-\u2BFF\u00A9\u00AE\u203C\u2049\u2122\u2139\u3030\u303D\u3297\u3299]"
EMOJI_PART = EMOJI_BASE + r"[\uFE0E\uFE0F]?[\U0001F3FB-\U0001F3FF]?"
TOKEN = re.compile(
    r"[0-9#*]\uFE0F?\u20E3"
    r"|[\U0001F1E6-\U0001F1FF]{2}"
    r"|" + EMOJI_PART + r"(?:\u200D" + EMOJI_PART + r")*(?:[\U000E0020-\U000E007E]+\U000E007F)?"
    r"|(?<!\w)(?:[xX][dD]+|[oO0][._][oO0]|[tT][._][tT])(?!\w)"
    r"|<3|[=:;8][\-^']?[)(/\\DPpOo*]+"
    r"|[^\W_]+(?:['’\-][^\W_]+)*",
    re.UNICODE,
)
XD_VARIANT = re.compile(r"xdd+")
IMAGE_PLACEHOLDER = "Image"


@dataclass
class Message:
    timestamp: datetime
    speaker: str
    text: str
    source_index: int = 0


def tokenize(text: str) -> list[str]:
    return [m.group().casefold() for m in TOKEN.finditer(unicodedata.normalize("NFC", text))]


def top_word_key(token: str) -> str:
    """Merge xd, xdd, xddd, ... into one entry; other tokens pass through."""
    return "XDDD" if XD_VARIANT.fullmatch(token) else token


def is_image_only(text: str) -> bool:
    return {line.strip() for line in text.splitlines()} - {""} == {IMAGE_PLACEHOLDER}


def parse_chat(text: str, speakers: set[str] | None = None):
    messages: list[Message] = []
    diagnostics = {"system_or_filtered_headers": 0, "orphan_lines": 0, "image_only_messages": 0}
    active = None
    for line_no, line in enumerate(text.splitlines(), 1):
        match = HEADER.match(line)
        if match:
            sender, timestamp = match.groups()
            try:
                stamp = datetime.strptime(timestamp, "%d/%m/%Y %H:%M")
            except ValueError as exc:
                raise ValueError(f"Invalid timestamp on line {line_no}: {timestamp}") from exc
            sender = sender.strip()
            if sender and (speakers is None or sender in speakers):
                active = Message(stamp, sender, "")
                messages.append(active)
                continue
            active = None
            diagnostics["system_or_filtered_headers"] += 1
        elif active is not None:
            active.text = active.text + "\n" + line if active.text else line
        elif line.strip():
            diagnostics["orphan_lines"] += 1
    retained = [message for message in messages if not is_image_only(message.text)]
    diagnostics["image_only_messages"] = len(messages) - len(retained)
    return retained, diagnostics


def percentile(values: list, p: float):
    """Linear interpolation at index (n - 1) * p / 100 (inclusive endpoints)."""
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def distribution(values: list):
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "percentiles": {f"P{p}": percentile(values, p) for p in (10, 50, 90, 99)},
    }


def mattr(tokens: list[str], window: int):
    if len(tokens) < window:
        return None
    counts = Counter(tokens[:window])
    unique_total = len(counts)
    for i in range(window, len(tokens)):
        outgoing = tokens[i - window]
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
        counts[tokens[i]] += 1
        unique_total += len(counts)
    return unique_total / ((len(tokens) - window + 1) * window)


def analyze(messages, diagnostics, gap_minutes=5.0, window=100, excluded=None, exclude_length=-1, top_n=20):
    if gap_minutes < 0 or window < 1 or top_n < 1:
        raise ValueError("Gap must be nonnegative; window and top_n must be positive.")
    excluded = excluded or set()
    names = sorted({m.speaker for m in messages})
    streams = {name: [] for name in names}
    lengths = {name: [] for name in names}
    # Per-speaker gaps are attributed to the speaker of the later message.
    incoming = {name: [] for name in names}
    same_speaker = {name: [] for name in names}
    switched = {name: [] for name in names}
    all_gaps, turns = [], []
    reversals = 0
    previous = None
    for message in messages:
        if previous is not None and previous.source_index != message.source_index:
            previous = None
        tokens = tokenize(message.text)
        streams[message.speaker].extend(tokens)
        lengths[message.speaker].append(len(tokens))
        gap = None
        if previous:
            gap = (message.timestamp - previous.timestamp).total_seconds()
            if gap < 0:
                reversals += 1
            else:
                all_gaps.append(gap)
                incoming[message.speaker].append(gap)
                target = same_speaker if previous.speaker == message.speaker else switched
                target[message.speaker].append(gap)
        if previous and previous.speaker == message.speaker and 0 <= gap <= gap_minutes * 60:
            turns[-1]["messages"] += 1
            turns[-1]["words"] += len(tokens)
        else:
            turns.append({"speaker": message.speaker, "messages": 1, "words": len(tokens)})
        previous = message
    result = {
        "schema_version": 1,
        "settings": {"turn_gap_minutes": gap_minutes, "mattr_window": window,
                     "excluded_from_top_words_only": sorted(excluded), "top_n": top_n},
        "definitions": {
            "words": "Case-folded word, number, emoji or ASCII emoticon tokens; punctuation excluded.",
            "top_words": "Ranked after exclusions; xd, xdd, xddd, ... are merged into one XDDD entry.",
            "percentiles": "Linear interpolation at (n - 1) * p / 100.",
            "turn": "Consecutive same-speaker messages, each adjacent gap within threshold.",
            "speaker_gaps": "Adjacent retained-chat gaps attributed to the later message's speaker; not time since that speaker last spoke.",
            "negative_gaps": "Excluded from gap distributions and break turns; counted in diagnostics.",
            "mattr": "Mean TTR of overlapping W-token windows of each speaker stream; null below W tokens.",
            "timestamps": "Local export times, no timezone or daylight-saving correction.",
        },
        "diagnostics": {**diagnostics, "negative_timestamp_gaps": reversals},
        "overall": {"total_messages": len(messages), "total_words": sum(map(len, streams.values())),
                    "total_turns": len(turns), "consecutive_message_gaps_seconds": distribution(all_gaps)},
        "speakers": {},
    }
    for name in names:
        tokens = streams[name]
        own_turns = [t for t in turns if t["speaker"] == name]
        excluded_keys = {top_word_key(word) for word in excluded}
        if exclude_length > 0:
            excluded_keys.update({top_word_key(word) for word in tokens if len(word) < exclude_length})
        counts = Counter(key for key in map(top_word_key, tokens) if key not in excluded_keys)
        result["speakers"][name] = {
            "total_messages": len(lengths[name]), "total_words": len(tokens),
            "message_length_words": distribution(lengths[name]),
            "top_words": [{"token": word, "count": count} for word, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:top_n]],
            "reconstructed_turns": {
                "count": len(own_turns),
                "length_words": distribution([t["words"] for t in own_turns]),
                "length_messages": distribution([t["messages"] for t in own_turns]),
            },
            "consecutive_message_gaps_seconds": distribution(incoming[name]),
            "same_speaker_consecutive_gaps_seconds": distribution(same_speaker[name]),
            "speaker_switch_gaps_seconds": distribution(switched[name]),
            "lexical_diversity": {"unique_tokens": len(set(tokens)),
                                  "ttr": len(set(tokens)) / len(tokens) if tokens else None,
                                  "mattr": mattr(tokens, window), "mattr_window": window,
                                  "mattr_windows": max(0, len(tokens) - window + 1)},
        }
    return result, {"lengths": lengths, "gaps": all_gaps}


def write_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, result):
    rows = []
    for name, stats in result["speakers"].items():
        rows.append({"speaker": name, "messages": stats["total_messages"], "words": stats["total_words"],
                     "median_message_words": stats["message_length_words"]["median"],
                     **stats["message_length_words"]["percentiles"],
                     "turns": stats["reconstructed_turns"]["count"],
                     "mean_turn_words": stats["reconstructed_turns"]["length_words"]["mean"],
                     "mean_turn_messages": stats["reconstructed_turns"]["length_messages"]["mean"],
                     "median_incoming_gap_seconds": stats["consecutive_message_gaps_seconds"]["median"],
                     **stats["lexical_diversity"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(directory, result, raw):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plots need matplotlib. Install with: python -m pip install matplotlib") from exc
    directory.mkdir(parents=True, exist_ok=True)
    names = list(raw["lengths"])
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.5), 5))
    ax.boxplot([raw["lengths"][name] for name in names])
    ax.set_xticks(range(1, len(names) + 1), names, rotation=20)
    ax.set(ylabel="Tokens per message", title="Message lengths by speaker")
    fig.tight_layout()
    fig.savefig(directory / "message_lengths.png", dpi=160)
    plt.close(fig)
    if raw["gaps"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        gaps = sorted(raw["gaps"])
        ax.step(gaps, [(i + 1) / len(gaps) for i in range(len(gaps))], where="post")
        ax.set_xscale("symlog", linthresh=60)
        ax.set(xlabel="Seconds between consecutive messages (symlog scale)",
               ylabel="Cumulative fraction", title="Consecutive-message time gaps", ylim=(0, 1.02))
        fig.tight_layout()
        fig.savefig(directory / "time_gaps.png", dpi=160)
        plt.close(fig)
    for index, (name, stats) in enumerate(result["speakers"].items(), 1):
        words = stats["top_words"][::-1]
        if not words:
            continue
        fig, ax = plt.subplots(figsize=(9, max(4, len(words) * 0.32)))
        ax.barh([w["token"] for w in words], [w["count"] for w in words])
        ax.set(title=f"Top words - {name.replace('_', ' ')}", xlabel="Occurrences")
        excluded = result["settings"]["excluded_from_top_words_only"]
        if excluded:
            note = textwrap.wrap("Excluded: " + ", ".join(excluded), width=110)
            fig.text(0.5, 0.02, "\n".join(note), ha="center", va="bottom", fontsize=8)
            fig.tight_layout(rect=(0, (len(note) * 0.16 + 0.15) / fig.get_figheight(), 1, 1))
        else:
            fig.tight_layout()
        # Numeric filenames avoid unsafe or platform-dependent speaker names.
        fig.savefig(directory / f"top_words_{index:02d}.png", dpi=160)
        plt.close(fig)


def analyze_files(
    chat_files,
    *,
    speaker_1_names=None,
    speaker_2_names=None,
    speakers=None,
    encoding="utf-8-sig",
    exclude=None,
    exclude_length=-1,
    top_n=20,
    turn_gap_minutes=5.0,
    mattr_window=100,
    json_path=None,
    csv_path=None,
    plots_dir=None,
):
    """Return combined statistics; optionally write JSON, CSV and PNG files.

    chat_files accepts a path or an iterable of paths. Speaker alias arrays contain
    exact, case-sensitive names and map to speaker_1 / speaker_2. When aliases are
    enabled, unlisted names are excluded. Otherwise original names are retained.
    Turns and time gaps stay within files; MATTR spans the combined token stream.
    Files are processed in supplied order without deduplication. Excluded tokens
    affect only top-word rankings. This function does not print or exit.
    """
    if isinstance(chat_files, (str, Path)):
        chat_files = [chat_files]
    paths = [Path(path) for path in chat_files]
    if not paths:
        raise ValueError("Provide at least one chat file.")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("The same input file was supplied more than once.")
    if not 0 <= turn_gap_minutes < float("inf"):
        raise ValueError("turn_gap_minutes must be finite and nonnegative.")
    if not isinstance(mattr_window, int) or mattr_window < 1:
        raise ValueError("mattr_window must be a positive integer.")
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    def name_list(values, label):
        if values is None:
            return None
        if isinstance(values, str):
            raise ValueError(f"{label} must be a list of names, not a string.")
        values = list(values)
        if any(not isinstance(name, str) or not name.strip() for name in values):
            raise ValueError(f"{label} must contain nonempty names.")
        return values

    names_1 = name_list(speaker_1_names, "speaker_1_names")
    names_2 = name_list(speaker_2_names, "speaker_2_names")
    selected = name_list(speakers, "speakers")
    aliases_enabled = names_1 is not None or names_2 is not None
    alias_map = {}
    if aliases_enabled:
        if not names_1 or not names_2:
            raise ValueError("Provide both nonempty speaker alias lists.")
        if selected is not None:
            raise ValueError("Use speakers or speaker alias lists, not both.")
        if set(names_1) & set(names_2):
            raise ValueError("A name cannot appear in both speaker alias lists.")
        alias_map.update(dict.fromkeys(names_1, "speaker_1"))
        alias_map.update(dict.fromkeys(names_2, "speaker_2"))
    allowed = set(alias_map) if aliases_enabled else (set(selected) if selected is not None else None)

    if isinstance(exclude, str):
        exclude = [exclude]
    excluded = {unicodedata.normalize("NFC", word).casefold() for word in (exclude or [])}

    messages = []
    diagnostics = {"system_or_filtered_headers": 0, "orphan_lines": 0, "image_only_messages": 0}
    sources = []
    for index, path in enumerate(paths):
        try:
            parsed, file_diagnostics = parse_chat(path.read_text(encoding=encoding), allowed)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        for message in parsed:
            message.source_index = index
            if aliases_enabled:
                message.speaker = alias_map[message.speaker]
        messages.extend(parsed)
        for key, count in file_diagnostics.items():
            diagnostics[key] += count
        sources.append({"path": str(path), "retained_messages": len(parsed),
                        "diagnostics": file_diagnostics})
    if not messages:
        raise ValueError("No messages parsed. Check the format, encoding and speaker names.")

    result, raw = analyze(messages, diagnostics, gap_minutes=turn_gap_minutes,
                          window=mattr_window, excluded=excluded, exclude_length = exclude_length, top_n=top_n)
    result["schema_version"] = 2
    result["sources"] = sources
    result["settings"].update({"speaker_1_names": names_1 or [],
                               "speaker_2_names": names_2 or [],
                               "speakers_filter": selected})
    result["definitions"]["file_boundaries"] = (
        "Files are aggregated in input order without deduplication. "
        "Turns and time gaps never cross file boundaries; MATTR does."
    )
    if json_path is not None:
        write_json(Path(json_path), result)
    if csv_path is not None:
        write_csv(Path(csv_path), result)
    if plots_dir is not None:
        save_plots(Path(plots_dir), result, raw)
    return result


if __name__ == "__main__":
    options = {
        "chat_files": ["./data/asia_chat_1.txt", "./data/asia_chat_2.txt", "./data/asia_chat_3.txt"],
        "speaker_1_names": ["Asia"],  # exact aliases combined under speaker_1
        "speaker_2_names": ["anierhej"],  # exact aliases combined under speaker_2
        "encoding": "utf-8-sig",
        "speakers": None,  # exact participant names
        "exclude": [ "to", "w", "i", "się", "na", "z", "ale", "a", "że", "no", "jak", "do", "jest", "o", "bo", "nie", "tak", "co", "po", "za", "ja", "image", "mi" ],  # tokens excluded only from top-word ranking
        "exclude_length": -1,  # exclude tokens shorter than this from top-word ranking
        "top_n": 10,
        "turn_gap_minutes": 120.0,
        "mattr_window": 100,
        "json_path": "./output/statistics/chat_analytics.json",  # write complete machine-readable statistics
        "csv_path": "./output/statistics/chat_analytics.csv",  # write per-speaker summary (UTF-8 BOM)
        "plots_dir": "./output/plots",  # directory for PNG plots (requires matplotlib)
    }

    try:
        result = analyze_files(**options)
        # Console output is also complete JSON, suitable for piping.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)


