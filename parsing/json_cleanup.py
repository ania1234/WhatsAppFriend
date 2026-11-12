import json

RESULT_MARKER = "Wynik:"
EXPLANATION_MARKER = "Wyjaśnienie:"
QUOTE_PAIRS = [('"', '"'), ("'", "'"), ("„", "”"), ("„", '"'), ("“", "”")]


def clean_user_message(text):
    # remove "Wynik:" and all newlines before the user message
    # (anything the model wrote before "Wynik:" is preamble, so drop it too)
    if RESULT_MARKER in text:
        text = text.split(RESULT_MARKER, 1)[1]

    # remove everything that happens after "Wyjaśnienie:" in the user message
    text = text.split(EXPLANATION_MARKER, 1)[0]

    text = text.strip()

    # the examples in the prompt show quoted answers, so the model often echoes the quotes
    for opening, closing in QUOTE_PAIRS:
        if len(text) >= 2 and text.startswith(opening) and text.endswith(closing):
            text = text[1:-1].strip()
            break

    return text


# clean up json file to remove any invalid entries (e.g. null user messages)
def clean_up_json_file(file_path, output_path=None):
    """Clean the synthetic dataset written by synthetic_data_generation.py.

    Input/output format: [{"conversations": [system, user, assistant]}, ...]
    If output_path is None, the input file is overwritten.
    """
    output_path = output_path or file_path

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []
    removed_null, removed_empty = 0, 0
    for example in data:
        messages = example["conversations"]
        for message in messages:
            if message["role"] == "user":
                message["content"] = clean_user_message(message["content"])

        user_messages = [m["content"] for m in messages if m["role"] == "user"]

        # remove the messages where the user message is starting with "NULL"
        if any(content.upper().startswith("NULL") for content in user_messages):
            removed_null += 1
            continue

        # remove empty messages (user or assistant)
        if any(not m["content"].strip() for m in messages if m["role"] != "system"):
            removed_empty += 1
            continue

        cleaned.append(example)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(
        f"{len(cleaned)}/{len(data)} conversations kept -> {output_path} "
        f"(removed: {removed_null} NULL, {removed_empty} empty)"
    )
    return cleaned


def merge_json_files(file_paths, output_path):
    """Merge cleaned-up datasets into one file, dropping exact duplicate conversations."""
    merged, seen, duplicates = [], set(), 0
    for file_path in file_paths:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        for example in data:
            key = json.dumps(example["conversations"], ensure_ascii=False, sort_keys=True)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            merged.append(example)
        print(f"{file_path}: {len(data)} conversations")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"{len(merged)} conversations merged -> {output_path} ({duplicates} exact duplicates removed)")
    return merged


if __name__ == "__main__":
    #clean_up_json_file("./data/synthetic_chat_ania.json", "./data/synthetic_chat_ania_cleaned.json")
    clean_up_json_file("./data/synthetic_chat_ania_v2.json", "./data/synthetic_chat_ania_v2_cleaned.json")
    clean_up_json_file("./data/synthetic_chat_ania_v3.json", "./data/synthetic_chat_ania_v3_cleaned.json")
    clean_up_json_file("./data/synthetic_chat_ania_v4.json", "./data/synthetic_chat_ania_v4_cleaned.json")
    merge_json_files([
        #"./data/synthetic_chat_ania_cleaned.json",
        "./data/synthetic_chat_ania_v2_cleaned.json",
        "./data/synthetic_chat_ania_v3_cleaned.json",
        "./data/synthetic_chat_ania_v4_cleaned.json",
    ], "./data/synthetic_chat_ania_merged.json")
