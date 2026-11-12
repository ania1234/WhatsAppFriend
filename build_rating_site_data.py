"""
Turn the generated answers (generate_human_tests.py) into the data file used by the
rating website, with the models anonymised.

Writes two files:
    site_data_path  -- shipped with the website; answers carry only an opaque letter:
        {"speaker_name": "Ania",
         "questions_per_rater": 25,
         "questions": [{"id": 1, "question": "...",
                        "answers": [{"id": "c", "text": "..."}, ...]}, ...]}
    answer_key_path -- stays local; maps the letters back to the models:
        {"1": {"c": "base", "a": "full_chat_dump_model", ...}, ...}

Run from the repository root:
    uv run python build_rating_site_data.py
"""

import json
import os
import random
import string


def build_rating_site_data(human_tests_path, site_data_path, answer_key_path, speaker_real_name,
                           questions_per_rater, seed=2137):
    """
    human_tests_path    -- JSON written by generate_human_tests
    site_data_path      -- JSON file for the website (no model names inside)
    answer_key_path     -- JSON file mapping answer letters to models (keep it out of the website)
    speaker_real_name   -- name shown to the raters
    questions_per_rater -- how many randomly chosen questions each rater gets
    seed                -- fixed seed, so re-running gives the same letters

    Returns the answer key.
    """
    with open(human_tests_path, encoding="utf-8") as f:
        human_tests = json.load(f)

    rng = random.Random(seed)
    site_questions, answer_key = [], {}

    for entry in human_tests["questions"]:
        model_keys = list(entry["answers"])
        rng.shuffle(model_keys)
        letters = string.ascii_lowercase[:len(model_keys)]

        site_questions.append({
            "id": entry["id"],
            "question": entry["question"],
            "answers": [{"id": letter, "text": entry["answers"][model_key]}
                        for letter, model_key in zip(letters, model_keys)],
        })
        answer_key[str(entry["id"])] = dict(zip(letters, model_keys))

    site_data = {
        "speaker_name": speaker_real_name,
        "questions_per_rater": min(questions_per_rater, len(site_questions)),
        "questions": site_questions,
    }

    for path, content in ((site_data_path, site_data), (answer_key_path, answer_key)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(site_questions)} questions to {site_data_path}, answer key to {answer_key_path}")
    return answer_key


if __name__ == "__main__":
    build_rating_site_data(
        human_tests_path="./output/human_tests/human_tests.json",
        site_data_path="./frontend_human_evaluation/public/questions.json",
        answer_key_path="./output/human_tests/answer_key.json",
        speaker_real_name="Ania",
        questions_per_rater=25,
    )
