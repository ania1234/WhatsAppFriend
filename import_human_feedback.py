"""
Download the ratings sent from the rating website (Vercel Blob) and decode the
anonymised answer letters back to model names.

Needs BLOB_READ_WRITE_TOKEN in .env (Vercel dashboard -> Storage -> the Blob store -> .env.local tab).

Writes:
    raw_dir/*.json  -- the submissions exactly as stored in Blob, one file per submission
    output_path     -- every rating from every submission, decoded:
        {"submissions": 3,
         "ratings": [{"rater": "Asia",
                      "submission": "2026-11-10T12-00-00-000Z-asia-AbC123.json",
                      "received_at": "2026-11-10T12:00:00.000Z",
                      "question_id": 31,
                      "best": "masked_chat_dump_q_k_model",
                      "worst": "base",
                      "passes": false,
                      "order_shown": ["masked_chat_dump_q_k_model", "base", ...]}, ...]}

Run from the repository root:
    uv run python import_human_feedback.py
"""

import json
import os

from dotenv import load_dotenv


def download_ratings(raw_dir, blob_prefix="ratings/", blob_access="private"):
    """Download every submission under blob_prefix into raw_dir. Returns the number of new files."""
    from vercel.blob import sync as blob

    load_dotenv()
    os.makedirs(raw_dir, exist_ok=True)

    downloaded = 0
    for item in blob.query_blobs(prefix=blob_prefix):
        target = os.path.join(raw_dir, os.path.basename(item.pathname))
        # Submissions are never overwritten in Blob, so a file that is already here is complete.
        if os.path.exists(target):
            continue
        result = blob.get_blob(item.pathname, access=blob_access)
        with open(target, "wb") as f:
            f.write(result.body)
        downloaded += 1

    print(f"Downloaded {downloaded} new submissions to {raw_dir}")
    return downloaded


def decode_ratings(raw_dir, answer_key_path, output_path):
    """Replace the answer letters in every submission in raw_dir with model names."""
    with open(answer_key_path, encoding="utf-8") as f:
        answer_key = json.load(f)

    file_names = sorted(name for name in os.listdir(raw_dir) if name.endswith(".json"))
    ratings, skipped = [], 0

    for file_name in file_names:
        with open(os.path.join(raw_dir, file_name), encoding="utf-8") as f:
            submission = json.load(f)

        for rating in submission["ratings"]:
            letters = answer_key.get(str(rating["question_id"]), {})
            used = [rating["best"], rating["worst"], *rating["order_shown"]]
            # A rating that does not fit the answer key was made against other site data.
            if not letters or any(letter not in letters for letter in used):
                skipped += 1
                continue
            ratings.append({
                "rater": submission["rater"],
                "submission": file_name,
                "received_at": submission.get("received_at"),
                "question_id": rating["question_id"],
                "best": letters[rating["best"]],
                "worst": letters[rating["worst"]],
                "passes": rating["passes"],
                "order_shown": [letters[letter] for letter in rating["order_shown"]],
            })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"submissions": len(file_names), "ratings": ratings}, f, ensure_ascii=False, indent=2)

    raters = sorted({rating["rater"] for rating in ratings})
    print(f"Decoded {len(ratings)} ratings from {len(file_names)} submissions "
          f"({len(raters)} raters: {', '.join(raters)}) to {output_path}")
    if skipped:
        print(f"WARNING: skipped {skipped} ratings that do not match {answer_key_path}")
    return ratings


def import_human_feedback(raw_dir, answer_key_path, output_path, blob_prefix="ratings/", blob_access="private"):
    """
    raw_dir         -- local directory for the downloaded submissions
    answer_key_path -- JSON written by build_rating_site_data (answer letter -> model)
    output_path     -- JSON file with all decoded ratings
    blob_prefix     -- folder in the Blob store the website writes to
    blob_access     -- "private" or "public", matching how the Blob store was created

    Returns the list of decoded ratings.
    """
    download_ratings(raw_dir, blob_prefix, blob_access)
    return decode_ratings(raw_dir, answer_key_path, output_path)


if __name__ == "__main__":
    import_human_feedback(
        raw_dir="./output/human_feedback/raw",
        answer_key_path="./output/human_tests/answer_key.json",
        output_path="./output/human_feedback/ratings.json",
    )
