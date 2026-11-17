"""
Generate answers to a fixed list of conversation starters with the base model
and every trained LoRA adapter, and save them to one JSON file for human rating.

Output format:
    {
      "generation": {"max_new_tokens": ..., "temperature": ..., "top_p": ..., "seed": ...},
      "models": {"Bielik-4.5B-v3.0-Instruct": {"model": "speakleash/Bielik-4.5B-v3.0-Instruct", "adapter": null,
                                               "system_prompt": "..."},
                 "full_chat_dump_model": {"model": "speakleash/...", "adapter": "./output/.../adapter",
                                          "system_prompt": "..."}, ...},
      "questions": [
        {"id": 1,
         "question": "Jak się czujesz?",
         "answers": {"Bielik-4.5B-v3.0-Instruct": "...", "full_chat_dump_model": "...", ...}},
        ...
      ]
    }

Run from the repository root:
    uv run python generate_human_tests.py
"""

import gc
import json  # noqa: I001
import os

# Unsloth must be imported before transformers/torch to apply its patches.
from dotenv import load_dotenv
from unsloth import FastLanguageModel
import torch
from peft import PeftModel

from parsing.chat_to_json import EXTENDED_PROMPT_CHAT, SYSTEM_PROMPT_TRAINING

# ========================================
# CONFIG
# ========================================

MAX_SEQ_LENGTH = 8192
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9
BATCH_SIZE = 4

QUESTIONS = [
    "Jak się czujesz?",
    "Co dzisiaj robisz?",
    "Ile to jest 2+2?",
    "Jak działa quicksort?",
    "Co u Piotra?",
    "Co u Golfika?",
    "Co nowego w świecie tetrisa?",
    "Lubisz frytki?",
    "Zrób mi loda.",
    "Ile jest podstawowych kolorów?",
    "Jaka jest stolica Macedonii?",
    "Wymień wszystkich prezydentów USA.",
    "Kiedy ostatnio rzygałaś?",
    "Czy podoba ci się twórczośc Jerzego?",
    "Lubisz pizzę?",
    "Co się działo w piątek?",
    "Co to jest DFS?",
    "Hej, śpisz?",
    "Idziemy dziś na karaoke?",
    "Co jadłaś na obiad?",
    "Jakie plany na weekend?",
    "O której jutro wstajesz?",
    "Oglądałaś coś ostatnio fajnego?",
    "Gdzie jedziesz na wakacje?",
    "Pamiętasz, co robiliśmy w sylwestra?",
    "Jak tam w pracy?",
    "Co u twojej mamy?",
    "Kiedy się w końcu widzimy?",
    "Kto to jest Wojtek?",
    "Mam dzisiaj fatalny dzień.",
    "Dostałam tę pracę!!!",
    "Jestem na ciebie zła.",
    "Nudzi mi się, powiedz coś.",
    "Przepraszam za wczoraj.",
    "Psy czy koty?",
    "Ananas na pizzy, tak czy nie?",
    "Co myślisz o polityce?",
    "Jaki jest najlepszy język programowania?",
    "Napisz mi wiersz o jesieni.",
    "Wytłumacz mi, co to jest LoRA.",
    "Przetłumacz na angielski: „nie chce mi się”.",
    "Napisz funkcję w Pythonie, która odwraca stringa.",
    "Jak ugotować jajko na miękko?",
    "Jesteś botem?",
    "Udowodnij, że jesteś człowiekiem.",
    "Jesteś głupia.",
    "Opowiedz mi żart.",
    "What's up?",
    "Tolerujesz laktozę?"
]


def model_key(model_name, adapter):
    """Name of a (model, adapter) pair in the output JSON: the model's name or the adapter's experiment name."""
    if adapter is None:
        # "speakleash/Bielik-4.5B-v3.0-Instruct" -> "Bielik-4.5B-v3.0-Instruct"
        return model_name.split("/")[-1]
    # "./output/full_chat_dump_model/adapter" -> "full_chat_dump_model"
    path = os.path.normpath(adapter)
    name = os.path.basename(path)
    return os.path.basename(os.path.dirname(path)) if name == "adapter" else name


@torch.no_grad()
def generate_answers(model_name, adapter, system_prompt, questions, seed):
    """Load the base model, apply the adapter on top (unless None) and answer every question."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
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

    # Same seed for every model, so the differences come from the weights, not the sampling.
    torch.manual_seed(seed)

    answers = []
    for start in range(0, len(questions), BATCH_SIZE):
        batch = questions[start:start + BATCH_SIZE]
        texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for question in batch
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        answers += [text.strip() for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
        print(f"  {min(start + BATCH_SIZE, len(questions))}/{len(questions)} answered")

    # Free the GPU before the next model is loaded.
    del model, tokenizer, inputs, output
    gc.collect()
    torch.cuda.empty_cache()

    return answers


def generate_human_tests(models, output_path, questions=QUESTIONS, seed=2137):
    """
    models      -- list of (Hugging Face base model name, adapter directory or None, system prompt)
                   triples; an entry without an adapter is saved under the model's name, the
                   others under their experiment name (see model_key)
    output_path -- JSON file to write
    questions   -- conversation starters; ids are their 1-based positions

    Returns the dict that was written to output_path.
    """
    keys = [model_key(model_name, adapter) for model_name, adapter, _ in models]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Duplicate model keys: {duplicates}")

    result = {
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": seed,
        },
        "models": {
            key: {"model": model_name, "adapter": adapter, "system_prompt": system_prompt}
            for key, (model_name, adapter, system_prompt) in zip(keys, models)
        },
        "questions": [
            {"id": question_id, "question": question, "answers": {}}
            for question_id, question in enumerate(questions, start=1)
        ],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    for key, (model_name, adapter, system_prompt) in zip(keys, models):
        print(f"Generating answers: {key} ({model_name} + {adapter})")
        answers = generate_answers(model_name, adapter, system_prompt, questions, seed)
        for entry, answer in zip(result["questions"], answers):
            entry["answers"][key] = answer

        # Saved after every model, so a crash does not lose the finished ones.
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(questions)} questions x {len(models)} models to {output_path}")
    return result


if __name__ == "__main__":
    load_dotenv()
    base_model_name = "speakleash/Bielik-4.5B-v3.0-Instruct"
    extended_model_name = "speakleash/Bielik-Minitron-7B-v3.0-Instruct"
    characteristics = "35 letnią programistką, rozmawiającą ze swoją koleżanką"
    base_prompt = EXTENDED_PROMPT_CHAT.format(name="Ania",
                    age=35,
                    place="Warszawa",
                    occupation="Programistka Gier Unity",
                    additional_information="Mieszka z Piotrem i Golfikiem, podczas rozmów czesto używa emotki XD",
                    examples=os.environ["CHAT_EXAMPLES"])
    adapter_prompt = SYSTEM_PROMPT_TRAINING.format(name="Ania", characteristics=characteristics)
    generate_human_tests(
        models=[
            (extended_model_name, None, base_prompt),
            #(base_model_name, None, base_prompt),
            (base_model_name, "./output/full_chat_dump_model/adapter", adapter_prompt),
            (base_model_name, "./output/masked_chat_dump_q_k_model/adapter", adapter_prompt),
            (base_model_name, "./output/masked_chat_dump_q_k_v_o_model/adapter", adapter_prompt),
            (base_model_name, "./output/synthetic_chat_model/adapter", adapter_prompt),
        ],
        output_path="./output/human_tests/human_tests.json",
    )
