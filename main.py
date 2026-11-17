# %%imports
import os
from dataclasses import dataclass

from datasets import load_dataset
from dotenv import load_dotenv

from evaluate_models import run_automatic_evaluations
from generate_human_tests import generate_human_tests
from import_human_feedback import import_human_feedback
from parsing.chat_analytics import analyze_files
from parsing.chat_to_json import (
    EXTENDED_PROMPT_CHAT,
    SYSTEM_PROMPT_TRAINING,
    convert_chat_to_training_data,
)
from parsing.json_cleanup import clean_up_json_file, merge_json_files
from parsing.synthetic_data_generation import generate_synthetic_chat_data
from single_model_training import train_model

# ========================================
# SHARED CONFIG
# ========================================

load_dotenv()  # CHAT_EXAMPLES for the extended baseline prompt

SEED = 2137

CHAT_FILES = ["./data/asia_chat_1.txt", "./data/asia_chat_2.txt", "./data/asia_chat_3.txt"]
CHAT_DATASET_PATH = "./data/final_chat_ania.json"
SYNTHETIC_MERGED_PATH = "./data/synthetic_chat_ania_merged.json"
TEST_SPLIT_DIR = "./data/test_split"
HUMAN_TESTS_PATH = "./output/human_tests/human_tests.json"
HUMAN_TESTS_ANSWER_KEY_PATH = "./output/human_tests/answer_key.json"  # which anonymised answer is which model
MODEL_EVALUATION_DIR = "./output/statistics/model_evaluation"  # automatic statistics and plots
HUMAN_FEEDBACK_RAW_DIR = "./output/human_feedback/raw"  # submissions as downloaded from Vercel Blob
HUMAN_FEEDBACK_PATH = "./output/human_feedback/ratings.json"  # all ratings, decoded to model names

SPEAKER_REAL_NAME = "Ania"
SPEAKER_CHARACTERISTICS = "35 letnią programistką, rozmawiającą ze swoją koleżanką"
# speaker profile for the extended (non-finetuned) baseline prompt in human tests
SPEAKER_AGE = 35
SPEAKER_PLACE = "Warszawa"
SPEAKER_OCCUPATION = "Programistka Gier Unity"
SPEAKER_ADDITIONAL_INFORMATION = "Mieszka z Piotrem i Golfikiem, podczas rozmów czesto używa emotki XD"
START_CONVERSATION_PROMPTS = ["Hej, co tam?", "Siema!", "Co słychać?"]

# Tokens excluded only from the top-word ranking in chat analytics.
POLISH_STOPWORDS = ["to", "w", "i", "się", "na", "z", "ale", "a", "że", "no", "jak", "do",
                    "jest", "o", "bo", "nie", "tak", "co", "po", "za", "ja", "image", "mi"]

SYNTHETIC_WINDOW_LENGTHS = (2, 4, 6)

TEST_SIZE = 25  # held out for evaluation after training
EVAL_SIZE = 25  # used for eval loss during training

BASE_MODEL = "speakleash/Bielik-4.5B-v3.0-Instruct"
EXTENDED_MODEL = "speakleash/Bielik-Minitron-7B-v3.0-Instruct"  # prompted-only baseline for human tests
LORA_R = 4
LORA_ALPHA = 8
QK_MODULES = ["q_proj", "k_proj"]
QKVO_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class Experiment:
    name: str
    dataset: str  # key into the splits dict: "chat" or "synthetic"
    target_modules: list[str]
    mask_prompt: bool


EXPERIMENTS = [
    Experiment("full_chat_dump_model",           "chat",      QKVO_MODULES, mask_prompt=False),
    Experiment("masked_chat_dump_q_k_model",     "chat",      QK_MODULES,   mask_prompt=True),
    Experiment("masked_chat_dump_q_k_v_o_model", "chat",      QKVO_MODULES, mask_prompt=True),
    Experiment("synthetic_chat_model",           "synthetic", QKVO_MODULES, mask_prompt=True),
]


# %%#convert initial chat logs to json format
convert_chat_to_training_data(
    file_names=CHAT_FILES,
    speaker_names=["anierhej"],
    start_conversation_prompts=START_CONVERSATION_PROMPTS,
    speaker_real_name=SPEAKER_REAL_NAME,
    speaker_characteristics=SPEAKER_CHARACTERISTICS,
    seed=SEED,
    output_path=CHAT_DATASET_PATH,
)

# %%#run initial chat analytics
analyze_files(
    CHAT_FILES,
    speaker_1_names=["Asia"],  # exact aliases combined under speaker_1
    speaker_2_names=["anierhej"],  # exact aliases combined under speaker_2
    exclude=POLISH_STOPWORDS,
    top_n=10,
    turn_gap_minutes=120.0,
    json_path="./output/statistics/chat_analytics.json",  # complete machine-readable statistics
    csv_path="./output/statistics/chat_analytics.csv",  # per-speaker summary (UTF-8 BOM)
    plots_dir="./output/plots",  # PNG plots (requires matplotlib)
)

# %%#generate synthetic chat data
for rolling_window_length in SYNTHETIC_WINDOW_LENGTHS:
    generate_synthetic_chat_data(
        chat_files=CHAT_FILES,
        rolling_window_length=rolling_window_length,
        speaker_names=["Asia", "anierhej"],
        speaker_real_name=SPEAKER_REAL_NAME,
        speaker_characteristics=SPEAKER_CHARACTERISTICS,
        output_path=f"./data/synthetic_chat_{rolling_window_length}.json",
        seed=SEED,
    )
    cleaned_output_path = f"./data/synthetic_chat_{rolling_window_length}_cleaned.json"
    #clean the generated synthetic data
    clean_up_json_file(f"./data/synthetic_chat_{rolling_window_length}.json", cleaned_output_path)

merge_json_files(
    [f"./data/synthetic_chat_{rolling_window_length}_cleaned.json" for rolling_window_length in SYNTHETIC_WINDOW_LENGTHS],
    SYNTHETIC_MERGED_PATH,
)

# %%#per dataset: hold out TEST_SIZE samples for evaluation after training, EVAL_SIZE samples for eval loss during training, rest is training data
chat_dataset = load_dataset("json", data_files=CHAT_DATASET_PATH, split="train")
synthetic_dataset = load_dataset("json", data_files=SYNTHETIC_MERGED_PATH, split="train")

chat_split = chat_dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
synthetic_split = synthetic_dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)

chat_train, chat_test = chat_split["train"], chat_split["test"]
synthetic_train, synthetic_test = synthetic_split["train"], synthetic_split["test"]

#carve out evaluation splits (used for eval loss during training) from the remaining training data
chat_eval_split = chat_train.train_test_split(test_size=EVAL_SIZE, seed=SEED)
synthetic_eval_split = synthetic_train.train_test_split(test_size=EVAL_SIZE, seed=SEED)

chat_train, chat_eval = chat_eval_split["train"], chat_eval_split["test"]
synthetic_train, synthetic_eval = synthetic_eval_split["train"], synthetic_eval_split["test"]

#save held-out samples for later evaluation
os.makedirs(TEST_SPLIT_DIR, exist_ok=True)
chat_test.to_json(f"{TEST_SPLIT_DIR}/chat_test.json", force_ascii=False)
synthetic_test.to_json(f"{TEST_SPLIT_DIR}/synthetic_test.json", force_ascii=False)

# %%#train all experiments
splits = {
    "chat": (chat_train, chat_eval),
    "synthetic": (synthetic_train, synthetic_eval),
}
for experiment in EXPERIMENTS:
    train_dataset, eval_dataset = splits[experiment.dataset]
    train_model(
        base_model_name=BASE_MODEL,
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=f"./output/{experiment.name}",
        target_modules=experiment.target_modules,
        lora_r=LORA_R,
        lora_a=LORA_ALPHA,
        mask_prompt=experiment.mask_prompt,
        seed=SEED,
    )

# %%#generate dataset for collecting user feedback
HUMAN_TESTS_BASELINE_PROMPT = EXTENDED_PROMPT_CHAT.format(name=SPEAKER_REAL_NAME,
                                                          age=SPEAKER_AGE,
                                                          place=SPEAKER_PLACE,
                                                          occupation=SPEAKER_OCCUPATION,
                                                          additional_information=SPEAKER_ADDITIONAL_INFORMATION,
                                                          examples=os.environ["CHAT_EXAMPLES"])
HUMAN_TESTS_ADAPTER_PROMPT = SYSTEM_PROMPT_TRAINING.format(name=SPEAKER_REAL_NAME,
                                                      characteristics=SPEAKER_CHARACTERISTICS)
generate_human_tests(
    models=[(BASE_MODEL, None, HUMAN_TESTS_BASELINE_PROMPT)]
    + [(BASE_MODEL, f"./output/{experiment.name}/adapter", HUMAN_TESTS_ADAPTER_PROMPT)
       for experiment in EXPERIMENTS],
    output_path=HUMAN_TESTS_PATH,
    seed=SEED,
)

# %%#run automatic evaluations
run_automatic_evaluations(
    base_model_name=BASE_MODEL,
    experiments=[
        {"name": experiment.name, "adapter": f"./output/{experiment.name}/adapter",
         "dataset": experiment.dataset, "mask_prompt": experiment.mask_prompt}
        for experiment in EXPERIMENTS
    ],
    test_sets={"chat": chat_test, "synthetic": synthetic_test},
    train_sets={"chat": chat_train, "synthetic": synthetic_train},
    output_dir=MODEL_EVALUATION_DIR,  # results.json, summary.md/csv/json and plots/
    seed=SEED,
)

# %%#import user feedback
import_human_feedback(
    raw_dir=HUMAN_FEEDBACK_RAW_DIR,
    answer_key_path=HUMAN_TESTS_ANSWER_KEY_PATH,
    output_path=HUMAN_FEEDBACK_PATH,
)

# %%#run user input evaluations
