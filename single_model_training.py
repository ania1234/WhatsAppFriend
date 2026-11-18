import os
from datetime import datetime

from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only


def train_model(base_model_name, dataset, eval_dataset, output_dir, target_modules, lora_r, lora_a, mask_prompt, lr_scheduler_type, lr_scheduler_kwargs, seed=2137):

    load_dotenv()

    # ========================================
    # CONFIG
    # ========================================

    TENSORBOARD_LOGGING_DIR = f"{output_dir}/tb_logs/{datetime.now():%Y%m%d-%H%M%S}"
    MAX_SEQ_LENGTH = 1024

    os.environ["TENSORBOARD_LOGGING_DIR"] = TENSORBOARD_LOGGING_DIR

    # ========================================
    # 1. LOAD MODEL IN 4-BIT
    # ========================================

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # Bielik powinien używać ChatML.
    probe = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Cześć"},
            {"role": "assistant", "content": "Hej!"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    assert "<|im_start|>user\n" in probe, probe
    assert "<|im_start|>assistant\n" in probe, probe

    # ========================================
    # 2. QLORA ADAPTER
    # ========================================

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=lora_a,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    # ========================================
    # 3. LOAD AND FORMAT DATASET
    # ========================================

    def format_conversations(batch):
        texts = [
            tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            for conversation in batch["conversations"]
        ]
        return {"text": texts}

    dataset = dataset.map(
        format_conversations,
        batched=True,
        remove_columns=dataset.column_names, #remove the old columns, only the "text" column is important
    )
    eval_dataset = eval_dataset.map(
        format_conversations,
        batched=True,
        remove_columns=eval_dataset.column_names, #remove the old columns, only the "text" column is important
    )

    # Avoid truncating assistant responses.
    def valid_length(example):
        ids = tokenizer(
            example["text"],
            add_special_tokens=False,
        )["input_ids"]
        return 0 < len(ids) <= MAX_SEQ_LENGTH

    original_count = len(dataset)
    dataset = dataset.filter(valid_length)

    print(f"Samples: {len(dataset)} / {original_count}")

    if len(dataset) < 100:
        raise ValueError("Too few valid training samples.")


    # ========================================
    # 4. TRAINING CONFIGURATION
    # ========================================

    config = SFTConfig(
        output_dir=output_dir,

        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False, #packing=True led to assistant responses loosing end-of-sequence token, which caused the model to generate garbage after the first sentence.

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,

        num_train_epochs=2,
        learning_rate=8e-5,

        optim="adamw_8bit",
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        warmup_ratio=0.05,
        weight_decay=0.01,

        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),

        logging_steps=10,
        eval_on_start=True,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="epoch",
        save_total_limit=2,

        assistant_only_loss=False, #would not work with the current data setup
        completion_only_loss=False, #would not work with the current data setup

        dataset_num_proc=1,
        report_to="tensorboard",
        seed=seed,
    )

    # ========================================
    # 5. CREATE TRAINER
    # ========================================

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=config,
    )

    if mask_prompt:
        # ========================================
        # 5a. PROMPT MASKING 
        # ========================================

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
            last_response_only=True,
        )

        # ========================================
        # 5b. VERIFY MASKING
        # ========================================

        batch = trainer.data_collator(
            [trainer.train_dataset[0]]
        )

        ids = batch["input_ids"][0].tolist()
        labels = batch["labels"][0].tolist()

        trained_ids = [
            token_id
            for token_id, label in zip(ids, labels)
            if label != -100
        ]

        assert len(trained_ids) > 0, "All tokens are masked!"

        print("\nTOKENS CONTRIBUTING TO LOSS:")
        print(tokenizer.decode(trained_ids))

    # ========================================
    # 6. TRAIN
    # ========================================

    trainer.train()

    # ========================================
    # 7. SAVE LORA ADAPTER
    # ========================================

    model.save_pretrained(output_dir + "/adapter")
    tokenizer.save_pretrained(output_dir + "/adapter")

    print("Training completed!")