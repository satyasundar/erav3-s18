import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    pipeline,
    logging
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer, SFTConfig
import os
import shutil
import json

# Configure logging
logging.set_verbosity_info()
os.environ["WANDB_DISABLED"] = "true"

def prepare_dataset(tokenizer):
    # Load the OpenAssistant dataset
    dataset = load_dataset("OpenAssistant/oasst1")

    # Filter for only English conversations
    dataset = dataset.filter(lambda x: x['lang'] == 'en')

    # Function to format conversations
    def format_conversation(example):
        if example['role'] == 'assistant':
            return f"Assistant: {example['text']}\n"
        else:
            return f"Human: {example['text']}\n"

    # Process and format the dataset
    train_dataset = dataset['train']
    train_dataset = train_dataset.map(
        lambda x: {
            'text': format_conversation(x)
        }
    )

    # Add text preprocessing function with tokenization
    def preprocess_function(examples, tokenizer):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=2048,
            return_tensors=None
        )

    # Apply preprocessing with tokenizer
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    return train_dataset

def setup_model_and_tokenizer():
    # Configure quantization
    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Add this debug print
    # print("Available modules:")
    # for name, _ in model.named_modules():
    #     print(name)

    model.config.use_cache = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer

def setup_lora():
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    return lora_config

def main():
    # Setup output directory
    output_dir = "phi2-assistant"
    os.makedirs(output_dir, exist_ok=True)

    # Path to the uploaded checkpoint .zip (update this based on your upload)
    # checkpoint_zip_path = "/kaggle/input/phi2-checkpoints/checkpoint-5.zip"  # Adjust this
    # checkpoint_dir = f"/kaggle/working/{output_dir}/checkpoint-5"  # Extracted path

    checkpoint_dir = f"/kaggle/working/phi2-assistant/phi-ckpt-300"  # Extracted path
    # Define log file path
    log_file = "/kaggle/working/training_logs.jsonl"

    # Debug: Check checkpoint
    if os.path.exists(checkpoint_dir):
        print(f"Checkpoint found at {checkpoint_dir}")
        print("Files in checkpoint:", os.listdir(checkpoint_dir))
    else:
        print(f"No checkpoint found at {checkpoint_dir}; starting from scratch")
        checkpoint_dir = None

    # Check if log file exists from a previous run
    if os.path.exists(log_file):
        print(f"Existing log file found at {log_file}; will append to it")
    else:
        print(f"No log file found; creating new one at {log_file}")

    # Setup model and tokenizer first
    model, tokenizer = setup_model_and_tokenizer()

    # Then prepare dataset with tokenizer
    train_dataset = prepare_dataset(tokenizer)

    # Setup LoRA
    lora_config = setup_lora()

    # Get PEFT model
    model = get_peft_model(model, lora_config)

    # Setup comprehensive SFT configuration
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=False,
        resume_from_checkpoint=checkpoint_dir if checkpoint_dir else True,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False
    )

    # Custom callback to archive checkpoints
    class ArchiveCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
            archive_path = f"/kaggle/working/checkpoint-{state.global_step}.zip"
            # Create a zip archive of the checkpoint directory
            shutil.make_archive(archive_path.replace('.zip', ''), 'zip', checkpoint_dir)
            print(f"Archived checkpoint-{state.global_step} to {archive_path}")

    # Custom callback to save logs to file
    class PersistentLoggingCallback(TrainerCallback):
        def __init__(self, log_file):
            self.log_file = log_file

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                # Append log entry as a JSON line
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps({
                        "step": state.global_step,
                        "epoch": state.epoch,
                        "loss": logs.get("loss"),
                        "learning_rate": logs.get("learning_rate"),
                        "train_runtime": logs.get("train_runtime")
                    }) + "\n")
                print(f"Logged step {state.global_step} to {self.log_file}")


    # Initialize trainer with SFTConfig
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=sft_config,
        tokenizer=tokenizer
    )

    # Add the archiving callback
    trainer.add_callback(ArchiveCheckpointCallback())
    trainer.add_callback(PersistentLoggingCallback(log_file))

    # Explicitly resume training from the checkpoint
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        print(f"Attempting to resume from {checkpoint_dir}")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        print("Starting training from scratch")
        trainer.train()

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model(output_dir)

    # Archive the final model as well
    final_archive_path = "/kaggle/working/phi2-assistant-final.zip"
    shutil.make_archive(final_archive_path.replace('.zip', ''), 'zip', output_dir)
    print(f"Archived final model to {final_archive_path}")

if __name__ == "__main__":
    main()