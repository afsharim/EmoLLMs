import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import wandb

# Dataclasses for configurations
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for the input and output"},
    )


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "Path to the training data file"})
    validation_file: str = field(metadata={"help": "Path to the validation data file"})
    test_file: str = field(metadata={"help": "Path to the test data file"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Path to cache directory"})


@dataclass
class TrainingConfig:
    output_dir: str = field(metadata={"help": "Directory to save the fine-tuned model"})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Training batch size"})
    per_device_eval_batch_size: int = field(default=4, metadata={"help": "Evaluation batch size"})
    num_train_epochs: int = field(default=3, metadata={"help": "Number of training epochs"})
    learning_rate: float = field(default=1e-6, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=1e-5, metadata={"help": "Weight decay"})
    logging_steps: int = field(default=100, metadata={"help": "Logging frequency"})
    save_steps: int = field(default=500, metadata={"help": "Checkpoint save frequency"})
    save_total_limit: int = field(default=3, metadata={"help": "Total number of saved checkpoints"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Warmup ratio for learning rate scheduler"})
    gradient_accumulation_steps: int = field(default=2, metadata={"help": "Gradient accumulation steps"})


# Preprocessing function
def preprocess_function(examples, tokenizer, max_length):
    inputs = examples["input_text"]
    targets = examples["output_text"]
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        targets,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Compute metrics for evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(torch.tensor(predictions), dim=-1)
    labels = torch.tensor(labels)
    accuracy = (predictions == labels).float().mean().item()
    return {"accuracy": accuracy}


def main():
    # Initialize WandB
    wandb.init(project="T5-FineTuning", name="T5-Experiment")

    # Arguments
    model_args = ModelArguments(
        model_name_or_path="/research/hal-afsharim/llms/t5-base",
        max_seq_length=128,
    )
    data_args = DataArguments(
        train_file="../data/train.json",
        validation_file="../data/dev.json",
        test_file="../data/test.json",  # Path to the test dataset
        cache_dir="hf_cache_dir",
    )
    training_args = TrainingConfig(
        output_dir="saved_models/t5_finetuned",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,  # Reduced for OOM prevention
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=1e-6,
    )

    # Load the model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=data_args.cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, cache_dir=data_args.cache_dir)

    # Load the dataset
    train_dataset = load_dataset("json", data_files=data_args.train_file, cache_dir=data_args.cache_dir)["train"]
    val_dataset = load_dataset("json", data_files=data_args.validation_file, cache_dir=data_args.cache_dir)["train"]
    test_dataset = load_dataset("json", data_files=data_args.test_file, cache_dir=data_args.cache_dir)["train"]

    # Preprocess the data
    train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer, model_args.max_seq_length), batched=True)
    val_dataset = val_dataset.map(lambda x: preprocess_function(x, tokenizer, model_args.max_seq_length), batched=True)
    test_dataset = test_dataset.map(lambda x: preprocess_function(x, tokenizer, model_args.max_seq_length), batched=True)

    # Define TrainingArguments
    training_arguments = TrainingArguments(
        output_dir=training_args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=training_args.learning_rate,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        num_train_epochs=training_args.num_train_epochs,
        weight_decay=training_args.weight_decay,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        save_total_limit=training_args.save_total_limit,
        warmup_ratio=training_args.warmup_ratio,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        fp16=True,
        load_best_model_at_end=True,
        logging_dir=f"{training_args.output_dir}/logs",
        report_to=["wandb"],
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate on the validation set
    val_results = trainer.evaluate(eval_dataset=val_dataset)
    print("Validation Results:", val_results)

    # Evaluate on the test set
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Test Results:", test_results)

    # Save the final model
    trainer.save_model(training_args.output_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
