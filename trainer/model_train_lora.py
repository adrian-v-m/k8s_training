import argparse
import json
import logging
import os
import torch
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from datasets.distributed import split_dataset_by_node
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from graceful_shutdown_callback import GracefulShutdownCallback

# Configure logger
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    log_formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)


def setup_model_and_tokenizer(model_name, use_local_files=False, add_special_tokens=False):
    """Set up the model and tokenizer for fine-tuning"""
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=use_local_files,
    )
    
    if add_special_tokens:
        special_tokens = {
            "pad_token": tokenizer.eos_token or "<PAD>",
            "additional_special_tokens": [
                "[INST]", "[/INST]", "<SYS>", "</SYS>", 
                "<USER>", "</USER>", "<ASSISTANT>", "</ASSISTANT>",
                "< EOT >", "<SEP>",
            ],
        }
        tokenizer.add_special_tokens(special_tokens)
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=use_local_files,
    )
    
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    
    return model, tokenizer


def load_dataset_from_json(dataset_path, split_ratio=0.1, seed=42):
    """Load dataset from a JSON file with already formatted text"""
    logger.info(f"Loading dataset from: {dataset_path}")
    
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.warning(f"Expected JSON data to be a list, got {type(data)}. Attempting to extract text field.")
            if isinstance(data, dict) and 'text' in data:
                data = data['text']
            elif all(isinstance(item, dict) and 'text' in item for item in data):
                data = [item['text'] for item in data]
        
        dataset = Dataset.from_dict({"text": data})
        splits = dataset.train_test_split(test_size=split_ratio, seed=seed)
        return splits['train'], splits['test']
    
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading JSON file: {e}")
        raise


def load_and_process_data(dataset_path, tokenizer=None, 
                         is_json=False, split_ratio=0.1, seed=42, block_size=1024):
    """Load and process dataset for training"""
    logger.info("Loading and processing dataset")
    
    # Load dataset
    if is_json:
        train_data, eval_data = load_dataset_from_json(dataset_path, split_ratio, seed)
    else:
        try:
            dataset = load_from_disk(dataset_path)
            
            # Extract train dataset
            train_data = dataset["train"] if "train" in dataset else dataset
            
            # Always create a validation split from the training data
            splits = train_data.train_test_split(test_size=split_ratio, seed=seed)
            train_data = splits["train"]
            eval_data = splits["test"]
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    # Process dataset if tokenizer provided
    if tokenizer:
        train_data, eval_data = tokenize_and_process_dataset(
            train_data, eval_data, tokenizer, block_size
        )
    
    # Handle distributed training
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        RANK, WORLD_SIZE = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        logger.info(f"Distributing dataset: WORLD_SIZE={WORLD_SIZE}, RANK={RANK}")
        
        if isinstance(train_data, Dataset):
            train_data = split_dataset_by_node(train_data, rank=RANK, world_size=WORLD_SIZE)
        if isinstance(eval_data, Dataset):
            eval_data = split_dataset_by_node(eval_data, rank=RANK, world_size=WORLD_SIZE)
    
    return train_data, eval_data


def tokenize_and_process_dataset(train_data, eval_data, tokenizer, block_size):
    """Tokenize and prepare dataset for training"""
    logger.info("Tokenizing and processing dataset")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=False, truncation=True)
    
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    # Process training data
    train_data = train_data.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    train_data = train_data.map(
        group_texts,
        batched=True,
        batch_size=1000,
        remove_columns=["input_ids", "attention_mask"],
    )
    
    # Process evaluation data if provided
    if eval_data:
        eval_data = eval_data.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )
        
        eval_data = eval_data.map(
            group_texts,
            batched=True,
            batch_size=1000,
            remove_columns=["input_ids", "attention_mask"],
        )
    
    return train_data, eval_data


def setup_and_apply_lora(model, config_dict=None, r=8):
    """Set up LoRA configuration and apply to model"""
    logger.info("Setting up and applying LoRA")
    
    # Default LoRA configuration
    default_config = {
        "r": r,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    
    # Update config from provided dictionary
    if config_dict:
        if isinstance(config_dict, str):
            try:
                config_dict = json.loads(config_dict)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing LoRA config JSON: {e}")
                raise
        default_config.update(config_dict)
    
    # Create and apply LoRA config
    lora_config = LoraConfig(**default_config)
    logger.info(f"LoRA config: {lora_config}")
    
    # Apply LoRA
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    
    # Log trainable parameters
    trainable_params, all_params = 0, 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(f"Trainable params: {trainable_params} ({100 * trainable_params / all_params:.2f}% of total)")
    return model


def get_training_args(args):
    """Create training arguments from CLI args and defaults"""
    logger.info("Setting up training arguments")
    
    # Base training args from CLI
    training_args_dict = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio,
    }
    
    # Default training arguments for settings not in CLI
    defaults = {
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "optim": "adamw_torch",
        "max_grad_norm": 0.3,
        "max_steps": 500,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": True,
        "lr_scheduler_type": "cosine",
    }
    
    # Apply defaults (if not already set)
    for k, v in defaults.items():
        if k not in training_args_dict:
            training_args_dict[k] = v
    
    # Override with training parameters if provided
    if args.training_parameters:
        try:
            training_params = json.loads(args.training_parameters)
            training_args_dict.update(training_params)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing training parameters JSON: {e}")
            raise
    
    training_args = TrainingArguments(**training_args_dict)
    return training_args


def train_model(model, train_data, eval_data, tokenizer, training_args,
               packing=False, max_seq_length=1024):
    """Train model using SFT Trainer from TRL"""
    logger.info("Setting up SFT Trainer and starting training")
    
    # Data collator for masking prompts
    data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template="[INST]",
        response_template="[/INST]",
        tokenizer=tokenizer,
    )
    
    # Callbacks
    callbacks = [EarlyStoppingCallback(early_stopping_patience=6)]
    
    # Add Kubernetes callbacks if running in K8s
    if "RANK" in os.environ and int(os.environ["RANK"]) == 0:
        # callbacks.append(ConfigMapUpdateCallback("results")) # Temporarily commented out
        callbacks.append(GracefulShutdownCallback())
    
    # Create and run SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataset_text_field="input_ids",
        packing=packing,
        max_seq_length=max_seq_length,
        callbacks=callbacks,
    )
    
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # Save model and tokenizer
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    return trainer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA")
    
    # Model and data arguments
    parser.add_argument("--model_name", required=True, help="Model name or path")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset")
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    parser.add_argument("--is_json", action="store_true", help="Dataset is a JSON file")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension")
    parser.add_argument("--lora_config", help="JSON string with LoRA configuration")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--local_files_only", action="store_true", help="Use local files only")
    parser.add_argument("--add_special_tokens", action="store_true", help="Add special tokens to tokenizer")
    parser.add_argument("--training_parameters", help="JSON string with training parameters")
    parser.add_argument("--block_size", type=int, default=1024, help="Block size for sequence grouping")
    parser.add_argument("--packing", action="store_true", help="Pack sequences (with SFT Trainer)")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Set up model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        use_local_files=args.local_files_only,
        add_special_tokens=args.add_special_tokens,
    )
    
    # Load and process data
    train_data, eval_data = load_and_process_data(
        args.dataset_path,
        tokenizer=tokenizer,
        is_json=args.is_json,
        block_size=args.block_size,
    )
    
    # Set up and apply LoRA
    model = setup_and_apply_lora(model, args.lora_config, args.lora_r)
    
    # Get training arguments
    training_args = get_training_args(args)
    
    # Train model
    trainer = train_model(
        model=model,
        train_data=train_data,
        eval_data=eval_data,
        tokenizer=tokenizer,
        training_args=training_args,
        packing=args.packing,
        max_seq_length=args.block_size,
    )
    
    logger.info("Training complete!")


if __name__ == "__main__":
    logger.info("Starting LoRA fine-tuning")
    main() 