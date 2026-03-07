import os
from pathlib import Path
import io
import pandas as pd
import numpy as np
import librosa
import torch
import evaluate
from torch.utils.data import Dataset, DataLoader
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
try:
    from transformers import DataCollatorSpeechSeq2SeqWithPadding
except ImportError:
    DataCollatorSpeechSeq2SeqWithPadding = None
try:
    from clearml import Task, Dataset as ClearMLDataset
except Exception:
    Task = None
    ClearMLDataset = None
import warnings
warnings.filterwarnings('ignore')

try:
    from src.config import TRAIN_PARQUET, TEST_PARQUET
except ModuleNotFoundError:
    from config import TRAIN_PARQUET, TEST_PARQUET
MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-small")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "whisper-finetuned")
LANGUAGE = os.getenv("WHISPER_LANGUAGE", "English")
TASK = os.getenv("WHISPER_TASK", "transcribe")
CLEARML_PROJECT = "NESTLAB Speech"
CLEARML_TASK_NAME = "whisper-finetuning"


class SpeechDataset(Dataset):
    def __init__(self, parquet_path, processor, feature_extractor, tokenizer):
        self.df = pd.read_parquet(parquet_path)
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_bytes = row['audio']['bytes']
        text = row['text']
        
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        input_features = self.feature_extractor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features[0]
        
        labels = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True
        ).input_ids[0]
        
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_features": input_features,
            "labels": labels
        }


class SpeechDataCollator:
    def __init__(self, processor, feature_extractor, tokenizer):
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        input_features = [f["input_features"] for f in features]
        labels = [f["labels"] for f in features]
        
        batch_input_features = torch.stack(input_features)
        
        batch_labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=8
        )["input_ids"]
        
        batch_labels[batch_labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_features": batch_input_features,
            "labels": batch_labels
        }


def load_processor(model_id):
    processor = WhisperProcessor.from_pretrained(model_id, language=LANGUAGE, task=TASK)
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    return processor, feature_extractor, tokenizer


def prepare_dataloaders(train_path, test_path, processor, feature_extractor, tokenizer, batch_size=4):
    train_dataset = SpeechDataset(train_path, processor, feature_extractor, tokenizer)
    test_dataset = SpeechDataset(test_path, processor, feature_extractor, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dataset, test_dataset, train_loader, test_loader


def setup_model(model_id):
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    return model


def train(
    model_id=MODEL_ID,
    output_dir=OUTPUT_DIR,
    train_path=None,
    test_path=None,
    num_epochs=3,
    batch_size=4,
    learning_rate=1e-5,
    warmup_steps=500,
    eval_steps=500,
    save_steps=500,
    logging_steps=25,
    use_clearml=True,
    clearml_dataset_id=None
):
    print(f"Starting training with model: {model_id}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if use_clearml and Task is None:
        raise RuntimeError("ClearML is enabled but 'clearml' package is not installed")

    if use_clearml:
        task = Task.init(
            project_name=CLEARML_PROJECT,
            task_name=CLEARML_TASK_NAME,
            output_uri=f"file://{os.path.abspath(output_dir)}"
        )
        logger = task.get_logger()
        task.connect({
            "model_id": model_id,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        })
    else:
        logger = None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading processor...")
    processor, feature_extractor, tokenizer = load_processor(model_id)
    
    print("Loading datasets...")
    train_data_path = Path(train_path) if train_path else TRAIN_PARQUET
    test_data_path = Path(test_path) if test_path else TEST_PARQUET
    train_dataset = SpeechDataset(train_data_path, processor, feature_extractor, tokenizer)
    test_dataset = SpeechDataset(test_data_path, processor, feature_extractor, tokenizer)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    print("Setting up model...")
    model = setup_model(model_id)
    model = model.to(device)
    
    if DataCollatorSpeechSeq2SeqWithPadding:
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt"
        )
    else:
        data_collator = SpeechDataCollator(processor, feature_extractor, tokenizer)
    
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="clearml" if use_clearml else "none"
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    if use_clearml:
        task.close()
    
    print(f"Training complete! Model saved to {output_dir}")
    return model, processor


def upload_dataset_to_clearml():
    if Task is None or ClearMLDataset is None:
        raise RuntimeError("ClearML package is not installed")

    task = Task.init(project_name=CLEARML_PROJECT, task_name="dataset-upload")
    
    dataset = ClearMLDataset.create(
        dataset_name="whisper-training-data",
        dataset_project=CLEARML_PROJECT
    )
    
    dataset.add_files(str(TRAIN_PARQUET.parent))
    dataset.upload()
    dataset.finalize()
    
    task.close()
    print(f"Dataset uploaded. ID: {dataset.id}")
    return dataset.id


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "upload-dataset"], default="train")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--train-path")
    parser.add_argument("--test-path")
    parser.add_argument("--no-clearml", action="store_true")
    args = parser.parse_args()
    
    if args.mode == "upload-dataset":
        upload_dataset_to_clearml()
    else:
        train(
            model_id=args.model_id,
            output_dir=args.output_dir,
            train_path=args.train_path,
            test_path=args.test_path,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            use_clearml=not args.no_clearml
        )



