import torch
from datasets import load_dataset, Dataset
from transformers import LongformerForSequenceClassification, LongformerTokenizer, Trainer, TrainingArguments
import evaluate

_dataset = None
_model_tokenizer = None

def get_dataset():
    global _dataset
    if _dataset is None:
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
        def split_into_sentences(example):
            sentences = example["article"].split(". ")
            labels = [1 if i < 2 else 0 for i in range(len(sentences))]
            return {"sentences": sentences, "labels": labels}
        dataset = dataset.map(split_into_sentences, batched=False)
        flattened = {"sentence": [], "label": []}
        for example in dataset:
            flattened["sentence"].extend(example["sentences"])
            flattened["label"].extend(example["labels"])
        _dataset = Dataset.from_dict(flattened)
    return _dataset

def get_model_and_tokenizer():
    global _model_tokenizer
    if _model_tokenizer is None:
        model_name = "allenai/longformer-base-4096"
        tokenizer = LongformerTokenizer.from_pretrained(model_name)
        model = LongformerForSequenceClassification.from_pretrained(model_name, num_labels=2)
        _model_tokenizer = (model, tokenizer)
    return _model_tokenizer

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=512)

def compute_rouge_extractive(model, tokenizer, flat_dataset, num_examples=20):
    rouge = evaluate.load("rouge")
    predictions = []
    references = []
    # Here, for simplicity, we assume each sample is one sentence.
    for example in flat_dataset.select(range(num_examples)):
        inputs = tokenizer(example["sentence"], return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        outputs = model(**inputs)
        score = torch.softmax(outputs.logits, dim=-1)[0][1].item()
        pred_summary = example["sentence"] if score > 0.5 else ""
        predictions.append(pred_summary)
        ref_summary = example["sentence"] if example["label"] == 1 else ""
        references.append(ref_summary)
    scores = rouge.compute(predictions=predictions, references=references)
    return scores

def main():
    flat_dataset = get_dataset()
    model, tokenizer = get_model_and_tokenizer()

    flat_dataset = flat_dataset.map(lambda ex: preprocess_function(ex, tokenizer), batched=True)
    split_dataset = flat_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    training_args = TrainingArguments(
        output_dir="./longformer_extractive_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    model.save_pretrained("./longformer_extractive_model")
    tokenizer.save_pretrained("./longformer_extractive_model")

    scores = compute_rouge_extractive(model, tokenizer, val_dataset)
    print("Extractive Summarization ROUGE Scores:")
    print(f"ROUGE-1: {scores['rouge1']:.4f}")
    print(f"ROUGE-2: {scores['rouge2']:.4f}")
    print(f"ROUGE-L: {scores['rougeL']:.4f}")
    
if __name__ == "__main__":
    main()
