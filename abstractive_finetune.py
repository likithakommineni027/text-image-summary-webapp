from datasets import load_dataset
from transformers import LEDForConditionalGeneration, LEDTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
import evaluate

_dataset = None
_model_tokenizer = None

def get_datasets():
    global _dataset
    if _dataset is None:
        dataset = load_dataset("cnn_dailymail", "3.0.0")
        train_dataset = dataset["train"].select(range(1000))
        val_dataset = dataset["validation"].select(range(100))
        _dataset = (train_dataset, val_dataset)
    return _dataset

def get_model_and_tokenizer():
    global _model_tokenizer
    if _model_tokenizer is None:
        model_name = "allenai/led-base-16384"
        tokenizer = LEDTokenizer.from_pretrained(model_name)
        model = LEDForConditionalGeneration.from_pretrained(model_name)
        _model_tokenizer = (model, tokenizer)
    return _model_tokenizer

def preprocess_function(examples, tokenizer):
    inputs = examples["article"]
    targets = examples["highlights"]
    model_inputs = tokenizer(inputs, max_length=4096, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=256, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_rouge_scores(model, tokenizer, val_dataset, num_examples=20):
    rouge = evaluate.load("rouge")
    predictions = []
    references = []
    for example in val_dataset.select(range(num_examples)):
        inputs = tokenizer.encode(example["article"], return_tensors="pt", max_length=4096, truncation=True)
        summary_ids = model.generate(inputs, num_beams=4, max_length=256, min_length=64, early_stopping=True)
        pred_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        predictions.append(pred_summary)
        references.append(example["highlights"])
    scores = rouge.compute(predictions=predictions, references=references)
    return scores

def main():
    train_dataset, val_dataset = get_datasets()
    model, tokenizer = get_model_and_tokenizer()

    train_dataset = train_dataset.map(lambda ex: preprocess_function(ex, tokenizer), batched=True,
                                      remove_columns=["article", "highlights", "id"])
    val_dataset = val_dataset.map(lambda ex: preprocess_function(ex, tokenizer), batched=True,
                                  remove_columns=["article", "highlights", "id"])

    # ✅ Add custom data collator to pad inputs to multiple of 1024
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="longest",
        pad_to_multiple_of=1024
    )

    training_args = TrainingArguments(
        output_dir="./led_abstractive_model",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator  # ✅ Included here
    )

    trainer.train()
    model.save_pretrained("./led_abstractive_model")
    tokenizer.save_pretrained("./led_abstractive_model")

    scores = compute_rouge_scores(model, tokenizer, val_dataset)
    print("Abstractive Summarization ROUGE Scores:")
    print(f"ROUGE-1: {scores['rouge1']:.4f}")
    print(f"ROUGE-2: {scores['rouge2']:.4f}")
    print(f"ROUGE-L: {scores['rougeL']:.4f}")

if __name__ == "__main__":
    main()
