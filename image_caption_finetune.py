from datasets import load_dataset
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, Trainer, TrainingArguments
import torch
import evaluate

_dataset = None
_model_tokenizer_fe = None

def get_dataset():
    global _dataset
    if _dataset is None:
        dataset = load_dataset("coco_captions", "2017", split="train[:1%]")
        def preprocess_function(examples):
            images = examples["image"]
            captions = [cap[0] if cap else "" for cap in examples["annotations"]["caption"]]
            return {"images": images, "captions": captions}
        dataset = dataset.map(preprocess_function, batched=True)
        _dataset = dataset
    return _dataset

def get_model_and_tokenizer_fe():
    global _model_tokenizer_fe
    if _model_tokenizer_fe is None:
        model_name = "nlpconnect/vit-gpt2-image-captioning"
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model_tokenizer_fe = (model, feature_extractor, tokenizer)
    return _model_tokenizer_fe

def prepare_features(examples, feature_extractor, tokenizer):
    images = [feature_extractor(image.convert("RGB"), return_tensors="pt").pixel_values[0]
              for image in examples["images"]]
    labels = tokenizer(examples["captions"], padding="max_length", max_length=16, truncation=True).input_ids
    examples["pixel_values"] = images
    examples["labels"] = labels
    return examples

def compute_rouge_image(model, feature_extractor, tokenizer, dataset, num_examples=20):
    rouge = evaluate.load("rouge")
    predictions = []
    references = []
    for example in dataset.select(range(num_examples)):
        image = example["images"]
        pixel_values = feature_extractor(image.convert("RGB"), return_tensors="pt").pixel_values
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
        pred_caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(pred_caption)
        references.append(example["captions"])
    scores = rouge.compute(predictions=predictions, references=references)
    return scores

def main():
    dataset = get_dataset()
    model, feature_extractor, tokenizer = get_model_and_tokenizer_fe()

    dataset = dataset.map(lambda ex: prepare_features(ex, feature_extractor, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir="./image_caption_model",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_total_limit=1,
        fp16=True,
    )

    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["labels"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    trainer.train()
    model.save_pretrained("./image_caption_model")
    tokenizer.save_pretrained("./image_caption_model")
    feature_extractor.save_pretrained("./image_caption_model")

    scores = compute_rouge_image(model, feature_extractor, tokenizer, dataset)
    print("Image Captioning ROUGE Scores:")
    print(f"ROUGE-1: {scores['rouge1']:.4f}")
    print(f"ROUGE-2: {scores['rouge2']:.4f}")
    print(f"ROUGE-L: {scores['rougeL']:.4f}")

if __name__ == "__main__":
    main()
