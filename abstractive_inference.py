from transformers import LEDForConditionalGeneration, LEDTokenizer

# Caching the model and tokenizer
_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        model_dir = "./led_abstractive_model"
        _tokenizer = LEDTokenizer.from_pretrained(model_dir)
        _model = LEDForConditionalGeneration.from_pretrained(model_dir)
    return _model, _tokenizer

def summarize_abstractive(text: str, max_length: int = 256, min_length: int = 64) -> str:
    model, tokenizer = get_model()
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=4096, truncation=True)
    summary_ids = model.generate(
        inputs,
        num_beams=4,
        max_length=max_length,
        min_length=min_length,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    sample_text = "Your very long text here..."
    print("Abstractive Summary:", summarize_abstractive(sample_text))
