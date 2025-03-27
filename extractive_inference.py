import torch
from transformers import LongformerForSequenceClassification, LongformerTokenizer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        model_dir = "./longformer_extractive_model"
        _tokenizer = LongformerTokenizer.from_pretrained(model_dir)
        _model = LongformerForSequenceClassification.from_pretrained(model_dir)
    return _model, _tokenizer

def summarize_extractive(text: str, top_k: int = 3) -> str:
    model, tokenizer = get_model()
    sentences = sent_tokenize(text)
    scores = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        outputs = model(**inputs)
        # We assume index 1 represents "important"
        score = torch.softmax(outputs.logits, dim=-1)[0][1].item()
        scores.append(score)
    top_sentences = [sent for _, sent in sorted(zip(scores, sentences), reverse=True)[:top_k]]
    return " ".join(top_sentences)

if __name__ == "__main__":
    sample_text = (
        "Your very long text here. It may have many sentences. "
        "This is another sentence with key details. More context follows."
    )
    print("Extractive Summary:", summarize_extractive(sample_text))
