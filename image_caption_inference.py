from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image

_model = None
_feature_extractor = None
_tokenizer = None

def get_model():
    global _model, _feature_extractor, _tokenizer
    if _model is None or _feature_extractor is None or _tokenizer is None:
        model_dir = "./image_caption_model"
        _model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        _feature_extractor = ViTFeatureExtractor.from_pretrained(model_dir)
        _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return _model, _feature_extractor, _tokenizer

def caption_image(image_path: str, max_length: int = 16, num_beams: int = 4) -> str:
    model, feature_extractor, tokenizer = get_model()
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    test_image_path = "path_to_your_image.jpg"  # Replace with an actual image path.
    print("Image Caption:", caption_image(test_image_path))
