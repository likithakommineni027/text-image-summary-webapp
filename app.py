from flask import Flask, request, jsonify, render_template
import os

# Import inference functions from our modules.
from abstractive_inference import summarize_abstractive
from extractive_inference import summarize_extractive
from image_caption_inference import caption_image

app = Flask(__name__)

# Folder to store uploaded images.
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/summarize_text', methods=['POST'])
def summarize_text():
    data = request.get_json()
    text = data.get("text", "")
    mode = data.get("mode", "abstractive")
    
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    if mode == "abstractive":
        summary = summarize_abstractive(text)
    elif mode == "extractive":
        summary = summarize_extractive(text)
    else:
        return jsonify({"error": "Invalid summarization mode"}), 400

    return jsonify({"summary": summary})

@app.route('/caption_image', methods=['POST'])
def caption_image_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    caption = caption_image(file_path)
    return jsonify({"caption": caption})

if __name__ == "__main__":
    app.run(debug=True)

