from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BartForConditionalGeneration, BartTokenizer

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

MODEL_NAME = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

@app.route('/summarize', methods=['POST'])
def summarize_text():
    text = request.json.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=30, max_length=400, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True)
