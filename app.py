from flask import Flask, request, jsonify,render_template
import re
import os
import nltk
from nltk.corpus import stopwords
from flask_cors import CORS
import speech_recognition as sr
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
CORS(app)

# Load the BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('./bert_model')
tokenizer = BertTokenizer.from_pretrained('./bert_tokenizer')
model.eval()

# Initialize LIME explainer
explainer = LimeTextExplainer(class_names=['Hate Speech', 'Not Hate Speech'])

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'@\w+', '', text)     
    text = re.sub(r'#\w+', '', text)    
    text = re.sub(r'\W', ' ', text)      
    text = text.lower().strip()
    return text

def predict_text_for_lime(texts):
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()
    return probabilities

def predict_text(text):
    cleaned_text = clean_text(text)
    inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    label = 'Not Hate Speech' if prediction == 2 else 'Hate Speech'
    return prediction, label

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided.'}), 400

    prediction, label = predict_text(text)

    if label == 'Hate Speech':
        # Use LIME to explain the prediction and highlight only hate speech-related keywords
        explanation = explainer.explain_instance(text, predict_text_for_lime, num_features=10)
        highlighted_text = [word for word, weight in explanation.as_list() 
                            if weight > 0 and word.lower() not in stop_words and word in text]

        return jsonify({'text': text, 'prediction': label, 'highlighted_keywords': highlighted_text})
    else:
        return jsonify({'text': text, 'prediction': label, 'highlighted_keywords': []})
    
@app.route('/upload', methods=['POST'])
def upload_media():
    print("Received files:", request.files)

    if 'audio' in request.files:
        media_file = request.files['audio']
        media_path = 'temp_audio.m4a'
    elif 'video' in request.files:
        media_file = request.files['video']
        media_path = 'temp_video.mp4'
        print("Processing video")
    else:
        return jsonify({'error': 'No audio or video file provided.'}), 400

    media_file.save(media_path)

    wav_path = 'temp_audio.wav'
    try:
        if media_path.endswith('.mp4'):
            print("Converting video to audio")
            clip = VideoFileClip(media_path)
            clip.audio.write_audiofile(wav_path)
            clip.close()
        else:
            print("Converting audio")
            audio = AudioSegment.from_file(media_path)
            audio.export(wav_path, format='wav')

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        os.remove(media_path)
        os.remove(wav_path)

        # Run the BERT prediction on transcribed text
        prediction = predict_text(text)

        return jsonify({'message': 'Media processed successfully.', 'transcription': text, 'prediction': prediction}), 200

    except Exception as e:
        if os.path.exists(media_path):
            os.remove(media_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
