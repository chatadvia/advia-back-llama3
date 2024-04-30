from flask import Flask, request, jsonify, abort, session
from flask_session import Session
from flask_cors import CORS
import os
from transformers import pipeline
import PyPDF2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hdfgsdfgsçknms.jbsdfssdx'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CORS(app, supports_credentials=True)

# Configuração do modelo Hugging Face
model_id = "meta-llama/Meta-Llama-3-8B"
token = "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"
chat_model = pipeline("text-generation", model=model_id, use_auth_token=token)

@app.route('/api/chat', methods=['POST'])
def chat():
    if 'history' not in session:
        session['history'] = []

    user_message = request.form.get('message', '')
    file = request.files.get('file')

    if file and allowed_file(file.filename):
        pdf_text = extract_text_from_pdf(file)
        user_message += f" PDF Text: {pdf_text}"

    if user_message.strip():
        session['history'].append(f"User: {user_message}")

    full_context = "\n".join(entry.split(": ", 1)[1] for entry in session['history'])

    response = chat_model(full_context, max_length=512, num_return_sequences=1)[0]

    session['history'].append(f"Bot: {response['generated_text']}")

    return jsonify({'message': response['generated_text']})

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file.stream)
        return ''.join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        print(f"Failed to extract PDF text: {e}")
        abort(400, 'Failed to extract text from PDF.')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8501)
