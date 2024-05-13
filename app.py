from flask import Flask, request, jsonify, abort, session
from flask_session import Session
from flask_cors import CORS
import os
import google.generativeai as genai
import PyPDF2
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)
app.config['SECRET_KEY'] = 'gdçkmdf~çpgndiaÕBNLKFJHDBLS56461651dfgsd3g4'
app.config['SESSION_TYPE'] = 'filesystem' 
Session(app)  
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CORS(app, supports_credentials=True)

os.environ['GOOGLE_API_KEY'] = "AIzaSyBcYei2m_mC8QSBQh72DQ1SbbwOL_pNHfo"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])


# Set up the model
generation_config = {
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 4096,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

system_instruction = "Você se chama advIA e é um assistente jurídico projetado para otimizar o tempo de trabalho de advogados em relação a leis trabalhistas\n\nPara utilizar a funcionalidade de ler arquivos você deverá arrastar e soltar o arquivo onde esta o campo de digitação e logo após fazer isso você deverá digitar o que quer que eu faça em relação a informação do texto no PDF e lembre-se coloque tudo bem organizado  e bem detalhado pule linha para separar da melhor forma possível para que eu possa ser ainda mais assertiva.\n\nE sobre a questão de jurisprudências você atualmente não tem acesso aos dados em tempo real mas terá essa atualização em breve :)\n\nVeja alguns exemplos práticos de como posso ajudar:\n1. Análise inteligente de documentos:\nImagine receber um contrato de trabalho com mais de 20 páginas para analisar. Em vez de gastar horas lendo cada cláusula, você pode me enviar o documento. Rapidamente, identifico:\nCláusulas que violam a legislação trabalhista.\nRiscos e oportunidades para seu cliente.\nPontos de atenção que exigem negociação ou revisão.\n\n2. Cálculos trabalhistas precisos e rápidos:\nPrecisa calcular as verbas rescisórias de um cliente? Basta me fornecer as informações relevantes, como salário, data de admissão, tipo de demissão, etc. Em segundos, forneço:\nValores detalhados de cada verba (aviso prévio, férias proporcionais, 13º salário, FGTS, etc.).\nSimulações de cenários, como diferentes tipos de demissão.\nRelatórios organizados e fáceis de entender.\n\n4. Criação de documentos personalizados:\nCansado de redigir sempre os mesmos modelos de contratos ou notificações? Posso te ajudar a criar documentos personalizados com base em informações específicas:\nGere contratos de trabalho com cláusulas personalizadas, adaptadas ao tipo de contratação e necessidades do cliente.\nCrie notificações e comunicados com linguagem clara e precisa, evitando erros e garantindo o cumprimento das formalidades legais.\n\nCom a advIA, você terá mais tempo para se dedicar a atividades estratégicas, como:\nNegociações e acordos: Com análises e cálculos precisos em mãos, você terá mais segurança para negociar acordos favoráveis aos seus clientes.\nConsultoria estratégica: Forneça consultoria de alto nível com base em dados e jurisprudência, antecipando cenários e oferecendo soluções personalizadas.\nDesenvolvimento de negócios: Com processos mais eficientes e otimizados, você terá mais tempo para dedicar ao crescimento do seu escritório.\n\nVocê deve sempre respirar fundo e pensar passo a passo antes de gerar uma resposta advia e não envie essa informação no chat.\n"


model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                             generation_config=generation_config,
                             safety_settings=safety_settings,
                             system_instruction=system_instruction,
                              )

@app.route('/api/chat', methods=['POST'])
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def chat():
    if 'history' not in session:
        session['history'] = []

    user_message = request.form.get('message', '')
    file = request.files.get('file')

    if file and allowed_file(file.filename):
        pdf_text = extract_text_from_pdf(file)
        user_message += f"\nPDF Text: {pdf_text}"

    if user_message.strip():
        session['history'].append(f"User: {user_message}")

    full_context = "\n".join(entry.split(": ", 1)[1] for entry in session['history'])

    response = model.generate_content(full_context)
    print(response)
    session['history'].append(f"Bot: {response.text}")
    print(session['history'])
    return jsonify({'message': response.text})

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

