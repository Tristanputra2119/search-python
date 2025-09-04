import os
import re
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from flask import send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Inisialisasi tools NLP Bahasa Indonesia
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_text(text):
    # Cleaning
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    
    # Tokenisasi
    tokens = text.split()
    
    # Filtering & Stemming
    cleaned_tokens = []
    for token in tokens:
        if token.strip() != '':
            stop_removed = stopword.remove(token)
            stemmed = stemmer.stem(stop_removed)
            if stemmed.strip() != '':
                cleaned_tokens.append(stemmed)
    
    return " ".join(cleaned_tokens)

def get_document_text(filepath):
    text = ""
    with open(filepath, 'rb') as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + " "
    return preprocess_text(text)

def build_search_index():
    documents = []
    filenames = []
    
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.endswith('.pdf'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            doc_text = get_document_text(filepath)
            documents.append(doc_text)
            filenames.append(filename)
    
    if documents:
        # Buat TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Simpan model dan data
        joblib.dump({
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'filenames': filenames,
            'documents': documents
        }, 'processed_data.pkl')
    else:
        # Hapus data lama jika tidak ada dokumen
        if os.path.exists('processed_data.pkl'):
            os.remove('processed_data.pkl')

def search_documents(query, top_k=5):
    if not os.path.exists('processed_data.pkl'):
        return []
    
    # Load model dan data
    data = joblib.load('processed_data.pkl')
    vectorizer = data['vectorizer']
    tfidf_matrix = data['tfidf_matrix']
    filenames = data['filenames']
    
    # Preprocess query
    processed_query = preprocess_text(query)
    query_vec = vectorizer.transform([processed_query])
    
    # Hitung cosine similarity
    cos_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Urutkan hasil
    results = []
    for i in np.argsort(cos_similarities)[::-1][:top_k]:
        if cos_similarities[i] > 0:
            results.append({
                'filename': filenames[i],
                'score': cos_similarities[i],
                'content_preview': " ".join(data['documents'][i].split()[:20]) + "..."
            })
    
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            build_search_index()
    # Ambil daftar file dan ukurannya (baik GET maupun POST)
    files = []
    for fname in os.listdir(app.config['UPLOAD_FOLDER']):
        if fname.endswith('.pdf'):
            size_kb = os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], fname)) / 1024
            files.append({'name': fname, 'size_kb': size_kb})
    return render_template('index.html', files=files)



@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    results = []
    if query:
        results = search_documents(query)
    return render_template('results.html', results=results, query=query)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)