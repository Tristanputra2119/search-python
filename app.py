import os
import re
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, send_from_directory
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Inisialisasi NLP tools
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

# Route untuk mengakses file upload
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Preprocessing teks
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    tokens = text.split()
    cleaned_tokens = []
    for token in tokens:
        if token.strip():
            stop_removed = stopword.remove(token)
            stemmed = stemmer.stem(stop_removed)
            if stemmed.strip():
                cleaned_tokens.append(stemmed)
    return " ".join(cleaned_tokens)

# Ambil teks mentah PDF
def get_raw_text(filepath):
    text = ""
    with open(filepath, 'rb') as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

# Build atau update search index secara incremental
def build_search_index():
    if os.path.exists('processed_data.pkl'):
        data = joblib.load('processed_data.pkl')
        filenames = data['filenames']
        documents = data['documents']
        raw_documents = data['raw_documents']
        vectorizer = data['vectorizer']
        tfidf_matrix = data['tfidf_matrix']
    else:
        filenames, documents, raw_documents = [], [], []
        vectorizer = TfidfVectorizer(max_features=10000)
        tfidf_matrix = None

    current_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')]
    new_files = [f for f in current_files if f not in filenames]

    new_docs_processed = []
    new_docs_raw = []
    for filename in new_files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        raw_text = get_raw_text(filepath)
        processed_text = preprocess_text(raw_text)
        new_docs_raw.append(raw_text)
        new_docs_processed.append(processed_text)
        filenames.append(filename)

    if new_docs_processed:
        if tfidf_matrix is None:
            # Build pertama kali
            tfidf_matrix = vectorizer.fit_transform(new_docs_processed)
            documents = new_docs_processed
            raw_documents = new_docs_raw
        else:
            # Incremental: transform dokumen baru & gabungkan
            new_tfidf = vectorizer.transform(new_docs_processed)
            tfidf_matrix = vstack([tfidf_matrix, new_tfidf])
            documents.extend(new_docs_processed)
            raw_documents.extend(new_docs_raw)

        # Simpan kembali
        joblib.dump({
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'filenames': filenames,
            'documents': documents,
            'raw_documents': raw_documents
        }, 'processed_data.pkl')

# Fungsi search dengan snippet + highlight
def search_documents(query, top_k=5):
    if not os.path.exists('processed_data.pkl'):
        return []

    data = joblib.load('processed_data.pkl')
    vectorizer = data['vectorizer']
    tfidf_matrix = data['tfidf_matrix']
    filenames = data['filenames']
    raw_documents = data['raw_documents']

    processed_query = preprocess_text(query)
    query_vec = vectorizer.transform([processed_query])
    cos_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(cos_similarities)[::-1][:top_k]

    results = []
    for i in top_indices:
        if cos_similarities[i] > 0:
            # Cari kata kunci untuk snippet
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            match = pattern.search(raw_documents[i])
            if match:
                start = max(match.start() - 50, 0)
                end = match.end() + 150
                snippet = raw_documents[i][start:end]
            else:
                snippet = raw_documents[i][:200]
            snippet = pattern.sub(r'<mark>\g<0></mark>', snippet)
            results.append({
                'filename': filenames[i],
                'score': cos_similarities[i],
                'content_preview': snippet + "..."
            })
    return results

# Halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            build_search_index()

    files = []
    for fname in os.listdir(app.config['UPLOAD_FOLDER']):
        if fname.endswith('.pdf'):
            size_kb = os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], fname)) / 1024
            files.append({'name': fname, 'size_kb': size_kb})
    return render_template('index.html', files=files)

# Halaman search
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
