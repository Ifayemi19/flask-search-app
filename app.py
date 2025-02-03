from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/search', methods=["GET"])
def search_utilisateur():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Veuillez entrer une requête valide."})
    
    results = recherche_expression(query)
    return render_template("results.html", query=query, results=results)

def recherche_expression(query):
    texts = [
        query,
        "Le traitement du langage naturel est fascinant.",
        "Le traitement des langues est une branche de l'intelligence artificielle.",
        "L'analyse de texte est utilisée pour la traduction automatique."
    ]

    vect = TfidfVectorizer()
    tfidf_mat = vect.fit_transform(texts).toarray()
    
    query_tf_idf = tfidf_mat[0]
    corpus = tfidf_mat[1:]
    results = []
    
    for idx, document_tf_idf in enumerate(corpus):
        pearson_corr, _ = pearsonr(query_tf_idf, document_tf_idf)
        if pearson_corr > 0.20:
            results.append({"ID": idx, "document": texts[idx + 1], "similarity": pearson_corr})
    
    return results

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
