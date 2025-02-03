from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr

app = Flask(__name__)

@app.route('/')
def index():
   return render_template("index.html")

@app.route('/search', methods=["GET","POST"])
def search_utilisateur():
   query = request.args.get("query") ### methode est GET
   #query = request.form.get("query") ### methode est POST
   return recherche_expression(query)


def recherche_expression(query):
   
   #### Recherche de l’utilisateur
   ####query = input("Saisir votre recherche")

   # Textes à comparer / Base de connaissances / KB / Corpus
   texts = [ 
   query, 
   "Le traitement du langage naturel est fascinant.", 
   "Le traitement des langues est une branche de l'intelligence artificielle.", 
   "L'analyse de texte est utilisée pour la traduction automatique." 
   ]

   # Vectorisation TF-IDF
   vect = TfidfVectorizer()
   tfidf_mat = vect.fit_transform(texts).toarray()

   query_tf_idf = tfidf_mat[0]
   corpus = tfidf_mat[1:]

   #print(f'Recherche User : {str(query_tf_idf)}')
   #print(f'Corpus / Base de connaissances : {str(corpus)}')


   # Corellation de pearson
   for id, document_tf_idf in enumerate(corpus):
      pearson_corr, _ = pearsonr(query_tf_idf, document_tf_idf)
      if pearson_corr > 0.20:
         result = {"ID": id, "document": texts[id+1], "similarity": pearson_corr}
         return result

   

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=80, debug=True)
