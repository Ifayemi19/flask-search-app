from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Texte de recherche
query = input("Saisir votre recherche")

# Documents
document1 = '''The release of DeepSeek R1 stunned Wall Street and Silicon Valley this month, ...'''
document2 = '''Two years ago, when big-name Chinese technology companies like Baidu and Alibaba ...'''
document3 = '''When a small Chinese company called DeepSeek revealed that it had created an A.I. system ...'''

texts = [query, document1, document2, document3]

# Vectorisation TF-IDF
vect = TfidfVectorizer()
tfidf_mat = vect.fit_transform(texts)

# Calcul de la similarité cosinus
similarities = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:]).flatten()

# Affichage des résultats
threshold = 0.20  # Seuil de similarité
for i, similarity in enumerate(similarities):
    if similarity > threshold:
        print(f"Document {i+1}:")
        print(texts[i+1])
        print(f"Similarité : {similarity:.4f}")
        print("-" * 50)
