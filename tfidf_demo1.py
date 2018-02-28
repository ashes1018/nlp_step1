from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



corpus = [ 'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',  ]
vector = CountVectorizer()
x = vector.fit_transform(corpus)
word = vector.get_feature_names()

print(x.toarray())
print(word)

transform = TfidfTransformer()
print(transform)
tfidf = transform.fit_transform(x)
print(tfidf.toarray())
