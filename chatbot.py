import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import pandas as pd


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stopwords.words('english')


faq = pd.read_csv('chatbot.csv')

tfidf = TfidfVectorizer(use_idf=True, analyzer='word', stop_words='english',
                        token_pattern=r'\b[^\d\W]+\b', ngram_range=(1, 2))


def predict(sentence):

    with open('chatbot.model', 'rb') as f:
        lr = pickle.load(f)

    loaded_vectorizer = pickle.load(open('chatbot_vectorizer.pickle', 'rb'))

    search_engine = loaded_vectorizer.transform([sentence])
    result = lr.predict(search_engine)
    for question in result:
        faq_data = faq.loc[faq.isin([question]).any(axis=1)]
        print(faq_data['Answers'].values)
        return faq_data['Answers'].values[0]

