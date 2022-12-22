import pickle


def predict_quiz(answers):
    with open('quiz.h5', 'rb') as f:
        model = pickle.load(f)

    loaded_vectorizer = pickle.load(open('quiz_vectorizer.pickle', 'rb'))
    answers_scaled = loaded_vectorizer.transform([answers])
    result = model.predict(answers_scaled)
    return result[0]
