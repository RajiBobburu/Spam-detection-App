def predict_text(text, model, tfidf):
    features = tfidf.transform([text])
    prediction = model.predict(features)[0]

    label = "SPAM" if prediction == 1 else "HAM"
    return label