# LarsChatBot
Home made chatbot


NLP sentence pipeline:
- Tokenization
- Lower case
- Remove noise tokens
- Remove noise within tokens.
- Remove stop words
- Stemming


Model (Support Vector Classifier from SKLearn):
- SVC(kernel='linear', degree=5)


Similarity function:
- cosine_similarity of feature vector
- similarity treshold = 0.2
