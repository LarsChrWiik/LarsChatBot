# LarsChatBot
Home made chatbot

<br>

NLP sentence pipeline:
- Tokenization
- Lower case
- Remove noise tokens
- Remove noise within tokens.
- Remove stop words
- Stemming

<br>

Model (Support Vector Classifier from SKLearn):
- SVC(kernel='linear', degree=5)

<br>

Similarity function:
- cosine_similarity of feature vector
- similarity treshold = 0.2
