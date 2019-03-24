
from Pipeline import convert_to_bag_of_words
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


filename_data = 'data.txt'
filename_known_words = 'known_words.csv'
encoding = 'utf-8' # utf-8 latin1

feature_encoder = LabelEncoder()
known_words = None


def __get_answers_all():
    df = read_csv_data()
    return df['answer'].values.tolist()


def __get_questions():
    df = pd.read_csv(filename_data, sep=";", encoding=encoding)
    return df['question'].values.tolist()


def __get_questions_bags():
    questions = __get_questions()
    return list(map(convert_to_bag_of_words, questions))


def __read_csv_known_words():
    """
    Reads the known words and convert it to a set of words.

    :return: list(str).
    """

    def __flatten(data):
        return [x for data2 in data for x in data2]

    questions_bags = __get_questions_bags()
    words = __flatten(questions_bags)
    return sorted(list(set(words)))


def __read_csv_known_words_all():
    """
    Reads the known words and convert it to a set of words.

    :return: list(str).
    """

    def __flatten(data):
        return [x for data2 in data for x in data2]

    questions_bags = __get_questions_bags()
    words = __flatten(questions_bags)
    return sorted(words)


def read_csv_data():
    """
    Reads the file containing questions and answers.
    Additionally, it duplicates the above answer is no answer is given.
    This allows an answer to be connected to multiple questions.

    :return: DataFrame with two columns (question, answer).
    """
    df = pd.read_csv(filename_data, sep=";", encoding=encoding)
    for i, df_row in df.iterrows():
        if pd.isnull(df_row['answer']):
            df.iloc[i]['answer'] = df.iloc[i - 1]['answer']
    return df


def update_known_words():
    global known_words
    global feature_encoder

    known_words = __read_csv_known_words_all()
    feature_encoder.fit(known_words)


def get_feature_vector(question_series):
    global known_words
    global feature_encoder

    question_words = convert_to_bag_of_words(question_series)
    question_words = [w for w in question_words if w in known_words]
    feature_vector = np.zeros(len(known_words))
    for i in feature_encoder.transform(question_words):
        feature_vector[i] += 1
    return feature_vector


def get_max_similarity(question):
    all_questions = __get_questions_bags()
    question = convert_to_bag_of_words(question)
    f1 = get_feature_vector(question)

    similarities = []
    for q in all_questions:
        f2 = get_feature_vector(q)
        sim = cosine_similarity([f1], [f2])[0][0]
        similarities.append(sim)
    return max(similarities)


def get_max_similarity_question(question):
    questions_raw = __get_questions()
    answers_raw = __get_answers_all()

    all_questions = __get_questions_bags()
    question = convert_to_bag_of_words(question)
    f1 = get_feature_vector(question)

    similarities = []
    for q in all_questions:
        f2 = get_feature_vector(q)
        similarities.append(cosine_similarity([f1], [f2])[0][0])
    max_index = np.argmax(similarities)

    return questions_raw[max_index], answers_raw[max_index]
