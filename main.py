
from sklearn.svm import SVC
from FeatureMapping import *


model = None


def train_new_model():
    update_known_words()
    df = read_csv_data()

    """ *************************
        *   Construct X and Y   *
        ************************* """
    X, Y = [], []
    for i, df_row in df.iterrows():
        X.append(get_feature_vector(df_row['question']))
        Y.append(df_row['answer'])

    # Machine Learning Model.
    clf = SVC(kernel='linear', degree=5)
    clf.fit(X, Y)

    global model
    model = clf


def main():
    # Example
    q = "Hei, hva må jeg gjøre for å søke om boliglån?"
    print(q)
    print("Bag =", convert_to_bag_of_words(q))
    print()

    # Calculate the similarity of the new question in regards to the known questions.
    max_sim = get_max_similarity(q)
    print("similarity =", round(max_sim, 3))
    if max_sim < 0.2: # This must be tuned.
        print("Hei, det kan jeg ikke svare på")
    else:
        print(model.predict([get_feature_vector(q)])[0])

    print()
    print("most similar question/answer was: ", get_max_similarity_question(q))


if __name__ == '__main__':
    train_new_model()
    main()
