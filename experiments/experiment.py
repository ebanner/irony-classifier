import annotation_stats as db

import sklearn

if __name__ == '__main__':
    # Get all comments from the /r/conservative
    comment_ids = db.get_all_comments_from_subreddit("Conservative")
    ids, texts, ys = db.get_texts_and_labels_for_sentences(comment_ids)

    # Extract features
    features = [ [len(text)] for text in texts ]

    # Run it through a learning algorithm
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(features, ys)

    # Plot the data
    training_examples = zip(features, ys)
    plusses = [ (feature, label) for feature, label in training_examples if label == 1 ]
    minuses = [ (feature, label) for feature, label in training_examples if label == -1 ]
    print('plusses: {}'.format(plusses))
    print('minuses: {}'.format(minuses))

    plus_features = [ feature[0] for feature, label in plusses ]
    minus_features = [ feature[0] for feature, label in minuses ]

    print('plusses = {}'.format(plus_features))
    print('minues = {}'.format(minus_features))
