import json
import numpy as np

import annotation_stats as db

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier


if __name__ == '__main__':
    # Get all comments from the /r/conservative. NOTE: we should probably be
    # querying for all the thricely labeled comments and filtering down.
    comment_ids = np.array(db.get_all_comments_from_subreddit2("Conservative"))
    ids, xs, ys = db.get_texts_and_labels_for_sentences(comment_ids)

    # Write the first ten sentences out to disk to play with
    examples = zip(xs, ys)
    poss = [ (x, y) for x, y in examples if y ==  1 ]
    negs = [ (x, y) for x, y in examples if y == -1 ]
    examples = { x:y for x, y in poss[:5] + negs[:5] }

    with open('comment.json', 'w') as comments_f:
        json.dump(examples, comments_f, sort_keys=True, indent=4)
    sys.exit()

    # Create arrays
    xs, ys = np.array(xs), np.array(ys)

    # Vectorize
    vectorizer = CountVectorizer(max_features=50000, ngram_range=(1,2), binary=True, stop_words="english")
    X = vectorizer.fit_transform(xs)

    # 5-fold test
    kf = KFold(len(xs), n_folds=5, shuffle=True)

    recalls, precisions, f_measures = [], [], []
    for train, test in kf:
        # Get training and test data for this round
        xs_train, xs_test = X[train], X[test]
        ys_train, ys_test = ys[train], ys[test]

        # Train svm
        svm = SGDClassifier(loss="hinge", penalty="l2", class_weight="auto", alpha=.01)
        parameters = { 'alpha': [.001, .01,  .1] }
        clf = GridSearchCV(svm, parameters, scoring='f1')
        clf.fit(xs_train, ys_train)

        # Make predictions
        predictions = clf.predict(xs_test)
