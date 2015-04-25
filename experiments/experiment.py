import json
import numpy as np

import annotation_stats as db
import sklearn

if __name__ == '__main__':
    # Get all comments from the /r/conservative
    comment_ids = db.get_all_comments_from_subreddit("Conservative")
    ids, texts, ys = db.get_texts_and_labels_for_sentences(comment_ids)
    sentiments = db.get_sentiments(comment_ids)    

    # Extract features
    length_features = [ len(text) for text in texts ] 
    sentiment_features = [ sentiment for sentiment in sentiments ]
    features = [ [sentiment_feature, length_feature] for length_feature, sentiment_feature in zip(length_features, sentiment_features)]

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

        predictions = clf.predict(xs_test)
