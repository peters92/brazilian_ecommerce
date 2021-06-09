from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (cross_val_score, RepeatedStratifiedKFold)
from sklearn.cluster import KMeans

def evaluate_model_with_cv(X, y, config):
    config = config['control']
    models = {}
    for penalty in config['regularization_penalties']:
        models[penalty] = \
            LogisticRegression(multi_class='multinomial', solver='lbfgs',
                               penalty='l2', C=penalty)

    cross_validation = \
        RepeatedStratifiedKFold(n_splits=config['cv_folds'],
                                n_repeats=config['cv_repeats'],
                                random_state=config['random_state'])
    
    cv_scores = {}
    print(f'{" Running cross-validation ":#^100}')
    for penalty, model in models.items():
        cv_score = cross_val_score(model, X, y, scoring='accuracy', 
                                   cv=cross_validation, n_jobs=-1)
        print(f'CV Score for model with l2 penalty = {penalty}\n'
              f'{cv_score}')

        cv_scores[penalty] = cv_score

    return cv_scores

def train_model(X, y):
    model = LogisticRegression(multi_class='multinomial',
                               solver='lbfgs')
    model.fit(X, y)

    return model

def train_kmeans(data, cluster_number=5):
    kmeans = KMeans(cluster_number)
    clusters = kmeans.fit_predict(data)
    return clusters