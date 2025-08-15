from sklearn import svm, metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import clone

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

svm_model = svm.SVC(kernel='linear', random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=3)
dt_model = DecisionTreeClassifier(random_state=42)
nb_model = GaussianNB()
regression_model = LogisticRegression(max_iter=500, random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)

single_models = [['SVM', svm_model],
          ['KNN', knn_model],
          ['Decision Tree', dt_model],
          ['Naive Bayes', nb_model],
          ['Logistic Regression', regression_model],
          ['Gradient Boost', gb_model]]

multioutput_models = [['SVM', MultiOutputClassifier(svm_model)],
          ['KNN', MultiOutputClassifier(knn_model)],
          ['Decision Tree', MultiOutputClassifier(dt_model)],
          ['Naive Bayes', MultiOutputClassifier(nb_model)],
          ['Logistic Regression', MultiOutputClassifier(regression_model)],
          ['Gradient Boost', MultiOutputClassifier(gb_model)]]

label_names = ['Substance', 'Volume']

def make_split(x, y, random_state):
    '''
    Splits data into training and test sets.

    args:
        x (pd.DataFrame): features
        y (pd.DataFrame): labels
        random_state (int): random state
    returns:
        x_train_scaled (pd.DataFrame): scaled training features
        x_test_scaled (pd.DataFrame): scaled test features
        y_train (pd.DataFrame): training labels
        y_test (pd.DataFrame): test labels
    '''

    if y.ndim == 2:
        combined_y = y['substance'].astype(str) + '_' + y['volume'].astype(str)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=combined_y, random_state=random_state)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=random_state)

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test

def train_single_models(x, y, random_state=42):
    '''
    Trains single models.

    args:
        x (pd.DataFrame): features
        y (pd.DataFrame): labels
        random_state (int): random state
    returns:
        accuracies (list): list of accuracies
        precisions (list): list of precisions
        recalls (list): list of recalls
        f1_scores (list): list of f1 scores
    '''
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    x_train_scaled, x_test_scaled, y_train, y_test = make_split(x, y, random_state)
    
    models_copy = [[name, clone(model)] for name, model in single_models]

    for model in models_copy:
        model[1].fit(x_train_scaled, y_train)
        y_pred = model[1].predict(x_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1score = f1_score(y_test, y_pred, average='weighted')
        model_confusion_matrix = confusion_matrix(y_test, y_pred)

        accuracies.append(accuracy * 100)
        precisions.append(precision * 100)
        recalls.append(recall * 100)
        f1_scores.append(f1score * 100)
        
        print('----------------------------')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 score: {f1score:.4f}')
        
        plt.figure(figsize=(4, 3))
        sns.heatmap(model_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model[1].classes_, yticklabels=model[1].classes_)
        plt.title(f'{model[0]} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    return accuracies, precisions, recalls, f1_scores

def train_multioutput_models(x, y, random_state=42, show_cm=False, show_metrics=False):
    '''
    Trains multioutput models.

    args:
        x (pd.DataFrame): features
        y (pd.DataFrame): labels
        random_state (int): random state
        show_cm (bool): whether to show confusion matrices
        show_metrics (bool): whether to show metrics
    returns:
        acc (list): list of accuracies
        per_label_acc (list): list of per-label accuracies
    '''

    acc = []
    per_label_acc = []

    x_train, x_test, y_train, y_test = make_split(x, y, random_state)
    
    models_copy = [[name, clone(model)] for name, model in multioutput_models]
    
    for model in models_copy:
        model[1].fit(x_train, y_train)
        y_pred = model[1].predict(x_test)
    
        # Exact match accuracy (all labels correct)
        exact_match_accuracy = np.mean(np.all(y_test.values == y_pred, axis=1))
    
        # Per-label accuracy
        per_label_accuracy = (y_test == y_pred).mean(axis=0)
    
        acc.append(exact_match_accuracy * 100)
        per_label_acc.append(per_label_accuracy * 100)

        # Confusion matrices for each label
        if show_cm:
            for i, col in enumerate(y_test.columns):
                cm = confusion_matrix(y_test.iloc[:, i], y_pred[:, i])
                plt.figure(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{model[0]} Confusion Matrix, {label_names[i]}')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.show()

    if show_metrics:
        print_metrics(acc, per_label_acc)

    return acc, per_label_acc

def print_metrics(exact_acc, per_label_acc):
    '''
    Prints metrics for multioutput models.

    args:
        exact_acc (list): list of exact accuracies
        per_label_acc (list): list of per-label accuracies
    returns:
        None
    '''
    model_names = [name for name, _ in multioutput_models]
    for i, name in enumerate(model_names):
        print(name)
        print(f'Exact Match Accuracy: {exact_acc[i]:.4f}')
        print(f'Per-Label Accuracies: {per_label_acc[i].values if hasattr(per_label_acc[i], "values") else per_label_acc[i]}')
        print('----------------------------')

def plot_acc(acc, plt_title):
    '''
    Plots accuracy for multioutput models.

    args:
        acc (list): list of accuracies
        plt_title (str): title of plot
    returns:
        None
    '''
    model_names = [name for name, _ in multioutput_models]

    plt.bar(model_names, acc, width=0.5)

    plt.xticks(rotation=-45)
    plt.xlabel('Model')
    plt.ylabel('Exact Accuracy (%)')
    plt.title(f'Exclude {plt_title}')
    plt.tight_layout()
    plt.show()