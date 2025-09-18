import numpy as np
import matplotlib.pyplot as plt
from src.models import single_models, multioutput_models

def plot_single_models(accuracies, precisions, recalls, f1_scores):
    '''
    Plots single models.
    
    args:
        accuracies (list): list of accuracies
        precisions (list): list of precisions
        recalls (list): list of recalls
        f1_scores (list): list of f1 scores
    '''
    model_names = [name for name, _ in single_models]

    x = np.arange(len(single_models))  
    width = 0.2                

    plt.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='dodgerblue')
    plt.bar(x - 0.5*width, precisions, width, label='Precision', color='deepskyblue')
    plt.bar(x + 0.5*width, recalls, width, label='Recall', color='lightskyblue')
    plt.bar(x + 1.5*width, f1_scores, width, label='F1 Score', color='lightblue')

    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.xticks(x, labels=model_names, rotation=45)
    plt.xlabel('Model')
    plt.ylabel('Percent')

    plt.tight_layout()
    plt.show()

def plot_acc(acc):
    '''
    Plots accuracy.

    args:
        accuracies (list): list of accuracies
    '''
    model_names = [name for name, _ in single_models]

    plt.bar(model_names, acc, width=0.5)

    for i in range(len(model_names)):
        plt.text(i, acc[i] // 2, round(acc[i], 1), ha='center')

    plt.xticks(rotation=45)
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')

    plt.show()
