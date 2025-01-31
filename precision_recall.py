# precision_recall
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
import pandas as pd
import seaborn as sns 

# Source code credit for this function: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Truth')
    plt.xlabel('Prediction')
    plt.show()
    
truth =      ["Dog","Not a dog","Dog","Dog",      "Dog", "Not a dog", "Not a dog", "Dog",       "Dog", "Not a dog"]
prediction = ["Dog","Dog",      "Dog","Not a dog","Dog", "Not a dog", "Dog",       "Not a dog", "Dog", "Dog"]
cm = confusion_matrix(truth,prediction)
print_confusion_matrix(cm,["Dog","Not a dog"])
    
print(classification_report(truth, prediction))
    
#f1 score for Dog class
    
f1_score = 2*(0.57*0.67/(0.57+0.67))
    
#f1 score for Not a dog class
f1_score_no_dog = 2*(0.33*0.25/(0.33+0.25))
    
print('F1 score for dog class:\n',f1_score)
print('F1 score for dog class:\n',f1_score_no_dog)
    
# precision is out of all the dog prediction how many you got it right?
# Truth positive = 4 False positive = 3
# precision = 4 / 7 = 0.57 
#Formula for Precision = TP/(TP+FP)
    
#Recall is out of all dog truth how many you got it right?
#Total Dog Truth Sample = 6   True Positive = 4 
#Recall = 4/6 = 0.67 
# Recall formula = TP/(TP + FN) =4/( 4 + 2 ) = Here 4 positive, total dog is available 6 so 4 - 6 = 2 is the FalseNegative(FN)
    
#F1 Score Formula is 2 * Precision * Recall / (Precision + Recall)

F1_SCORE_FOR_DOG = 2 * (0.57 *0.67) / (0.57 + 0.67)
print('F1 score for Dog is:',F1_SCORE_FOR_DOG)
    
    