"""
A collection of useful tools for data analysis.

Initially created on Apr 26 2019
This version: Jun 3 2019

@author: Shichao Ma
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, confusion_matrix, \
    accuracy_score, roc_auc_score, precision_recall_curve
import itertools
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce


def get_part_of_day(timestamp):
    '''
    How to use:
    dataset['timestamp'] = dataset['timestamp'].apply(lambda x: get_part_of_day(x))
    '''
    hour = timestamp.hour
    if 6 <= hour < 12:
        return 'morning'
    if 12 <= hour < 18:
        return 'afternoon'
    if 18 <= hour < 24:
        return 'evening'
    if 0 <= hour < 6:
        return 'midnight'


def compute_datetime_diff(timestamp1, timestamp2, output='days'):
    if output == 'days':
        return (timestamp1 - timestamp2).dt.days
    elif output == 'seconds':
        return (timestamp1 - timestamp2).dt.seconds
    else:
        raise Error('Invalid type')

def plot_preprocessing(X, y):
    '''
    Target encoding and scaling for drawing pictures
    '''
    encoder = ce.TargetEncoder(cols=categorical_columns)
    X_encoded = encoder.fit_transform(X, y)
    X_std = MinMaxScaler().fit_transform(X_encoded)
    X_encoded = pd.DataFrame(X_std, index=X_encoded.index, columns=X_encoded.columns)
    # X_encoded can only be used for drawing pictures
    # Do NOT use it for training models as target encoding will leak info
    return X_encoded
    

def categorical_value_counts(dataset, categorical_variables):
    for column in categorical_variables:
        print(dataset[column].value_counts())
        print()
    return 


def compute_percentage_of_missing_values(dataset):
    print(100 * dataset.isna().sum()/len(dataset))
    return


def try_parsing_datetime(time_string, formats):
    '''
    Try parsing a datetime string that may have various formats.
    How to use: 
    formats = ['%m.%d.%Y', '%Y-%m-%d']
    dataset['date'] = dataset['date'].apply(lambda x: try_parsing_date(x, formats))
    '''
    if pd.isnull(time_string):
        return pd.NaT
    for fmt in formats:
        try:
            return datetime.strptime(time_string, fmt)
        except ValueError:
            pass
    raise ValueError('No valid date format found. The time string is ' + time_string)
    

def check_valid_datetime(timestamp, left_boundary=None, right_boundary=None):
    '''
    Check if a datatime object is in a given boundary
    boundary = None means infinity
    '''
    if pd.isna(timestamp):
        return False
    decision = True
    if left_boundary is not None and timestamp < left_boundary:
        decision = False
    if right_boundary is not None and timestamp > right_boundary:
        decision = False
    return decision


def plot_hist(x, variable_name=None, zero_rm=False):
    '''
    A better version of histogram plot.
    Use zero_rm=True if too many zero values in a distribution
    with non-negative support.
    '''
    if not variable_name:
        variable_name = x.name
    x = x[pd.notnull(x)]
    if zero_rm:
        x = x[x>0]
    plt.figure()
    n, bins, patches = plt.hist(x, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of ' + variable_name)
    return


def mean_str(col):
    '''
    This function finds the "mean" value in a string column:
    1. If all the rows have the same value, return that value.
    2. If they have different values, return 'conflicted'.
    3. If all the rows are empty, return NaN.
    '''
    col_unique = col.nunique()
    if col_unique == 1:
        return col.dropna().unique()
    elif col_unique > 1:
        return 'conflicted'
    else:
        return np.NaN
    

def plot_preprocessing(X, y):
    '''
    Target encoding and scaling for drawing pictures
    '''
    encoder = ce.TargetEncoder(cols=categorical_columns)
    X_encoded = encoder.fit_transform(X, y)
    X_std = MinMaxScaler().fit_transform(X_encoded)
    X_encoded = pd.DataFrame(X_std, index=X_encoded.index, columns=X_encoded.columns)
    # X_encoded can only be used for drawing pictures
    # Do NOT use it for training models as target encoding will leak info
    return X_encoded


def plot_violin(X, y):
    '''
    Plot conditional distributions of X's each column.
    '''
    df = pd.concat([X, y], axis=1)
    df_long = pd.melt(df, y.name, var_name='Features', value_name='Value')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.violinplot(x='Features',y='Value',inner='quartile',\
                   data=df_long,hue=y.name,split=False,ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.show()
    return


def plot_correlation(X):
    corr = X.corr()
    plt.figure()
    sns.heatmap(corr, xticklabels=X.columns, yticklabels=X.columns, annot=False)
    return
   

def plot_2d_space(X, y, label=None): 
    
    '''
    This function applies PCA to X and plot it against Y.
    It is mainly used for exploratory analysis to see if 
    Y is separable by X.
    '''
    
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    return



def plot_roc(y, y_pred, model_name):
    """
    This function plots the ROC curve.
    """
    fpr, tpr = dict(), dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y, y_pred)
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    plt.plot(fpr[1], tpr[1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (recall)')
    plt.title('ROC for %s' % model_name)
    return


def plot_precision_recall(y_test, y_predict_prob, modelString):
    precision, recall, thresholds = precision_recall_curve(y_test, y_predict_prob)
    plt.figure()
    plt.plot(precision, recall)
    plt.xlabel("Precision") 
    plt.ylabel("Recall") 
    plt.title('Precision-Recall Curve of %s' % modelString)
    return


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.rcParams.update({'font.size': 14})

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return 


def calculate_auc(y, y_pred):
    """ 
    This function calculates the AUROC score.
    y: ground truth
    y_pred: probability of being positive
    """
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    return auc(fpr, tpr)


def evaluate_pred(y, y_pred):
    """ 
    This function calculates sensitivity and specificity
    y: ground truth
    y_pred: predicted zeros and ones
    """
    result = pd.DataFrame(data = {'y': y, 'y_pred': y_pred})
    tp = sum((result['y'] == 1) & (result['y'] == 1)) # true positive
    fn = sum((result['y'] == 1) & (result['y'] == 0)) # false negative
    fp = sum((result['y'] == 0) & (result['y'] == 1)) # false positive
    tn = sum((result['y'] == 0) & (result['y'] == 0)) # true negative
    sensitivity = float(tp) / (tp + fn)
    specificity = 1 - float(fp) / (fp + tn)
    return {'sensitivity': sensitivity, 'specificity': specificity}


def plot_coefficients(classifier, feature_names, top_features=20):
    '''
    This function plots top features from a regression.
    '''
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure()
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()
    return


def plot_importance(importances, features, indices):
    '''
    This function plots feature importance.
    '''
    plt.figure()
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
    plt.yticks(range(len(indices)), features[indices]) 
    plt.xlabel('Relative Importance')
    plt.show()
    return
    
