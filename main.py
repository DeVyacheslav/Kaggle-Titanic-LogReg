import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import re
import collections


def prepare_df(df):

    min_occurence = 10
    # all passengers have titles
    df['Name'] = df['Name'].apply(lambda x: re.search(r', (.*?)\.', x)[1])

    # binning - change title with low occurence to 'Misc' bin
    misc_titles = (df['Name'].value_counts() < min_occurence)
    df['Name'] = df['Name'].apply(lambda x: 'Misc' if misc_titles.loc[x] == True else x)

    # filling empty age with random age in 1 standard deviation
    df['Age'] = df['Age'].fillna(random.randrange( int(df['Age'].mean() - df['Age'].std()), int(df['Age'].mean() + df['Age'].std())))

    # binning - age column
    bins = [0, 2, 18, 35, 65, np.inf]
    names = ['1', '2', '3', '4', '5'] 
    df['Age'] = pd.cut(df['Age'], bins, labels=names)
    
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

    df['Family'] = np.where(df['SibSp'] > 0 | df['Parch'], 1, 0)

    # generating new embeddings from bin columns
    one_hot_sex = pd.get_dummies(df['Sex'], prefix='sex')
    one_hot_class = pd.get_dummies(df['Pclass'], prefix='pclass')
    one_hot_family = pd.get_dummies(df['Family'], prefix='family')
    one_hot_age = pd.get_dummies(df['Age'], prefix='age')
    one_hot_name = pd.get_dummies(df['Age'], prefix='name')

    df = df.join(one_hot_sex)
    df = df.join(one_hot_class)
    df = df.join(one_hot_age)
    df = df.join(one_hot_name)
    df = df.join(one_hot_family)

    df.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'Sex', 'Pclass', 'SibSp', 'Parch', 'PassengerId', 'Age'], axis=1, inplace=True)

    return df

test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')
submission = pd.read_csv('gender_submission.csv')

train_data = prepare_df(train_data)
test_data = prepare_df(test_data)

train, test = train_test_split(train_data, test_size=0.05, random_state=42)

y_train = train['Survived']
y_test = test['Survived']

train.drop('Survived', axis=1, inplace=True)
test.drop('Survived', axis=1, inplace=True)

pipeline = Pipeline([
    #('scaler', StandardScaler()),
    ('clf', LogisticRegression())
    #('clf', GradientBoostingClassifier())
    #('clf', RandomForestClassifier())
    #('clf', MLPClassifier(hidden_layer_sizes=(100, 50, 25, 10, 2), alpha=0.0001, activation='tanh', shuffle=True, early_stopping=True, verbose=True))
])

pipeline.fit(train, y_train)

y_pred = pipeline.predict(test)

results = classification_report(y_test, y_pred)
print(results)

y_pred = pipeline.predict(test_data)

submission['Survived'] = y_pred

submission.to_csv('gender_submission_final.csv', index=False)
