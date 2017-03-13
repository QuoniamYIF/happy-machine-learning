import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch','Fare']

X_train = train[selected_features]
X_test = test[selected_features]

y_train = train['Survived']
y_test = train['Survived']

X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)

X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_train['Fare'].fillna(X_train['Fare'].mean(), inplace=True)

X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)

dict_vec=DictVectorizer(sparse=False)
X_train=dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test=dict_vec.fit_transform(X_test.to_dict(orient='record'))

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)
rfc_submission = pd.DataFrame({
    'PassengerId': X_test['passenger'], 'Survived': rfc_y_predict
})
rfc_submission.to_csv('./rfc_submission.csv', index=False)



