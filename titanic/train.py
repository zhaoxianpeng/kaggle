# -*- coding: utf-8 -*-
from sklearn import linear_model
import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from feature_eng import *

if __name__ == '__main__':
    data_train = pd.read_csv("train.csv")
    df,rfr,scaler,age_scale_param,fare_scale_param = feature_process(data_train)
    
    # 用正则取出我们要的属性值
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    # print(df)
    train_np = train_df.as_matrix()
    # print(train_np)
    
    # y即Survival结果
    y = train_np[:, 0]
    
    # X即特征属性值
    X = train_np[:, 1:]
    
    # fit到LogisticRegression之中
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)
    
    print(clf)
    
    data_test=pd.read_csv("test.csv")
    data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
    tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[data_test.Age.isnull()].as_matrix()
    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges
    
    data_test = set_Cabin_type(data_test)
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')
    
    df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)
    df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)    
    
    test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(test)
    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
    result.to_csv("logistic_regression_predictions.csv", index=False)
    