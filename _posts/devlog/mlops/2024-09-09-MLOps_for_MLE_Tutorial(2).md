---
layout: post
title: MLOps) MLOps for MLE Tutorial (2) - Model Development
description: >
  MLOps for MLE Tutorial Model Development Part
sitemap: false
categories: [devlog, mlops]
hide_last_modified: true
related_posts:
  - ./2024-09-01-WSL2_WindowDocker.md
  - ./2024-09-08-MLOps_for_MLE_Tutorial(1).md
---

# MLOps) MLOps for MLE Tutorial (2) - Model Development

![database workflow](./../../../images/2024-09-09-MLOps_for_MLE_Tutorial(2)/model-development-2-f3f112ac4bf173365572e764bf7dc750.png)

(1)번까지의 과정에서 Data Generator를 생성해서 Postgres server에 연결하는 것까지 마쳤다.

이번 튜토리얼에서는 Query Data를 기반으로 데이터를 추출하고 `scikit-learn`으로 간단하게 모델 학습 후 모델을 저장하는 것 까지 한다.

추가적으로 `data-generator`에서 생성해서 postgreSQL에 삽입한 데이터를 기반으로 불러와서 모델을 생성하는 과정까지 한다.

아래는 train 과정과 `joblib`을 통해서 모델까지 저장하는 코드

~~~python
#//file: "db_train.py"
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import psycopg2
import pandas as pd
from config import DB_config
# db connect & get data

db_connector = psycopg2.connect(
        user=DB_config.DB_USER,
        password=DB_config.DB_PASSWORD,
        host="localhost",
        database=DB_config.DB_DATABASE,
        port=DB_config.DB_PORT
    )

df = pd.read_sql("SELECT * FROM iris_data ORDER BY id DESC LIMIT 100", db_connector)
X = df.drop(["id", "timestamp", "target"], axis="columns")
y = df["target"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2024)

model_pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
model_pipeline.fit(X_train, y_train)

train_pred = model_pipeline.predict(X_train)
valid_pred = model_pipeline.predict(X_valid)

train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)

print(f"Train Accuracy : {train_acc}")
print(f"Valid Accuracy : {valid_acc}")

# Pipline Model Save
joblib.dump(model_pipeline, "model_pipeline.joblib")


# save csv
df.to_csv("data.csv", index=False)
~~~







[MLOps for MLE](https://mlops-for-mle.github.io/tutorial/docs/intro)





