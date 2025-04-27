import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


# (1) 데이터 불러오기
fires = pd.read_csv("./sanbul2district-divby100.csv", sep=",")
fires['burned_area'] = np.log(fires['burned_area'] + 1)

# (2) StratifiedShuffleSplit을 사용해서 train/test 분리
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]

# (3) 레이블과 특성 나누기
fires_train = strat_train_set.drop(["burned_area"], axis=1)
fires_labels_train = strat_train_set["burned_area"].copy()

fires_test = strat_test_set.drop(["burned_area"], axis=1)
fires_labels_test = strat_test_set["burned_area"].copy()

# (4) 수치형/범주형 특성 나누기
num_attribs = list(fires_train.drop(["month", "day"], axis=1))
cat_attribs = ["month", "day"]

# (5) 전처리 파이프라인
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# (6) 전처리 데이터 준비
fires_prepared = full_pipeline.fit_transform(fires_train)
fires_test_prepared = full_pipeline.transform(fires_test)

# (7) train/validation 분리
X_train, X_valid, y_train, y_valid = train_test_split(
    fires_prepared, fires_labels_train, test_size=0.2, random_state=42
)

# (8) X_test 준비
X_test = fires_test_prepared
y_test = fires_labels_test

# (9) 랜덤 시드 고정 (재현성 확보)
np.random.seed(42)
tf.random.set_seed(42)

# (10) Keras 모델 만들기
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)  # 회귀니까 출력층은 1개 (활성화 함수 없음)
])

# (11) 모델 구조 요약
model.summary()

# (12) 모델 컴파일
model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.SGD(learning_rate=1e-3)
)

# (13) 모델 학습
history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_data=(X_valid, y_valid)
)

# (14) 모델 저장
model.save('fires_model.keras')

# (15) 모델 일부 예측 (테스트셋 3개 샘플)
X_new = X_test[:3]
predictions = np.round(model.predict(X_new), 2)

print("\n[2020810057 이정환] 모델 예측 결과 (테스트셋 일부):")
print(predictions)

# 학습한 full_pipeline 저장
joblib.dump(full_pipeline, 'full_pipeline.pkl')