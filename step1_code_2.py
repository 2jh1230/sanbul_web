import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 한글 폰트 설정 (Windows 기준)
mpl.rc('font', family='Malgun Gothic')  # Windows용 기본 한글 폰트
mpl.rc('axes', unicode_minus=False)     # 마이너스(-) 깨짐 방지

# 데이터 불러오기
fires = pd.read_csv('./sanbul2district-divby100.csv')
fires['burned_area'] = np.log(fires['burned_area'] + 1)

# 1-5 Scikit-Learn의 train_test_split을 이용하여 training/test set 분리 / Test set 비율 확인하기

# (1) 단순 무작위 분리
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)
print("[무작위 분리] 테스트셋 일부 미리보기(2020810057 이정환):")
print(test_set.head())

# month 분포 히스토그램
fires["month"].hist()
plt.title('Month 분포 히스토그램(2020810057 이정환)')
plt.xlabel('Month')
plt.ylabel('빈도수')
plt.show()

# (2) StratifiedShuffleSplit (층화 샘플링)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]

print("\n[Test Set - Month 비율](2020810057 이정환)")
print(strat_test_set["month"].value_counts() / len(strat_test_set))

print("\n[전체 데이터 - Month 비율](2020810057 이정환)")
print(fires["month"].value_counts() / len(fires))

# 1-6 Pandas scatter_matrix() 함수를 이용하여 4개 이상의 특성에 대해 matrix 출력하기

# 특성 리스트
attributes = ["avg_temp", "max_temp", "max_wind_speed", "avg_wind", "burned_area"]

# scatter_matrix 그리기
scatter_matrix(fires[attributes], figsize=(12, 8), alpha=0.5)
plt.suptitle('특성 간 관계 (scatter matrix) (2020810057 이정환)')
plt.show()

# 1-7 지역별로 ‘burned_area’에 대해 plot 하기: 원의 반경은 max_temp(옵션 s), 컬러는 burned_area(옵션 c)를 의미

# scatter plot
fires.plot(kind="scatter", 
           x="longitude", 
           y="latitude", 
           alpha=0.4,
           s=fires["max_temp"],                # 점 크기 (원의 반경) = max_temp
           c=fires["burned_area"],              # 색깔 = burned_area
           cmap=plt.get_cmap("jet"),            # jet 컬러맵 적용
           colorbar=True,
           figsize=(12,8),
           label="max_temp"
)

plt.title('지역별 산불 발생 (원의 크기: 최대 온도, 색상: 산불 면적)(2020810057 이정환)')
plt.xlabel('경도 (longitude)')
plt.ylabel('위도 (latitude)')
plt.legend()
plt.show()

# 1-8 카테고리형 특성 month, day에 대해 OneHotEncoder()를 이용한 인코딩/출력

# (1) 레이블과 특성 나누기
fires = strat_train_set.drop(["burned_area"], axis=1)  # label 제거
fires_labels = strat_train_set["burned_area"].copy()   # label 저장

# (2) 수치형 특성과 범주형 특성 분리
fires_num = fires.drop(["month", "day"], axis=1)       # 수치형 특성만 남김
fires_cat = fires[["month", "day"]]                    # 범주형 특성만 추출

# (3) OneHotEncoder로 카테고리형 데이터 변환
encoder = OneHotEncoder()
fires_cat_1hot = encoder.fit_transform(fires_cat)

# (4) 결과 확인
print("2020810057 이정환")
print(fires_cat_1hot.toarray())  # 희소 행렬(Sparse matrix)이므로 toarray()로 변환해서 출력
print(fires_cat_1hot.shape)      # (샘플 수, 변환된 특성 수)

# 1-9 Scikit-Learn의 Pipeline, StandardScaler를 이용하여 카테고리형 특성을 인코딩한 training set 생성하기

# 수치형 특성과 범주형 특성 나누기
num_attribs = list(fires_num)      # 수치형 특성들
cat_attribs = ["month", "day"]      # 범주형 특성들

print("\n\n########################################################################")
print("(2020810057 이정환)Now let's build a pipeline for preprocessing the numerical attributes:")

# (1) 수치형 파이프라인: 표준화(StandardScaler) 적용
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

# (2) 전체 파이프라인: 수치형 + 범주형 특성 처리
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),        # 수치형 특성에는 StandardScaler 적용
    ("cat", OneHotEncoder(), cat_attribs),      # 범주형 특성에는 OneHotEncoder 적용
])

# (3) fires 데이터셋에 파이프라인 적용
fires_prepared = full_pipeline.fit_transform(fires)

print("\nfires_prepared 결과 shape: ", fires_prepared.shape)
