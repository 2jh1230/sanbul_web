import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 한글 폰트 설정 (Windows 기준)
mpl.rc('font', family='Malgun Gothic')  # Windows용 기본 한글 폰트
mpl.rc('axes', unicode_minus=False)     # 마이너스(-) 깨짐 방지



# 1-1 Data 불러오기
#2020810057 이정환
fires = pd.read_csv("./sanbul2district-divby100.csv", sep=",")
fires['burned_area'] = np.log(fires['burned_area'] + 1)
print(fires.columns)

# 1-2 fires.head(), fires.info(), fires.describe(), 카테고리형 특성 month, day에 대해 value_counts() 출력하기
#2020810057 이정환
print("2020810057 이정환")
# 데이터 상위 5개 미리보기
print("---fires.head()---")
print(fires.head())
# 데이터 기본 요약
print("---fires.info()---")
print(fires.info())
# 수치형 데이터 통계 요약
print("---fires.describe()---")
print(fires.describe())
# month별 데이터 개수
print("---month.value_counts()---")
print(fires['month'].value_counts())
# day별 데이터 개수
print("---day.value_counts()---")
print(fires['day'].value_counts())

# 1-3 데이터 시각화 (예: 히스토그램 또는 2개 속성 비교 등)
#2020810057 이정환
# 1. 전체 수치형 컬럼 히스토그램
fig = fires.hist(bins=50, figsize=(20,15))
plt.suptitle('2020810057 이정환', fontsize=20)  # suptitle 사용!
plt.show()

# 2. 두 특성 간 관계 시각화 (scatter plot)
fires.plot(kind="scatter", x="avg_temp", y="burned_area", alpha=0.5)
plt.title('2020810057 이정환')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Burned Area (log scale)')
plt.show()

# 1-4 특성 burned_area 왜곡 현상(아래 왼쪽 그림) 개선을 위해 로그 함수(y=ln(burned_area+1))를 이용한 변환
#2020810057 이정환

# 변환 후 burned_area 분포 확인
fires['burned_area'].hist(bins=50)
plt.title('변환된 burned_area 분포 (로그 적용) 2020810057 이정환')
plt.xlabel('log(burned_area + 1)')
plt.ylabel('빈도수')
plt.show()