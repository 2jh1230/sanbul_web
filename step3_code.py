import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import joblib

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# 랜덤 시드 고정
np.random.seed(42)

# Flask 앱 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

# 폼 클래스 정의
class LabForm(FlaskForm):
    longitude = StringField('longitude (예: -8.6)', validators=[DataRequired()])
    latitude = StringField('latitude (예: 41.8)', validators=[DataRequired()])
    month = StringField('month (1~12)', validators=[DataRequired()])
    day = StringField('day (0~7)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp (°C)', validators=[DataRequired()])
    max_temp = StringField('max_temp (°C)', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed (km/h)', validators=[DataRequired()])
    avg_wind = StringField('avg_wind (km/h)', validators=[DataRequired()])
    submit = SubmitField('Submit')

# 🔥 모델과 full_pipeline 불러오기
model = keras.models.load_model('fires_model.keras')
full_pipeline = joblib.load('full_pipeline.pkl')  # 반드시 test3.py에서 저장해놔야 함

# 루트(index) 페이지
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

# prediction 페이지
@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        # 폼 데이터 가져오기
        data = {
            'longitude': [float(form.longitude.data)],
            'latitude': [float(form.latitude.data)],
            'month': [form.month.data],
            'day': [form.day.data],
            'avg_temp': [float(form.avg_temp.data)],
            'max_temp': [float(form.max_temp.data)],
            'max_wind_speed': [float(form.max_wind_speed.data)],
            'avg_wind': [float(form.avg_wind.data)]
        }
        input_df = pd.DataFrame(data)

        # 전처리
        input_prepared = full_pipeline.transform(input_df)

        # 예측
        prediction = model.predict(input_prepared)
        
        # 로그 복원 + 단위 변환
        area_in_hectare = np.expm1(prediction[0][0])    # (1/100 헥타르 단위)
        area_in_m2 = np.round(area_in_hectare * 100, 2)  # m²로 변환

        return render_template('result.html', result=area_in_m2)
    return render_template('prediction.html', form=form)


# 서버 실행
if __name__ == '__main__':
    app.run(debug=True)
