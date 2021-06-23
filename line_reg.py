import pandas
from sklearn import linear_model
import joblib

filename1 = 'line_reg_pre.sav'
filename2 = 'line_reg_hum.sav'


df = pandas.read_csv("weather.csv")

X = df['temperature']
y = df['pressure']
z = df['humidity']
X = X.values.reshape(-1,1)

regr1 = linear_model.LinearRegression()
regr2 = linear_model.LinearRegression()
regr1.fit(X, y)
regr2.fit(X, z)
joblib.dump(regr1, filename1)
joblib.dump(regr2, filename2)

t = 9.4
loaded_model1 = joblib.load(filename1)
loaded_model2 = joblib.load(filename2)
predictedpre = loaded_model1.predict([[t]])
predictedhum = loaded_model2.predict([[t]])

print(predictedpre)
print(predictedhum)