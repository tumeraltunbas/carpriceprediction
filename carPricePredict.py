#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
#Data reading
df = pd.read_excel('merc.xlsx')
#Data understanding
result = df.describe() 
result = df.isnull() 
result = df.isnull().sum() 

#Colleration
result = df.corr()
sbn.distplot(df['price'])
#plt.show()

#Data cleaning
result = df.sort_values('price',ascending=False).head()
result = len(df) 
result = len(df) * 0.01
result = df.sort_values('price', ascending=False).iloc[131:]
df=result
df = df.drop('transmission',axis=1)

#Prepare for training
y = df['price'].values
x = df.drop('price',axis=1) 
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=10) 
scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Creating model
model = Sequential()
model.add(Dense(12,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(1)) #Output layer

model.compile(optimizer='adam', loss='mse')

#Training
model.fit(x=x_train, y=y_train, epochs=300,validation_data=(x_test,y_test),batch_size=250)

lossData = pd.DataFrame(model.history.history)
#lossData.plot() 

#Test
guessList= model.predict(x_test) 

mean_absolute_error(y_test,x_test) 

#Testing
newCar = df.drop('price',axis=1).iloc[2]
newCar = scaler.transform(newCar.values.reshape(-1,5))
model.predict(newCar)