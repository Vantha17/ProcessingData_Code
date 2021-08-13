# 1. Import the library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

 #2. Load data
# 2.1 Input data
pXm1 = pd.read_csv("Xs1_M1_Df.csv")
pXm2 = pd.read_csv("Xs2_M2_Df.csv")
pXm3 = pd.read_csv("Xs3_M3_Df.csv")
pXm4 = pd.read_csv("Xss1_M4_Df.csv")
pXm5 = pd.read_csv("Xss2_M5_Df.csv")
pXm6 = pd.read_csv("Xss3_M6_Df.csv")


# 2.2 Target Data
kYm1 = pd.read_csv("Ys1_M1_Df.csv")
kYm2 = pd.read_csv("Ys2_M2_Df.csv")
kYm3 = pd.read_csv("Ys3_M3_Df.csv")
kYm4 = pd.read_csv("Yss1_M4_Df.csv")
kYm5 = pd.read_csv("Yss2_M5_Df.csv")
kYm6 = pd.read_csv("Yss3_M6_Df.csv")

# 3. Pre-processing data
# 3.1 Convert data to array
pXam1 = np.array(pXm1)
pXam2 = np.array(pXm2)
pXam3 = np.array(pXm3)
pXam4 = np.array(pXm4)
pXam5 = np.array(pXm5)
pXam6 = np.array(pXm6)

kYam1 = np.array(kYm1)
kYam2 = np.array(kYm2)
kYam3 = np.array(kYm3)
kYam4 = np.array(kYm4)
kYam5 = np.array(kYm5)
kYam6 = np.array(kYm6)


# 3.2 Transpose the matrix
xTm1 = np.transpose(pXam1)
xTm2 = np.transpose(pXam2)
xTm3 = np.transpose(pXam3)
xTm4 = np.transpose(pXam4)
xTm5 = np.transpose(pXam5)
xTm6 = np.transpose(pXam6)


yTm1 = np.transpose(kYam1)
yTm2 = np.transpose(kYam2)
yTm3 = np.transpose(kYam3)
yTm4 = np.transpose(kYam4)
yTm5 = np.transpose(kYam5)
yTm6 = np.transpose(kYam6)




# 4. Normalize the data
# 4.1 formula to normalize
def norm(x):
    return (x - x.min()) / (x.max() - x.min())


xNm1 = norm(xTm1)
xNm2 = norm(xTm2)
xNm3 = norm(xTm3)
xNm4 = norm(xTm4)
xNm5 = norm(xTm5)
xNm6 = norm(xTm6)



# 5 Label the target output
# 5.1 formula to label the target output
def label(y):
    z = (y - y.min())
    return np.round(z)


yLm1 = label(yTm1)
yLm2 = label(yTm2)
yLm3 = label(yTm3)
yLm4 = label(yTm4)
yLm5 = label(yTm5)
yLm6 = label(yTm6)

# create loocv procedure
cv = LeaveOneOut()

# ANN_M1
for train_ix, test_ix in cv.split(xNm1,yLm1):
    # split data
    X_train, X_test = xNm1[train_ix, :], xNm1[test_ix, :]
    y_train, y_test = yLm1[train_ix], yLm1[test_ix]
    # Create the model
    model1 = Sequential([
        Dense(units=250, input_dim=X_train.shape[1], activation='sigmoid'),
        Dense(units=150, activation='relu'),
        Dense(X_train.shape[1], activation='elu')
    ])
    # Add Optimizer to minimize the loss
    model1.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.007), metrics=['mse'])
    model1.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=50)
    # save the models
    model1.save('ANN_M1.h5')


# ANN_M2
for train_ix, test_ix in cv.split(xNm2,yLm2):
    # split data
    X_train, X_test = xNm2[train_ix, :], xNm2[test_ix, :]
    y_train, y_test = yLm2[train_ix], yLm2[test_ix]
    # Create the model
    model2 = Sequential([
        Dense(units=250, input_dim=X_train.shape[1], activation='sigmoid'),
        Dense(units=150, activation='relu'),
        Dense(X_train.shape[1], activation='elu')
    ])
    # Add Optimizer to minimize the loss
    model2.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.007), metrics=['mse'])
    model2.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=50)
    # save the models
    model2.save('ANN_M2.h5')


# ANN_M3
for train_ix, test_ix in cv.split(xNm3,yLm3):
    # split data
    X_train, X_test = xNm3[train_ix, :], xNm3[test_ix, :]
    y_train, y_test = yLm3[train_ix], yLm3[test_ix]
    # Create the model
    model3 = Sequential([
        Dense(units=250, input_dim=X_train.shape[1], activation='sigmoid'),
        Dense(units=150, activation='relu'),
        Dense(X_train.shape[1], activation='elu')
    ])
    # Add Optimizer to minimize the loss
    model3.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.007), metrics=['mse'])
    model3.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=50)
    # save the models
    model3.save('ANN_M3.h5')


# ANN_M4
for train_ix, test_ix in cv.split(xNm4,yLm4):
    # split data
    X_train, X_test = xNm4[train_ix, :], xNm4[test_ix, :]
    y_train, y_test = yLm4[train_ix], yLm4[test_ix]
    # Create the model
    model4 = Sequential([
        Dense(units=250, input_dim=X_train.shape[1], activation='sigmoid'),
        Dense(units=150, activation='relu'),
        Dense(X_train.shape[1], activation='elu')
    ])
    # Add Optimizer to minimize the loss
    model4.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.007), metrics=['mse'])
    model4.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=50)
    # save the models
    model4.save('ANN_M4.h5')


# ANN_M5
for train_ix, test_ix in cv.split(xNm5,yLm5):
    # split data
    X_train, X_test = xNm5[train_ix, :], xNm5[test_ix, :]
    y_train, y_test = yLm5[train_ix], yLm5[test_ix]
    # Create the model
    model5 = Sequential([
        Dense(units=250, input_dim=X_train.shape[1], activation='sigmoid'),
        Dense(units=150, activation='relu'),
        Dense(X_train.shape[1], activation='elu')
    ])
    # 8. Add Optimizer to minimize the loss
    model5.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.007), metrics=['mse'])
    model5.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=50)
    # 11. save the models
    model5.save('ANN_M5.h5')


# ANN_M6
for train_ix, test_ix in cv.split(xNm6,yLm6):
    # split data
    X_train, X_test = xNm6[train_ix, :], xNm6[test_ix, :]
    y_train, y_test = yLm6[train_ix], yLm6[test_ix]
    # Create the model
    model6 = Sequential([
        Dense(units=250, input_dim=X_train.shape[1], activation='sigmoid'),
        Dense(units=150, activation='relu'),
        Dense(X_train.shape[1], activation='elu')
    ])
    # Add Optimizer to minimize the loss
    model6.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.007), metrics=['mse'])
    model6.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=50)
    # save the models
    model6.save('ANN_M6.h5')




