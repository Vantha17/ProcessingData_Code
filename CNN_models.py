# 1. Import the library
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import LeaveOneOut
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout     # nodes
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dense       # layer
from tensorflow.keras import layers, models

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


# 5 Label the target output
# 5.1 formula to label the target output
def label(y):
    z = (y - y.min())
    return np.round(z)


xNm1 = norm(xTm1)
xNm2 = norm(xTm2)
xNm3 = norm(xTm3)
xNm4 = norm(xTm4)
xNm5 = norm(xTm5)
xNm6 = norm(xTm6)

yLm2 = label(yTm2)
yLm1 = label(yTm1)
yLm3 = label(yTm3)
yLm4 = label(yTm4)
yLm5 = label(yTm5)
yLm6 = label(yTm6)

# convert 2d to 3d
xNm1 = xNm1.reshape(xNm1.shape[0], xNm1.shape[1], 1)
xNm2 = xNm2.reshape(xNm2.shape[0], xNm2.shape[1], 1)
xNm3 = xNm3.reshape(xNm3.shape[0], xNm3.shape[1], 1)
xNm4 = xNm4.reshape(xNm4.shape[0], xNm4.shape[1], 1)
xNm5 = xNm5.reshape(xNm5.shape[0], xNm5.shape[1], 1)
xNm6 = xNm6.reshape(xNm6.shape[0], xNm6.shape[1], 1)

yLm1 = yLm1.reshape(yLm1.shape[0], yLm1.shape[1], 1)
yLm2 = yLm2.reshape(yLm2.shape[0], yLm2.shape[1], 1)
yLm3 = yLm3.reshape(yLm3.shape[0], yLm3.shape[1], 1)
yLm4 = yLm4.reshape(yLm4.shape[0], yLm4.shape[1], 1)
yLm5 = yLm5.reshape(yLm5.shape[0], yLm5.shape[1], 1)
yLm6 = yLm6.reshape(yLm6.shape[0], yLm6.shape[1], 1)


# create loocv procedure

cv = LeaveOneOut()

# CNN_M1
for train_ix, test_ix in cv.split(xNm1,yLm1):
    # split data
    X_train, X_test = xNm1[train_ix, :], xNm1[test_ix, :]
    y_train, y_test = yLm1[train_ix], yLm1[test_ix]
    # 7. Create the model
    cnn_M1 = Sequential()
    cnn_M1.add(Conv1D(filters=32, kernel_size=3, activation='sigmoid', input_shape=(100,1)))
    cnn_M1.add(MaxPooling1D(pool_size=2))
    cnn_M1.add(Dropout(0.5))

    cnn_M1.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    cnn_M1.add(MaxPooling1D(pool_size=2))
    cnn_M1.add(Dropout(0.5))

    cnn_M1.add(Flatten())
    cnn_M1.add(Dense(100, activation='elu'))
    cnn_M1.add(layers.Dense(100, activation='linear'))

    # 8. Add Optimizer to minimize the loss
    cnn_M1.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.007), metrics=['mse'])
    cnn_M1.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=50, batch_size=32, verbose=1)
    cnn_M1.save('CNN_M1.h5')

# CNN_M2
for train_ix, test_ix in cv.split(xNm2,yLm2):
    # split data
    X_train, X_test = xNm2[train_ix, :], xNm2[test_ix, :]
    y_train, y_test = yLm2[train_ix], yLm2[test_ix]
    # Create the model
    cnn_M2 = Sequential()
    cnn_M2.add(Conv1D(filters=32, kernel_size=3, activation='sigmoid', input_shape=(100,1)))
    cnn_M2.add(MaxPooling1D(pool_size=2))
    cnn_M2.add(Dropout(0.5))

    cnn_M2.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    cnn_M2.add(MaxPooling1D(pool_size=2))
    cnn_M2.add(Dropout(0.5))

    cnn_M2.add(Flatten())
    cnn_M2.add(Dense(100, activation='elu'))
    cnn_M2.add(layers.Dense(100, activation='linear'))


    cnn_M2.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.007), metrics=['mse'])
    cnn_M2.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=50, batch_size=32, verbose=1)
    cnn_M2.save('CNN_M2.h5')

# CNN_M3
for train_ix, test_ix in cv.split(xNm3,yLm3):
    # split data
    X_train, X_test = xNm3[train_ix, :], xNm3[test_ix, :]
    y_train, y_test = yLm3[train_ix], yLm3[test_ix]
    # 7. Create the model
    cnn_M3 = Sequential()
    cnn_M3.add(Conv1D(filters=32, kernel_size=3, activation='sigmoid', input_shape=(100,1)))
    cnn_M3.add(MaxPooling1D(pool_size=2))
    cnn_M3.add(Dropout(0.5))

    cnn_M3.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    cnn_M3.add(MaxPooling1D(pool_size=2))
    cnn_M3.add(Dropout(0.5))

    cnn_M3.add(Flatten())
    cnn_M3.add(Dense(100, activation='elu'))
    cnn_M3.add(layers.Dense(100, activation='linear'))

    # 8. Add Optimizer to minimize the loss
    cnn_M3.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.007), metrics=['mse'])
    cnn_M3.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=50, batch_size=32, verbose=1)
    cnn_M3.save('CNN_M3.h5')

# CNN_M4
for train_ix, test_ix in cv.split(xNm4,yLm4):
    # split data
    X_train, X_test = xNm4[train_ix, :], xNm4[test_ix, :]
    y_train, y_test = yLm4[train_ix], yLm4[test_ix]
    # 7. Create the model
    cnn_M4 = Sequential()
    cnn_M4.add(Conv1D(filters=32, kernel_size=3, activation='sigmoid', input_shape=(100,1)))
    cnn_M4.add(MaxPooling1D(pool_size=2))
    cnn_M4.add(Dropout(0.5))

    cnn_M4.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    cnn_M4.add(MaxPooling1D(pool_size=2))
    cnn_M4.add(Dropout(0.5))

    cnn_M4.add(Flatten())
    cnn_M4.add(Dense(100, activation='elu'))
    cnn_M4.add(layers.Dense(100, activation='linear'))

    # 8. Add Optimizer to minimize the loss
    cnn_M4.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.007), metrics=['mse'])
    cnn_M4.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=50, batch_size=32, verbose=1)
    cnn_M4.save('CNN_M4.h5')


# CNN_M5
for train_ix, test_ix in cv.split(xNm5,yLm5):
    # split data
    X_train, X_test = xNm5[train_ix, :], xNm5[test_ix, :]
    y_train, y_test = yLm5[train_ix], yLm5[test_ix]
    # 7. Create the model
    cnn_M5 = Sequential()
    cnn_M5.add(Conv1D(filters=32, kernel_size=3, activation='sigmoid', input_shape=(100,1)))
    cnn_M5.add(MaxPooling1D(pool_size=2))
    cnn_M5.add(Dropout(0.5))

    cnn_M5.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    cnn_M5.add(MaxPooling1D(pool_size=2))
    cnn_M5.add(Dropout(0.5))

    cnn_M5.add(Flatten())
    cnn_M5.add(Dense(100, activation='elu'))
    cnn_M5.add(layers.Dense(100, activation='linear'))

    # 8. Add Optimizer to minimize the loss
    cnn_M5.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.007), metrics=['mse'])
    cnn_M5.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=50, batch_size=32, verbose=1)
    cnn_M5.save('CNN_M5.h5')

# CNN_M6
for train_ix, test_ix in cv.split(xNm6,yLm6):
    # split data
    X_train, X_test = xNm6[train_ix, :], xNm6[test_ix, :]
    y_train, y_test = yLm6[train_ix], yLm6[test_ix]
    # 7. Create the model
    cnn_M6 = Sequential()
    cnn_M6.add(Conv1D(filters=32, kernel_size=3, activation='sigmoid', input_shape=(100,1)))
    cnn_M6.add(MaxPooling1D(pool_size=2))
    cnn_M6.add(Dropout(0.5))

    cnn_M6.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    cnn_M6.add(MaxPooling1D(pool_size=2))
    cnn_M6.add(Dropout(0.5))

    cnn_M6.add(Flatten())
    cnn_M6.add(Dense(100, activation='elu'))
    cnn_M6.add(layers.Dense(100, activation='linear'))

    # 8. Add Optimizer to minimize the loss
    cnn_M6.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.007), metrics=['mse'])
    cnn_M6.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=50, batch_size=32, verbose=1)
    cnn_M6.save('CNN_M6.h5')


