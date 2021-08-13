# 1. Import the library
import numpy as np
import pandas as pd
import tensorflow as tf

# 2. Load data
# 2.1 Target Data
kYm1 = pd.read_csv("Ys1_M1_Df.csv")
kYm2 = pd.read_csv("Ys2_M2_Df.csv")
kYm3 = pd.read_csv("Ys3_M3_Df.csv")
kYm4 = pd.read_csv("Yss1_M4_Df.csv")
kYm5 = pd.read_csv("Yss2_M5_Df.csv")
kYm6 = pd.read_csv("Yss3_M6_Df.csv")


# 2.2 Convert data to array
kYam1 = np.array(kYm1)
kYam2 = np.array(kYm2)
kYam3 = np.array(kYm3)
kYam4 = np.array(kYm4)
kYam5 = np.array(kYm5)
kYam6 = np.array(kYm6)

# 2.3 Transpose target data
yTm1 = np.transpose(kYam1)
yTm2 = np.transpose(kYam2)
yTm3 = np.transpose(kYam3)
yTm4 = np.transpose(kYam4)
yTm5 = np.transpose(kYam5)
yTm6 = np.transpose(kYam6)

# 3.1 Load Test data
pXt = pd.read_csv("X_test_Df.csv")

# 3.2 Convert data to array
pXta = np.array(pXt)
# 3.3 Transpose target data
xTest = np.transpose(pXta)

# 3.4 Normalize the data
# 3.4.1 formula to normalize
def norm(x):
    return (x - x.min()) / (x.max() - x.min())


xTn = norm(xTest)
xTn = xTn.reshape(xTn.shape[0], xTn.shape[1], 1)

# 4. Recreate the exact same model, including its weights and the optimizer
new_model1 = tf.keras.models.load_model('CNN_M1_3.h5')
new_model2 = tf.keras.models.load_model('CNN_M2_3.h5')
new_model3 = tf.keras.models.load_model('CNN_M3_3.h5')
new_model4 = tf.keras.models.load_model('CNN_M4_3.h5')
new_model5 = tf.keras.models.load_model('CNN_M5_3.h5')
new_model6 = tf.keras.models.load_model('CNN_M6_3.h5')


# 5. Prediction the test data with models
preM1 = new_model1.predict(xTn)
preM2 = new_model2.predict(xTn)
preM3 = new_model3.predict(xTn)
preM4 = new_model4.predict(xTn)
preM5 = new_model5.predict(xTn)
preM6 = new_model6.predict(xTn)

# 6. Anti normalize
resM1 = preM1 + yTm1.min()
resM2 = preM2 + yTm2.min()
resM3 = preM3 + yTm3.min()
resM4 = preM4 + yTm4.min()
resM5 = preM5 + yTm5.min()
resM6 = preM6 + yTm6.min()

# 7. Save data to execl field

df1 = pd.DataFrame(np.transpose(resM1))
df1.to_csv('Res_CNN_M1.csv')

df2 = pd.DataFrame(np.transpose(resM2))
df2.to_csv('Res_CNN_M2.csv')

df3 = pd.DataFrame(np.transpose(resM3))
df3.to_csv('Res_CNN_M3.csv')

df4 = pd.DataFrame(np.transpose(resM4))
df4.to_csv('Res_CNN_M4.csv')

df5 = pd.DataFrame(np.transpose(resM5))
df5.to_csv('Res_CNN_M5.csv')

df6 = pd.DataFrame(np.transpose(resM6))
df6.to_csv('Res_CNN_M6.csv')




