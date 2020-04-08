
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("china_gdp.csv")
df.head(10)

plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))
plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y
 
xdata = (x_data-min(x_data))/(max(x_data)-min(x_data))
ydata = (y_data-min(y_data))/(max(y_data)-min(y_data))


beta_1 = 0.5
beta_2 = 0.1
#logistic function
Y_pred = sigmoid(xdata, beta_1 , beta_2)
#plot initial prediction against datapoints
plt.plot(xdata, Y_pred)
plt.plot(xdata, ydata, 'ro')


from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))


Y_pred = sigmoid(xdata, *popt)
plt.figure(figsize=(8,5))
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(xdata, Y_pred, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()




