import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


np.random.seed(0)
num_samples = 1000
mileage = np.random.randint(10000, 100000, size=num_samples)
age = np.random.randint(1, 20, size=num_samples)
price = 20000 + 100 * mileage - 500 * age + np.random.normal(0, 5000, size=num_samples)


car_data = pd.DataFrame({'Mileage': mileage, 'Age': age, 'Price': price})


print(car_data.head())


X = car_data[['Mileage', 'Age']]
y = car_data['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Prices')
plt.show()
