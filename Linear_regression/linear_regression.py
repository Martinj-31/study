from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

Boston_house = pd.read_csv("/Users/mingyucheon/Desktop/dataset/Boston_house.csv")
# x1 = Boston_house[["CRIM", "RAD"]]
x2 = Boston_house[["CRIM", "RAD", "RM", "AGE"]]
# x3 = Boston_house[["CRIM", "RAD", "RM", "AGE", "DIS", "INDUS", "LSTAT", "NOX", "PTRATIO", "TAX"]]
y = Boston_house[["Target"]]

# for i in [x1, x2, x3]:
#     x_train, x_test, y_train, y_test = train_test_split(i, y, test_size=0.2, random_state=42)

#     regression = LinearRegression()
#     regression.fit(x_train, y_train)
#     y_predict = regression.predict(x_test)

#     plt.scatter(y_test, y_predict, alpha=0.4)
#     plt.xlabel("Actual Price")
#     plt.ylabel("Predicted Price")
#     plt.title("Linear Regression")
#     plt.show()

x_train, x_test, y_train, y_test = train_test_split(x2, y, test_size=0.2, random_state=42)

regression = LinearRegression()
regression.fit(x_train, y_train)
y_predict = regression.predict(x_test)

print(regression.coef_)

plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression")
plt.show()
plt.scatter(Boston_house[["CRIM"]], Boston_house[["Target"]], alpha=0.4)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("CRIM")
plt.show()
plt.scatter(Boston_house[["RAD"]], Boston_house[["Target"]], alpha=0.4)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("RAD")
plt.show()
plt.scatter(Boston_house[["RM"]], Boston_house[["Target"]], alpha=0.4)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("RM")
plt.show()
plt.scatter(Boston_house[["AGE"]], Boston_house[["Target"]], alpha=0.4)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("AGE")
plt.show()
