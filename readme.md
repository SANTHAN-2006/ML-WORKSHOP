# SA1 ASSIGNMENT

## Objective 1 :
## To Create a scatter plot between cylinder vs Co2Emission (green color)
# Code :
```python
'''
Developed by : K SANTHAN KUMAR
Register Number : 212223240065
'''
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/content/FuelConsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission')
plt.show()
```
# Output :
![image](https://github.com/SANTHAN-2006/ML-WORKSHOP/assets/80164014/bfcf27e4-51fb-46ac-8d0e-35d24553be08)

## Objective 2 :
## Using scatter plot compare data   cylinder vs Co2Emission and Enginesize Vs Co2Emission using different colors
# Code :
```python
'''
Developed by : K SANTHAN KUMAR
Register Number : 212223240065
'''
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/content/FuelConsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='blue', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='red', label='Engine Size')
plt.xlabel('Cylinders/Engine Size')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission and Engine Size vs CO2 Emission')
plt.legend()
plt.show()
```
# Output :
![image](https://github.com/SANTHAN-2006/ML-WORKSHOP/assets/80164014/7ed071ee-c87a-4deb-9ebe-d04d2b547aa7)

## Objective 3 :
## Using scatter plot compare data   cylinder vs Co2Emission and Enginesize Vs Co2Emission and FuelConsumption_comb Co2Emission using different colors
# Code :
```python
'''
Developed by : K SANTHAN KUMAR
Register Number : 212223240065
'''
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/content/FuelConsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='blue', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='red', label='Engine Size')
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='orange', label='Fuel Consumption')
plt.xlabel('Cylinders/Engine Size/Fuel Consumption')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission, Engine Size vs CO2 Emission, and Fuel Consumption vs CO2 Emission')
plt.legend()
plt.show()
```
# Output :
![image](https://github.com/SANTHAN-2006/ML-WORKSHOP/assets/80164014/96e02597-153b-4486-aa89-de6fa267c6b7)

## Objective 4 :
## Train your model with independent variable as cylinder and dependent variable as Co2Emission
# Code :
```python
'''
Developed by : K SANTHAN KUMAR
Register Number : 212223240065
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/content/FuelConsumption.csv')

X_cylinder = df[['CYLINDERS']]
y_co2 = df['CO2EMISSIONS']

X_train_cylinder, X_test_cylinder, y_train_cylinder, y_test_cylinder = train_test_split(X_cylinder, y_co2, test_size=0.2, random_state=42)

model_cylinder = LinearRegression()
model_cylinder.fit(X_train_cylinder, y_train_cylinder)

```
# Output :
![image](https://github.com/SANTHAN-2006/ML-WORKSHOP/assets/80164014/cbf759a5-5a23-4899-a19c-d721d6c811bf)
## Objective 5 :
## Train another model with independent variable as FuelConsumption_comb and dependent variable as Co2Emission
# Code :
```python
'''
Developed by : K SANTHAN KUMAR
Register Number : 212223240065
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/content/FuelConsumption.csv')

X_fuel = df[['FUELCONSUMPTION_COMB']]
y_co2 = df['CO2EMISSIONS']

X_train_fuel, X_test_fuel, y_train_fuel, y_test_fuel = train_test_split(X_fuel, y_co2, test_size=0.2, random_state=42)

model_fuel = LinearRegression()
model_fuel.fit(X_train_fuel, y_train_fuel)
```
# Output :
![image](https://github.com/SANTHAN-2006/ML-WORKSHOP/assets/80164014/096561d4-4b43-463d-983f-3063fa69a15f)

## Objective 6 :
## Train your model on different train test ratio and train the models and note down their accuracies
# Code :
```python
'''
Developed by : K SANTHAN KUMAR
Register Number : 212223240065
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('/content/FuelConsumption.csv')

X_cylinder = df[['CYLINDERS']]
y_co2 = df['CO2EMISSIONS']

ratios = [0.2, 0.3, 0.4, 0.5]

for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X_cylinder, y_co2, test_size=ratio, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Train-Test Ratio: {1-ratio}:{ratio} - Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}')
```
# Output :
![image](https://github.com/SANTHAN-2006/ML-WORKSHOP/assets/80164014/b26d872c-7671-43c8-bdb4-2362cef0696b)
# Result: Successfully executed all the programs
## Programs Developed by : K SANTHAN KUMAR
## Register Number : 212223240065
## Dept : AIML













