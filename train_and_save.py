import pandas as pd, pickle, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv("car data.csv")
df.replace({'Fuel_Type': {'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace=True)
df = pd.get_dummies(df, columns=['Seller_Type', 'Transmission'], drop_first=True)

X = df.drop(['Car_Name','Selling_Price'], axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler(); 
X_train_scaled = scaler.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/model.pkl","wb"))
pickle.dump(scaler, open("models/scaler.pkl","wb"))
pickle.dump(list(X.columns), open("models/feature_columns.pkl","wb"))

print("Saved model.pkl, scaler.pkl, feature_columns.pkl successfully!")
