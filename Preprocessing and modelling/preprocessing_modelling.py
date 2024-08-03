import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load and filter data
final_df = pd.read_csv("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/final_set.csv")
final_df = final_df[final_df["TypeOfSale"] == "residential_sale"]

final_df_apt = final_df[final_df["TypeOfProperty"] == "Apartment"]
final_df_house = final_df[final_df["TypeOfProperty"] == "House"]

relevant_columns_apt = [
    'LivingArea', 'BedroomCount', 'SwimmingPool', 'Density', 'Median_revenue', 'ConstructionYear', 'PEB_numerical', 
    'StateOfBuilding_numerical', 'BathroomCount', 'Kitchen_numerical', 'Terrace', 'Garden', 'Province_numerical'
]

relevant_columns_house = [
    'LivingArea', 'SurfaceOfPlot', 'BedroomCount', 'SwimmingPool', 'Density', 'Median_revenue', 'NumberOfFacades', 
    'ConstructionYear', 'PEB_numerical', 'StateOfBuilding_numerical', 'BathroomCount', 'Kitchen_numerical', 
    'Terrace', 'Garden', 'Province_numerical'
]

# Save copies of house and apartment dataframes in CSV format with relevant columns
final_df_apt[relevant_columns_apt].to_csv("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/apartments.csv", index=False)
final_df_house[relevant_columns_house].to_csv("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/houses.csv", index=False)

# Function to preprocess and split data
def preprocess_and_split_data(df, relevant_columns):
    X = df.drop(columns=["Price"])
    y = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train[relevant_columns]
    X_test = X_test[relevant_columns]
    pipeline = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5))])
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    return X_train_processed, X_test_processed, y_train, y_test

# Preprocess and split apartment data
X_apt_train_processed, X_apt_test_processed, y_apt_train, y_apt_test = preprocess_and_split_data(
    final_df_apt, relevant_columns_apt)

# Preprocess and split house data
X_house_train_processed, X_house_test_processed, y_house_train, y_house_test = preprocess_and_split_data(
    final_df_house, relevant_columns_house)

# Function to train and save model
def train_and_save_model(X_train, y_train, model_params, filepath):
    model = XGBRegressor(**model_params, random_state=42)
    model.fit(X_train, y_train)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    return model

# Train and save apartment model
xgb_model_apt = train_and_save_model(
    X_apt_train_processed, y_apt_train, 
    {"n_estimators": 670, "learning_rate": 0.055, "max_depth": 11, "colsample_bytree": 0.82, "subsample": 0.865},
    '/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/Preprocessing and modelling/xgb_model_apt.pkl'
)

# Train and save house model
xgb_model_house = train_and_save_model(
    X_house_train_processed, y_house_train, 
    {"n_estimators": 663, "learning_rate": 0.074, "max_depth": 11, "colsample_bytree": 0.68, "subsample": 0.6},
    '/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/Preprocessing and modelling/xgb_model_house.pkl'
)

# Function to evaluate and print model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")
    return y_pred

# Evaluate apartment model
print("Apartment Model Performance:")
y_apt_pred = evaluate_model(xgb_model_apt, X_apt_test_processed, y_apt_test)

# Evaluate house model
print("House Model Performance:")
y_house_pred = evaluate_model(xgb_model_house, X_house_test_processed, y_house_test)

# Function to plot feature importances
def plot_feature_importances(model, relevant_columns, title):
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': relevant_columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 10))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

# Plot feature importances for apartment model
plot_feature_importances(xgb_model_apt, relevant_columns_apt, 'Feature Importances for Apartment Model')

# Plot feature importances for house model
plot_feature_importances(xgb_model_house, relevant_columns_house, 'Feature Importances for House Model')