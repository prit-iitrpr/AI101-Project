import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(df):
    columns_to_drop = [
        "Unnamed: 0", "Model", "UTAC_model", "CNIT", "Type_variant_version_(TVV)", "Field_V9",
        "CO_type_I(g/km)", "HC(g/km)", "NOX(g/km)", "HC+NOX(g/km)", "Particules(g/km)",
        "Administrative_power",
        "Urban_consumption(l/100km)", "Extra-urban_consumption(l/100km)",
        "Europe_empty_mass_maximum(kg)",
        "Brand", "Range",
    ]
    df = df.drop(columns=columns_to_drop)

    df = df.rename(columns={
        "Maximal_power(kW)": "power_kw",
        "Mixed_consumption(l/100km)": "fuel_consumption",
        "Europe_empty_mass_minimum(kg)": "mass_kg",
        "CO2(g/km)": "co2",
    })

    rows_before = len(df)
    df = df.dropna()
    print("Rows removed because of missing values:", rows_before - len(df))

    rows_before = len(df)
    df = df[df["fuel_consumption"] > 0]
    print("Rows removed because of impossible fuel consumption:", rows_before - len(df))

    print("Rows left:", len(df))
    return df


def make_new_columns(df):
    df = df.copy()

    fuel_names = {"ES": "Petrol", "GO": "Diesel", "EH": "Hybrid", "GH": "Hybrid"}
    df["fuel_type"] = df["Fuel"].map(fuel_names)
    df["fuel_type"] = df["fuel_type"].fillna("Other")      

    df["gearbox_type"] = df["Gearbox"].str.split(" ").str[0]        
    df["gears"] = df["Gearbox"].str.split(" ").str[1].astype(int)     

    
    common_bodies = df["Body"].value_counts().head(5).index
    df.loc[~df["Body"].isin(common_bodies), "Body"] = "OTHER"

   
    df = df.drop(columns=["Fuel", "Gearbox"])
    return df


def encode_and_split(df):

    df_encoded = pd.get_dummies(df, columns=["fuel_type", "Hybrid", "gearbox_type", "Body"],
                                drop_first=True, dtype=float)

    X = df_encoded.drop(columns=["co2"])     
    y = df_encoded["co2"]                    

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("Training rows:", len(X_train), "| Testing rows:", len(X_test))
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    mean = X_train.mean()
    std = X_train.std()
    std[std == 0] = 1                        
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled
