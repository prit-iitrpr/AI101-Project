# STEP 1 - Understand the data (mean, median, std, covariance, correlation)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_basic_info(df):
    print("Shape (rows, columns):", df.shape)
    print()
    print("Missing values in each column:")
    missing = df.isnull().sum()
    print(missing[missing > 0])         
    print()
    print("Duplicate rows:", df.duplicated().sum())


def get_numeric_columns(df):
    numeric = df.select_dtypes(include="number")
    numeric = numeric.drop(columns=["Unnamed: 0"], errors="ignore")
    return numeric


def get_statistics(df):
    numeric = get_numeric_columns(df)
    stats = pd.DataFrame()
    stats["mean"] = numeric.mean()
    stats["median"] = numeric.median()
    stats["std"] = numeric.std()
    stats["min"] = numeric.min()
    stats["max"] = numeric.max()
    return stats


def get_covariance(df):
    return get_numeric_columns(df).cov()


def get_correlation(df):
    return get_numeric_columns(df).corr()


def plot_correlation_heatmap(correlation):
    plt.figure(figsize=(12, 9))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 8})
    plt.title("Correlation between numeric columns")
    plt.show()


def plot_target_histogram(df):
    plt.figure(figsize=(8, 4))
    plt.hist(df["CO2(g/km)"].dropna(), bins=50)
    plt.xlabel("CO2 emissions (g/km)")
    plt.ylabel("Number of vehicles")
    plt.title("Distribution of CO2 emissions")
    plt.show()


def plot_scatter_plots(df):
    columns = ["Mixed_consumption(l/100km)", "Maximal_power(kW)",
               "Europe_empty_mass_minimum(kg)", "Urban_consumption(l/100km)"]
    small = df.sample(3000, random_state=42)

    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.scatter(small[columns[i]], small["CO2(g/km)"], alpha=0.3, s=8)
        plt.xlabel(columns[i])
        plt.ylabel("CO2 (g/km)")
    plt.tight_layout()
    plt.show()


def plot_co2_by_fuel(df):
    plt.figure(figsize=(12, 4))
    sns.boxplot(data=df, x="Fuel", y="CO2(g/km)")
    plt.title("CO2 emissions for each fuel type")
    plt.show()
