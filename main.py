import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import explore
import preprocess
import train
import evaluate

warnings.filterwarnings("ignore")
pd.set_option("display.width", 150)

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_FOLDER, "data", "vehicles_co2_emissions.csv")
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def save_plot(file_name):
    plt.savefig(os.path.join(OUTPUT_FOLDER, file_name), bbox_inches="tight")
    plt.close("all")
    print("Saved outputs/" + file_name)

print("\n===== STEP 1: UNDERSTAND THE DATA =====")
df = pd.read_csv(DATA_FILE)
explore.show_basic_info(df)

print("\nMean, median and standard deviation:")
print(explore.get_statistics(df).round(2))

correlation = explore.get_correlation(df)
covariance = explore.get_covariance(df)
print("\nCorrelation of every column with CO2:")
print(correlation["CO2(g/km)"].sort_values(ascending=False).round(3))
print("\nCovariance of every column with CO2:")
print(covariance["CO2(g/km)"].round(3))

explore.plot_correlation_heatmap(correlation)
save_plot("1_correlation_heatmap.png")
explore.plot_target_histogram(df)
save_plot("2_co2_histogram.png")
explore.plot_scatter_plots(df)
save_plot("3_scatter_plots.png")
explore.plot_co2_by_fuel(df)
save_plot("4_co2_by_fuel.png")


print("\n===== STEP 2: PREPROCESSING =====")
clean_df = preprocess.clean_data(df)
clean_df = preprocess.make_new_columns(clean_df)
X_train, X_test, y_train, y_test = preprocess.encode_and_split(clean_df)
X_train_scaled, X_test_scaled = preprocess.scale_features(X_train, X_test)

print("\n===== STEP 3: TRAINING (gradient descent from scratch) =====")
y_mean = y_train.mean()
y_std = y_train.std()
y_train_scaled = (y_train - y_mean) / y_std

train.compare_learning_rates(X_train_scaled, y_train_scaled, [0.001, 0.01, 0.1, 0.3], 1000)
save_plot("5_learning_rates.png")

weights, bias, loss_history = train.gradient_descent(X_train_scaled, y_train_scaled, 0.3, 3000)
train.plot_loss(loss_history, "Full model - loss while training")
save_plot("6_loss_full_model.png")

sklearn_model = LinearRegression().fit(X_train_scaled, y_train_scaled)
difference = np.abs(weights - sklearn_model.coef_).max()
print("Largest weight difference vs scikit-learn:", round(difference, 5))

print("\n===== STEP 4: EVALUATION =====")
train_pred = train.predict(X_train_scaled, weights, bias) * y_std + y_mean
test_pred = train.predict(X_test_scaled, weights, bias) * y_std + y_mean
full_table = evaluate.metrics_table(y_train, train_pred, y_test, test_pred)
print("\nFULL MODEL (uses fuel consumption)")
print(full_table)
evaluate.plot_actual_vs_predicted(y_test, test_pred, "Full model (test data)")
save_plot("7_actual_vs_predicted_full.png")
evaluate.plot_residuals(y_test, test_pred, "Full model")
save_plot("8_residuals_full.png")

spec_columns = [c for c in X_train_scaled.columns if c != "fuel_consumption"]
spec_weights, spec_bias, spec_loss = train.gradient_descent(
    X_train_scaled[spec_columns], y_train_scaled, 0.3, 3000)
spec_train_pred = train.predict(X_train_scaled[spec_columns], spec_weights, spec_bias) * y_std + y_mean
spec_test_pred = train.predict(X_test_scaled[spec_columns], spec_weights, spec_bias) * y_std + y_mean
spec_table = evaluate.metrics_table(y_train, spec_train_pred, y_test, spec_test_pred)
print("\nSPECS-ONLY MODEL (no fuel consumption)")
print(spec_table)
evaluate.plot_actual_vs_predicted(y_test, spec_test_pred, "Specs-only model (test data)")
save_plot("9_actual_vs_predicted_specs_only.png")
evaluate.plot_residuals(y_test, spec_test_pred, "Specs-only model")
save_plot("10_residuals_specs_only.png")

print("\n===== STEP 5: WHICH FEATURES MATTER? =====")
coefficients = pd.Series(weights * y_std, index=X_train_scaled.columns).sort_values()
plt.figure(figsize=(8, 6))
plt.barh(coefficients.index, coefficients.values)
plt.xlabel("Change in CO2 (g/km)")
plt.title("Full model: effect of each feature")
save_plot("11_feature_effects.png")

all_metrics = pd.concat([full_table, spec_table], keys=["Full model", "Specs-only model"])
all_metrics.to_csv(os.path.join(OUTPUT_FOLDER, "metrics.csv"))
print("Saved outputs/metrics.csv")
print("\nDone.")
