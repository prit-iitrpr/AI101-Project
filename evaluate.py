import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_predicted):
    mae = mean_absolute_error(y_true, y_predicted)      
    mse = mean_squared_error(y_true, y_predicted)       
    rmse = np.sqrt(mse)                                
    r2 = r2_score(y_true, y_predicted)                 
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def metrics_table(y_train, train_predictions, y_test, test_predictions):
    table = pd.DataFrame()
    table["Training"] = calculate_metrics(y_train, train_predictions)
    table["Testing"] = calculate_metrics(y_test, test_predictions)
    return table.round(4)


def plot_actual_vs_predicted(y_true, y_predicted, title):
    plt.scatter(y_true, y_predicted, alpha=0.3, s=8)
    low = min(y_true.min(), y_predicted.min())
    high = max(y_true.max(), y_predicted.max())
    plt.plot([low, high], [low, high], color="red")
    plt.xlabel("Actual CO2 (g/km)")
    plt.ylabel("Predicted CO2 (g/km)")
    plt.title(title)
    plt.show()


def plot_residuals(y_true, y_predicted, title):
    residuals = np.array(y_true) - y_predicted
    plt.scatter(y_predicted, residuals, alpha=0.3, s=8)
    plt.axhline(0, color="red")
    plt.xlabel("Predicted CO2 (g/km)")
    plt.ylabel("Residual")
    plt.title(title + " - residuals vs predicted")

    plt.hist(residuals, bins=50)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Histogram of residuals")
    plt.show()
