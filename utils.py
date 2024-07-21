 # util.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from dash import html
import dash_bootstrap_components as dbc




# Initialize global variables for storing metrics and drifted data
metrics = []
drifted_data_list = []
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
target_column = 'Outcome'
# Palette de couleurs définie
colors = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#bcbd22",
    "cyan": "#17becf"
}


def load_data():
    return pd.read_csv('diabetes.csv')

# Entrainement initiale

def initial_training(df):
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy, precision, recall, f1 = get_metrics(model, X_test, y_test)

    return model, X_train, X_test, y_train, y_test, accuracy, precision, recall, f1



# Update these variables with latest metrics
def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print the new performance metrics
    print(f"\nPerformance du modèle au {datetime.now()}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"--"*20)

    # historisation 
    metric = {
        'Timestamp': datetime.now(),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    metrics.append(metric)

    return accuracy, precision, recall, f1


# Function to generate random data and make predictions
def generate_random_data():
    # Generate random values for each feature
    data = {
        'Pregnancies': np.random.randint(0, 20),
        'Glucose': np.random.randint(70, 200),
        'BloodPressure': np.random.randint(30, 120),
        'SkinThickness': np.random.randint(0, 100),
        'Insulin': np.random.randint(0, 600),
        'BMI': round(np.random.uniform(15, 50), 1),
        'DiabetesPedigreeFunction': round(np.random.uniform(0, 2.5), 3),
        'Age': np.random.randint(18, 80)
    }
    return pd.DataFrame([data])


# Function to introduce drift into the data
def introduce_drift(data, drift_probability=0.5):
    random = np.random.rand()
    drifted_data = data.copy()
    if random < drift_probability:
        #original_glucose = drifted_data['Glucose'].values[0]
        #original_bmi = drifted_data['BMI'].values[0]
        #drifted_data['Glucose'] = round(drifted_data['Glucose'] + np.random.randint(30, 100))
        #drifted_data['BMI'] = round(drifted_data['BMI'] + np.random.normal(-5, 5), 1)
        #print(f"Original Glucose: {original_glucose}, Drifted Glucose: {drifted_data['Glucose'].values[0]}")
        #print(f"Original BMI: {original_bmi}, Drifted BMI: {drifted_data['BMI'].values[0]}")
        print("random :", random)
        print("drift_probability :", drift_probability)
    return drifted_data



# Data ingestion with drift introduction
def ingest_data_stream_with_drift():
    while True:
    # Generate random data
        data = generate_random_data()
        # Introduce drift with a certain probability
        data_with_drift = introduce_drift(data, 0.5)
        yield data_with_drift.to_dict('records')[0]
        time.sleep(1) 


# Drift detection
def detect_drift(X_train, X_test, drift_threshold):
    # Perform Kolmogorov-Smirnov test to detect data drift
    drift_scores = []
    for col in X_train.columns:
        drift_score = ks_2samp(X_train[col], X_test[col]).statistic
        drift_scores.append(drift_score)
        print(f"Score drift pour {col}: {drift_score:.2f}")
    # Check if the maximum drift score exceeds a threshold
    if max(drift_scores) > drift_threshold:
        return True, drift_scores
    else:
        return False, drift_scores
    


def notify_and_recommend(drift_detected, drift_scores, columns, model, X_test, y_test):
    if drift_detected:
        toast_message = [
            html.P("Étudier les caractéristiques ayant des scores de dérive élevés :"),
            html.Ul([html.Li(f"{columns[i]}: {score:.2f}") for i, score in enumerate(drift_scores)]),
            dbc.Alert([
                html.I(className="bi bi-check-circle-fill"),
                "Modèle mise à jour",
            ],
                color="warning",
                className="d-flex align-items-center",
            ),
        ]
    else:
        toast_message = "Aucune dérive des données n'a été détectée. (Modèle mise à jour)"

    return toast_message


# Track diabetic distribution
def track_diabetic_distribution(df):
    df['Timestamp'] = pd.to_datetime(df.index, unit='s')
    df['Diabetic'] = df['Outcome']
    df.set_index('Timestamp', inplace=True)
    df['Diabetic'].resample('D').mean().plot(title='Répartition des diabétiques au fil du temps')
    plt.xlabel('Date')
    plt.ylabel('Proportion de diabétiques')
    plt.show()
    


# Save metrics and drifted data to files
def save_metrics_and_data(drifted_data_list, metrics):
    pd.DataFrame(drifted_data_list).to_csv('drifted_data.csv', index=False)
    pd.DataFrame(metrics).to_csv('performance_metrics.csv', index=False)