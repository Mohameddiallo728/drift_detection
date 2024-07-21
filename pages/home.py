# home.py
import dash_bootstrap_components as dbc
from dash import html, dcc
from utils import *
import plotly.subplots as sp
import plotly.graph_objs as go


# Load initial data and model
df = load_data()
model, X_train, X_test, y_train, y_test, accuracy, precision, recall, f1 = initial_training(df)


layout = dbc.Container([
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Accuracy"),dbc.CardBody(html.H1("Accuracy", id='accuracy-value'))], color="cadetblue", inverse=True), width=3),
        dbc.Col(dbc.Card([dbc.CardHeader("Precision"),dbc.CardBody(html.H1("Precision", id='precision-value'))], color="black", inverse=True), width=3),
        dbc.Col(dbc.Card([dbc.CardHeader("Recall"),dbc.CardBody(html.H1("Recall", id='recall-value'))], color="#609FFF", inverse=True), width=3),
        dbc.Col(dbc.Card([dbc.CardHeader("F1 Score"),dbc.CardBody(html.H1("F1 Score", id='f1-value'))], color="rebeccapurple", inverse=True), width=3)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='live-histogram', config={'displayModeBar': False}), style={"margin-top":"20px"}),
    ]),
    dbc.Toast(
        [
            html.P("", className="mb-0", id='toast-message')
        ],
        id="drift-toast",
        header="Dérive des données détectée !",
        is_open=False,
        dismissable=True,
        duration=10000,  # Duration in milliseconds
        icon="danger",
        style={"position": "fixed", "top": 66, "right": 30, "width": 350, "background" : "#fff", "fontSize": "17px", "color":"#000"},
    ),
    dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0)  # Check every 10 seconds
], fluid=True)


def update_metrics(n_intervals, is_open):
    global accuracy, precision, recall, f1, model, X_train, y_train
    try:
        # Générer des nouvelles données avec dérive
        data_stream = ingest_data_stream_with_drift()

        new_data = next(data_stream)
        X_new = pd.DataFrame([new_data])[feature_columns]

        # Faire une prédiction pour l'Outcome
        y_pred = model.predict(X_new) 
        
        # Assurez-vous que y_pred est en 0 ou 1
        y_pred = (y_pred > 0.5).astype(int)

        # Ajouter la prédiction au DataFrame
        y_new = pd.Series([y_pred[0]], name=target_column)
        
        # Vérifiez si X_new existe déjà dans X_train pour éviter les doublons
        if not X_train.equals(pd.concat([X_train, X_new]).drop_duplicates()):
            X_train = pd.concat([X_train, X_new], ignore_index=True)
            y_train = pd.concat([y_train, y_new], ignore_index=True)

            # Détecter la dérive
            drift_detected, drift_scores = detect_drift(X_train, X_test, 0.09)

            # Réentraîner le modèle
            model.fit(X_train, y_train)

            # Enregistrer les métriques et les données
            if drift_detected:
                print(f"Drift Detected on : {new_data}")
                print(f"Predicted as : {y_pred}")

                if len(drifted_data_list) == 0 or new_data != drifted_data_list[-1]:
                    drifted_data_list.append(new_data)

                toast_message = notify_and_recommend(drift_detected, drift_scores, feature_columns, model, X_test, y_test)
                accuracy, precision, recall, f1 = get_metrics(model, X_test, y_test)
                
                # Enregistrer les métriques et les données
                save_metrics_and_data(drifted_data_list, metrics)
                return (
                    f"{accuracy:.2f}",
                    f"{precision:.2f}",
                    f"{recall:.2f}",
                    f"{f1:.2f}",
                    toast_message,
                    True,  # Montrer le toast
                    X_train.to_dict('records')
                )
            else:
                accuracy, precision, recall, f1 = get_metrics(model, X_test, y_test)
                # Enregistrer les métriques et les données
                save_metrics_and_data(drifted_data_list, metrics)
                return (
                    f"{accuracy:.2f}",
                    f"{precision:.2f}",
                    f"{recall:.2f}",
                    f"{f1:.2f}",
                    "",
                    False,  # Ne pas montrer le toast
                    X_train.to_dict('records')
                )
        else:
            # Si les nouvelles données sont déjà dans X_train, ne rien faire
            return (
                f"{accuracy:.2f}",
                f"{precision:.2f}",
                f"{recall:.2f}",
                f"{f1:.2f}",
                "",
                False,  # Ne pas montrer le toast
                X_train.to_dict('records')
            )
    except Exception as e:
        print(f"Erreur lors de la mise à jour des métriques: {e}")
        # Retourner des valeurs par défaut en cas d'erreur
        return (
            f"{accuracy:.2f}",
            f"{precision:.2f}",
            f"{recall:.2f}",
            f"{f1:.2f}",
            "Erreur lors de la mise à jour des métriques",
            False,  # Ne pas montrer le toast
            X_train.to_dict('records')
        )


def update_histogram(X_train):
    # Convertir la liste de dictionnaires en DataFrame
    df_train = pd.DataFrame(X_train)
    
    # Liste de couleurs à utiliser pour les histogrammes
    color_list = list(colors.values())
    
    # Déterminer le nombre de lignes nécessaires pour deux colonnes
    num_features = len(df_train.columns)
    num_cols = 2
    num_rows = (num_features + num_cols - 1) // num_cols  # Calculer le nombre de lignes nécessaires

    # Créer des sous-figures avec le nombre de lignes et colonnes désiré
    fig = sp.make_subplots(rows=num_rows, cols=num_cols, subplot_titles=df_train.columns)
    
    for idx, column in enumerate(df_train.columns):
        row = idx // num_cols + 1
        col = idx % num_cols + 1
        fig.add_trace(
            go.Histogram(
                x=df_train[column],
                marker_color=color_list[idx % len(color_list)],
                name=column
            ),
            row=row, col=col
        )
    
    layout_style = dict(
        plot_bgcolor='#f9f9f9',
        title_font_size=22,
        height=450 * num_rows,  # Ajuster la hauteur en fonction du nombre de lignes
        showlegend=False
    )

    fig.update_layout(
        title_text="Distribution des données",
        **layout_style
    )
    
    return fig