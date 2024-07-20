# callbacks.py
from pages import home
from pages import notfound
import plotly.graph_objs as go
from utils import *


def render_page_content(pathname):
    if pathname == "/":
        return home.layout
    elif pathname == "/home":
        return home.layout
    else:
        return notfound.layout


def toggle_classname(n, classname):
    if n and classname == "":
        return "collapsed"
    return ""


def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


def update_metrics(n_intervals,is_open):
    results = home.update_metrics(n_intervals, is_open)
    if results is None:
        return ["N/A", "N/A", "N/A", "N/A", "No data available", False, go.Figure()]
    
    accuracy, precision, recall, f1, toast_message, toast_open, X_train = results
    figure = home.update_histogram(X_train)
    return accuracy, precision, recall, f1, toast_message, toast_open, figure

