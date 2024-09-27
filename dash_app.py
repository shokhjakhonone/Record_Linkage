import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

def init_app(server):
    dash_app = dash.Dash(__name__, server=server, url_base_pathname='/dash/')

    dash_app.layout = html.Div([
        html.H1("Dashboard"),
        dcc.Graph(id='bar_chart'),
        dcc.Dropdown(
            id='filter_dropdown',
            options=[{'label': name, 'value': name} for name in ['Sample Name']],  # Замените на реальные имена из данных
            value='Sample Name'
        )
    ])

    @dash_app.callback(
        Output('bar_chart', 'figure'),
        [Input('filter_dropdown', 'value')]
    )
    def update_bar_chart(selected_name):
        filtered_data = pd.DataFrame()  # Замените на данные, фильтруемые по выбранному имени
        fig_bar = px.bar(filtered_data, x='contact_info', y='other_info', title=f'Информация по {selected_name}')
        return fig_bar
