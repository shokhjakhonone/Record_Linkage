from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
from ml import create_central_repository
import os
import seaborn as sns
import plotly.express as px
import dash_app # Импортируем Dash-приложение

app = Flask(__name__)

# Присоединяем Dash-приложение к Flask
dash_app.init_app(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file1' not in request.files or 'file2' not in request.files:
        return redirect(request.url)

    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1.filename == '' or file2.filename == '':
        return redirect(request.url)

    dataset1 = pd.read_csv(file1)
    dataset2 = pd.read_csv(file2)

    central_repository = create_central_repository(dataset1, dataset2)

    output_file = 'central_repository.csv'
    central_repository.to_csv(output_file, index=False)

    sns_plot = sns.countplot(data=central_repository, x='full_name')
    sns_plot.figure.savefig(os.path.join('static', 'seaborn_plot.png'), bbox_inches='tight')

    fig_histogram = px.histogram(central_repository, x='contact_info', title='Распределение контактной информации', color='full_name', hover_data=['contact_info'])
    fig_histogram.write_html(os.path.join('static', 'plotly_histogram.html'))

    fig_bar = px.bar(central_repository, x='full_name', title='Количество записей по полным именам')
    fig_bar.write_html(os.path.join('static', 'plotly_bar.html'))

    fig_scatter = px.scatter(central_repository, x='full_name', y='contact_info', title='Диаграмма рассеяния')
    fig_scatter.write_html(os.path.join('static', 'plotly_scatter.html'))

    return render_template('results1.html', tables=[central_repository.to_html(classes='data')], titles=central_repository.columns.values, download_link=output_file)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=7577)
