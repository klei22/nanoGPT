from flask import Flask, render_template, send_from_directory, jsonify, abort, url_for
import os

app = Flask(__name__)
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def list_report_dirs():
    return [d for d in os.listdir(REPORTS_DIR)
            if os.path.isdir(os.path.join(REPORTS_DIR, d)) and not d.startswith('_')]


@app.route('/')
def index():
    dirs = list_report_dirs()
    return render_template('index.html', dirs=dirs)


@app.route('/folder/<folder>')
def folder_view(folder):
    folder_path = os.path.join(REPORTS_DIR, folder)
    if not os.path.isdir(folder_path):
        abort(404)
    pdfs = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    return render_template('folder.html', folder=folder, pdfs=pdfs)


@app.route('/view/<folder>/<pdf>')
def view_pdf(folder, pdf):
    folder_path = os.path.join(REPORTS_DIR, folder)
    pdf_path = os.path.join(folder_path, pdf)
    if not (os.path.isfile(pdf_path) and pdf.lower().endswith('.pdf')):
        abort(404)
    mtime = os.path.getmtime(pdf_path)
    return render_template('view.html', folder=folder, pdf=pdf, mtime=mtime)


@app.route('/pdf/<folder>/<pdf>')
def pdf_file(folder, pdf):
    folder_path = os.path.join(REPORTS_DIR, folder)
    return send_from_directory(folder_path, pdf)


@app.route('/mtime/<folder>/<pdf>')
def pdf_mtime(folder, pdf):
    folder_path = os.path.join(REPORTS_DIR, folder)
    pdf_path = os.path.join(folder_path, pdf)
    if not os.path.isfile(pdf_path):
        abort(404)
    mtime = os.path.getmtime(pdf_path)
    return jsonify({'mtime': mtime})


if __name__ == '__main__':
    app.run(debug=True)
