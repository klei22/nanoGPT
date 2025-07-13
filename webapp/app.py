import subprocess
import threading
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

process_logs = {}
process_status = {}


def run_command(cmd, key):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    process_status[key] = 'running'
    lines = []
    for line in proc.stdout:
        lines.append(line)
    proc.wait()
    process_logs[key] = ''.join(lines)
    process_status[key] = 'finished'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run_experiments', methods=['POST'])
def run_experiments():
    config = request.form['config']
    fmt = request.form.get('format', 'yaml')
    output_dir = request.form.get('output_dir', 'out')
    prefix = request.form.get('prefix', '')
    use_ts = request.form.get('use_timestamp') == 'on'

    cmd = ['python3', 'optimization_and_search/run_experiments.py',
           '-c', config, '--config_format', fmt, '-o', output_dir]
    if prefix:
        cmd += ['--prefix', prefix]
    if use_ts:
        cmd.append('--use_timestamp')

    key = 'run_experiments'
    thread = threading.Thread(target=run_command, args=(cmd, key))
    thread.start()
    return redirect(url_for('output', key=key))


@app.route('/run_train', methods=['POST'])
def run_train():
    args = request.form['train_args'].split()
    cmd = ['python3', 'train.py'] + args
    key = 'train'
    thread = threading.Thread(target=run_command, args=(cmd, key))
    thread.start()
    return redirect(url_for('output', key=key))


@app.route('/output/<key>')
def output(key):
    log = process_logs.get(key, '')
    status = process_status.get(key, 'running')
    return render_template('output.html', log=log, status=status, key=key)


@app.route('/tensorboard')
def tensorboard():
    logdir = request.args.get('logdir', 'runs')
    port = request.args.get('port', '6006')
    if 'tensorboard' not in process_status:
        subprocess.Popen(['tensorboard', '--logdir', logdir, '--port', str(port)])
        process_status['tensorboard'] = 'running'
    return redirect(f'http://localhost:{port}')


if __name__ == '__main__':
    app.run(port=8080)
