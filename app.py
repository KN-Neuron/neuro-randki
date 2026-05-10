import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from database import init_db, get_db

app = Flask(__name__)
app.config['DATABASE'] = os.path.join(app.root_path, 'neuro_randki.db')
app.secret_key = 'dev_key_for_neuro_randki'

# Ensure database is initialized
with app.app_context():
    init_db()

@app.route('/')
def index():
    return redirect(url_for('register'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Logic to save users will go here in Phase 2
        # For now, just redirect to waiting
        return redirect(url_for('waiting'))
    return render_template('register.html')

@app.route('/waiting')
def waiting():
    return render_template('waiting.html')

@app.route('/task')
def task():
    return render_template('task.html')

@app.route('/results')
def results():
    # Mock results for Phase 1
    mock_score = 85
    return render_template('results.html', score=mock_score)

if __name__ == '__main__':
    app.run(debug=True)
