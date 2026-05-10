import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, session as flask_session
from database import init_db, get_db
import random

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
        nick1 = request.form.get('nick1')
        nick2 = request.form.get('nick2')
        
        db = get_db()
        # Insert users
        cursor = db.cursor()
        cursor.execute("INSERT INTO user (nickname) VALUES (?)", (nick1,))
        u1_id = cursor.lastrowid
        cursor.execute("INSERT INTO user (nickname) VALUES (?)", (nick2,))
        u2_id = cursor.lastrowid
        
        # Create session
        cursor.execute("INSERT INTO session (user1_id, user2_id) VALUES (?, ?)", (u1_id, u2_id))
        session_id = cursor.lastrowid
        db.commit()
        
        # Store in flask session for results
        flask_session['session_id'] = session_id
        flask_session['nick1'] = nick1
        flask_session['nick2'] = nick2
        
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
    session_id = flask_session.get('session_id')
    if not session_id:
        return redirect(url_for('register'))
        
    db = get_db()
    
    # Generate mock score if not already saved
    # In a real app, this would be calculated from EEG data
    mock_score = random.randint(70, 99)
    
    # Check if result already exists for this session
    res = db.execute("SELECT score FROM result WHERE session_id = ?", (session_id,)).fetchone()
    if not res:
        db.execute("INSERT INTO result (session_id, score) VALUES (?, ?)", (session_id, mock_score))
        db.commit()
        score = mock_score
    else:
        score = res['score']

    # Fetch Leaderboard
    leaderboard = db.execute(
        """
        SELECT u1.nickname as nick1, u2.nickname as nick2, r.score 
        FROM result r
        JOIN session s ON r.session_id = s.id
        JOIN user u1 ON s.user1_id = u1.id
        JOIN user u2 ON s.user2_id = u2.id
        ORDER BY r.score DESC
        LIMIT 10
        """
    ).fetchall()
    
    return render_template('results.html', 
                           score=score, 
                           nick1=flask_session.get('nick1'), 
                           nick2=flask_session.get('nick2'),
                           leaderboard=leaderboard)

if __name__ == '__main__':
    app.run(debug=True)
