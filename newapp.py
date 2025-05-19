from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import numpy as np
import sqlite3
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import random
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# ----------- Prevent Back Button Cache (After Logout) -----------
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# ----------- Database Setup -----------
def init_db():
    if not os.path.exists('users.db'):
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

init_db()

# ----------- Model Training -----------
model_file = 'new_heart_disease_model.pkl'
scaler_file = 'new_scaler.pkl'

if not os.path.exists(model_file) or not os.path.exists(scaler_file):
    print("Training new model...")
    try:
        data = pd.read_csv('cardio_train_5001_rows.csv')
    except FileNotFoundError:
        print("Error: cardio_train_5001_rows.csv not found.")
        exit()

    if 'cardio' not in data.columns:
        print("Error: 'cardio' column not found.")
        exit()

    X = data.drop(['id', 'cardio'], axis=1)
    y = data['cardio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf_model = RandomForestClassifier(random_state=42)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)

    voting_model = VotingClassifier(estimators=[
        ('rf', rf_model),
        ('lr', lr_model)
    ], voting='soft')

    voting_model.fit(X_train, y_train)

    pickle.dump(voting_model, open(model_file, 'wb'))
    pickle.dump(scaler, open(scaler_file, 'wb'))
else:
    try:
        model = pickle.load(open(model_file, 'rb'))
        scaler = pickle.load(open(scaler_file, 'rb'))
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

# ----------- Email Sending -----------
def send_otp_email(to_email, otp):
    gmail_user = os.environ.get("EMAIL_USER")
    gmail_password = os.environ.get("EMAIL_PASS")

    subject = 'Your OTP for Heal My Heart Registration'
    body = f'Your OTP to complete registration is: {otp}'

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = gmail_user
    msg['To'] = to_email

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# ----------- Routes -----------

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', prediction_text=None, gif_url=None)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid login. Try again.")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            return render_template('register.html', error="Passwords do not match.")

        # Check if email is already used
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        existing_user = cursor.fetchone()
        conn.close()
        if existing_user:
            return render_template('register.html', error="This email is already registered. Please log in or use another email.")

        # Generate OTP and save registration info in session
        otp = str(random.randint(100000, 999999))
        session['reg_username'] = username
        session['reg_email'] = email
        session['reg_password'] = password
        session['reg_otp'] = otp

        if not send_otp_email(email, otp):
            return render_template('register.html', error="Failed to send OTP email.")

        return redirect(url_for('verify_otp'))

    return render_template('register.html', error=None)

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        entered_otp = request.form['otp']
        real_otp = session.get('reg_otp')

        if not real_otp:
            flash("Session expired. Please register again.")
            return redirect(url_for('register'))

        if entered_otp == real_otp:
            username = session['reg_username']
            email = session['reg_email']
            password = session['reg_password']

            try:
                conn = sqlite3.connect('users.db')
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                               (username, email, password))
                conn.commit()
                conn.close()
            except sqlite3.IntegrityError:
                flash("Username or Email already exists.")
                return redirect(url_for('register'))

            session.pop('reg_username', None)
            session.pop('reg_email', None)
            session.pop('reg_password', None)
            session.pop('reg_otp', None)

            flash("Registration successful! Please login.")
            return redirect(url_for('login'))
        else:
            flash("Incorrect OTP. Please try again.")
            return render_template('verify_otp.html')

    return render_template('verify_otp.html')

def calculate_age_in_days(dob_str):
    try:
        dob = datetime.strptime(dob_str, '%d-%m-%Y')
        today = datetime.now()
        return (today - dob).days
    except ValueError:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    try:
        dob_str = request.form['dob']
        age_in_days = calculate_age_in_days(dob_str)
        if age_in_days is None:
            return render_template('index.html', error="Invalid DOB format. Use DD-MM-YYYY.")

        gender = int(request.form['gender'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        ap_hi = int(request.form['ap_hi'])
        ap_lo = int(request.form['ap_lo'])
        cholesterol = int(request.form['cholesterol'])
        gluc = int(request.form['gluc'])
        smoke = int(request.form['smoke'])
        alco = int(request.form['alco'])
        active = int(request.form['active'])

        features = [age_in_days, gender, height, weight, ap_hi, ap_lo,
                    cholesterol, gluc, smoke, alco, active]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)

        session['prediction'] = int(prediction[0])
        return redirect(url_for('result'))

    except Exception as e:
        return render_template('index.html', error=f"Error occurred: {str(e)}")

@app.route('/result')
def result():
    if 'username' not in session:
        return redirect(url_for('login'))
    prediction = session.get('prediction')
    if prediction is None:
        return redirect(url_for('home'))

    if prediction == 1:
        result = "⚠️ Alert: Your data indicates a higher risk of heart disease. Please consult a healthcare professional."
        gif_url = url_for('static', filename='crying-heart-crying.gif')
    else:
        result = "✅ Everything looks fine. Keep up the great work for a healthier heart!"
        gif_url = url_for('static', filename='heart_ann.gif')
    return render_template('result.html', prediction_text=result, gif_url=gif_url)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('welcome'))

if __name__ == "__main__":
    app.run(debug=True)
