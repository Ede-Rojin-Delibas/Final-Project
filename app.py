from flask import Flask, render_template, request, redirect, url_for, flash, send_file,session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import numpy as np
import os
from io import BytesIO
from datetime import datetime
import random
import string
from faker import Faker
from werkzeug.security import generate_password_hash, check_password_hash
import re  # Regex kütüphanesi
from sdv.single_table import CTGANSynthesizer
import tempfile
import sdv
from sdv.metadata import SingleTableMetadata
from flask_wtf import CSRFProtect #İzinsiz işlemleri engellemek için
import secrets
from datetime import timedelta


app = Flask(__name__)
csrf = CSRFProtect(app)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config['SESSION_COOKIE_SECURE'] = False        # HTTPS ile çalışır (yayın ortamında aktif)
app.config['SESSION_COOKIE_HTTPONLY'] = True      # JavaScript erişemez
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'     # CSRF koruması
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.permanent_session_lifetime = timedelta(minutes=30)  # Oturum süresi (30 dakika)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Kullanıcı Modeli
class User(db.Model, UserMixin):  # UserMixin ekledik
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password)  # Şifre hashleme

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)  # Şifre doğrulama

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))  # Kullanıcıyı ID ile getir(veri tabanından al)

# Ana Sayfa
@app.route('/', methods=['GET', 'POST'])
def home():
    query = request.form.get('search_query', '')

    if query:
        users = User.query.filter(User.email.contains(query)).all()
    else:
        users = User.query.all()
    return render_template('index.html', users=users, query=query)
#sahte veri üretimi
def generate_fake_data(rows, columns, data_types):
    fake = Faker()
    data = pd.DataFrame()

    for i in range(columns):
        if data_types[i] == 'integer':
            data[f'Sütun_{i+1}'] = [fake.random_int(min=0, max=100) for _ in range(rows)]
        elif data_types[i] == 'float':
            data[f'Sütun_{i+1}'] = [fake.pyfloat(left_digits=2, right_digits=2, positive=True) for _ in range(rows)]
        elif data_types[i] == 'text':
            data[f'Sütun_{i+1}'] = [fake.word() for _ in range(rows)]
        elif data_types[i] == 'date':
            data[f'Sütun_{i+1}'] = [fake.date_this_decade().strftime('%Y-%m-%d') for _ in range(rows)]
        elif data_types[i] == 'email':
            data[f'Sütun_{i+1}'] = [fake.email() for _ in range(rows)]
        elif data_types[i] == 'name':
            data[f'Sütun_{i+1}'] = [fake.name() for _ in range(rows)]
        elif data_types[i] == 'address':
            data[f'Sütun_{i+1}'] = [fake.address() for _ in range(rows)]
        else:
            data[f'Sütun_{i+1}'] = [fake.word() for _ in range(rows)]

    return data
#about sayfası
@app.route('/about')
def about():
    return render_template('about.html')

# Model Tabanlı Veri Üretimi Sayfası
@app.route('/generate_model', methods=['GET', 'POST'])
@login_required
def generate_model_data():
    if request.method == 'POST':
        # Dosya ve satır sayısını al
        uploaded_file = request.files.get('dataset')
        num_rows = int(request.form.get('num_rows', 100))
        file_format = request.form.get('format', 'csv')

        if not uploaded_file:
            flash("Lütfen bir CSV dosyası yükleyin!", "danger")
            return redirect(request.url)

        try:
            # Veriyi oku ve eğit
            real_data = pd.read_csv(uploaded_file)
            meta_data=SingleTableMetadata()
            meta_data.detect_from_dataframe(data=real_data)
            model = CTGANSynthesizer(metadata=meta_data)
            model.fit(real_data)

            # Yeni veri üret
            synthetic_data = model.sample(num_rows)

            # Dosya olarak hazırla
            output = BytesIO()
            if file_format == 'csv':
                synthetic_data.to_csv(output, index=False)
                output.seek(0)
                return send_file(output, as_attachment=True, download_name='synthetic_data.csv', mimetype='text/csv')
            elif file_format == 'xlsx':
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    synthetic_data.to_excel(writer, index=False)
                output.seek(0)
                return send_file(output, as_attachment=True, download_name='synthetic_data.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        except Exception as e:
            flash(f"Veri üretimi sırasında bir hata oluştu: {str(e)}", "danger")
            return redirect(request.url)

    return render_template('generate_model.html')
#şifre koruma mekanizmaları

def is_password_strong(password):
    """Parolanın güçlü olup olmadığını kontrol eder"""
    if len(password) < 8:
        return False, "Şifre en az 8 karakter olmalıdır."
    if not re.search(r'[A-Z]', password):
        return False, "Şifre en az bir büyük harf içermelidir."
    if not re.search(r'[a-z]', password):
        return False, "Şifre en az bir küçük harf içermelidir."
    if not re.search(r'\d', password):
        return False, "Şifre en az bir rakam içermelidir."
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Şifre en az bir özel karakter içermelidir."
    
    return True, "Şifre güçlüdür."


# Kayıt Olma
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Şifre doğrulama
        if password != confirm_password:
            flash("Şifreler uyuşmuyor! Lütfen tekrar deneyin.", "danger")
            return redirect(url_for('register'))

        # Kullanıcının zaten olup olmadığını kontrol et
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Bu e-posta adresi zaten kullanılıyor. Lütfen giriş yapın.", "warning")
            return redirect(url_for('login'))

        # **Doğru şekilde şifreyi hashle**
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        email = request.form['email'].strip().lower()
        is_admin = (email == "deyenazdar@gmail.com")  

        # Yeni kullanıcı oluştur
        new_user = User(email=email, password_hash=hashed_password, is_admin=is_admin)

        # Veritabanına kaydet
        db.session.add(new_user)
        db.session.commit()

        flash("Başarıyla kayıt oldunuz! Giriş yapabilirsiniz.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


# Giriş Yapma
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if not email or not password:
            flash("E-posta ve şifre zorunludur!", "danger")
            return redirect(url_for('login'))
        
        user = User.query.filter_by(email=email).first()  # Email ile sorgula
        print("Kullanıcı nesnesi:", user)  # Test çıktısı
        # if user:
        #     print("Veritabanındaki hash:", user.password_hash)  # Kullanıcının hash'lenmiş şifresini gör
        #     print("Girilen şifre:", password)  # Kullanıcının girdiği şifreyi gör
        #     print("Hash Check Result:", bcrypt.check_password_hash(user.password_hash, password))  # Doğrulama sonucunu yazdır
        
        if user and bcrypt.check_password_hash(user.password_hash, password):  # Hash doğrulama
            session.clear()               #Session fixation önle
            session.permanent = True     #Oturumu zamanlayalım
            login_user(user)
            flash('Giriş başarılı!', 'success')
            return redirect(url_for('generate_data'))
        else:
            flash('Giriş başarısız. Lütfen bilgilerinizi kontrol edin.', 'danger')
    
    return render_template('login.html')

from authlib.integrations.flask_client import OAuth

# OAuth ayarları
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id='1012906305536-80b6pm3k19j75fiu4mfjro8ni32jrulb.apps.googleusercontent.com',  
    client_secret='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',  
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    client_kwargs={'scope': 'openid email profile'}
)

# Google ile giriş yapma butonuna tıklanınca çalışacak
@app.route('/login/google')
def login_google():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

# Google girişinden sonra yönlendirme
@app.route('/login/google/callback')
def google_callback():
    token = google.authorize_access_token()
    user_info = google.get('https://openidconnect.googleapis.com/v1/userinfo').json()
    
    # Kullanıcıyı veritabanında bul veya ekle
    user = User.query.filter_by(email=user_info['email']).first()
    if not user:
        # Geçici şifre oluştur (kullanıcı tarafından asla kullanılmayacak)
        temp_password = generate_password_hash("google_login_placeholder")
        user = User(email=user_info['email'], password_hash=temp_password, is_admin=False)
        db.session.add(user)
        db.session.commit()
    
    login_user(user)
    flash("Google ile giriş yapıldı!", "success")
    return redirect(url_for('home'))

# Çıkış Yapma
@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()  #Oturumu temizle
    flash('Başarıyla çıkış yaptınız.', 'info')
    return redirect(url_for('home'))

# Veri Üretimi
@app.route('/generate', methods=['GET', 'POST'])
@login_required
def generate_data():
    print("Current User:", current_user)  # Test için ekleyelim
    print("Is Authenticated:", current_user.is_authenticated)  # Kullanıcı giriş yaptı mı?
    
    if request.method == 'POST':
        rows = int(request.form['rows'])
        columns = int(request.form['columns'])
        file_format = request.form['format']

        data_types = []

        # **Veri Türlerini Doğru Al**
        for i in range(columns):
            data_type_key = f'data_type_{i}'
            if data_type_key in request.form:
                data_types.append(request.form[data_type_key])
            else:
                flash(f"Sütun {i+1} için veri türü seçilmemiş!", "danger")
                return redirect(url_for('generate_data'))

        # **Hata Kontrolü: Kullanıcı tüm veri türlerini seçti mi?**
        if len(data_types) != columns:
            flash("Lütfen tüm sütunlar için bir veri türü seçin!", "danger")
            return redirect(url_for('generate_data'))

        data = generate_fake_data(rows, columns, data_types)

        # **Veriyi Oluştur**
        #for i in range(columns):
        #   if data_types[i] == 'integer':
        #       data[f'Sütun_{i+1}'] = np.random.randint(0, 100, rows)
        #  elif data_types[i] == 'float':
        #        data[f'Sütun_{i+1}'] = np.random.uniform(0, 100, rows)
        #    elif data_types[i] == 'text':
        #        data[f'Sütun_{i+1}'] = [''.join(random.choices(string.ascii_letters, k=5)) for _ in range(rows)]
        #    elif data_types[i] == 'date':
        #        data[f'Sütun_{i+1}'] = [datetime.now().strftime('%Y-%m-%d') for _ in range(rows)]

        # **Dosya Oluştur ve Kullanıcıya Gönder**
        output = BytesIO()
        if file_format == 'csv':
            data.to_csv(output, index=False)
            output.seek(0)
            return send_file(output, as_attachment=True, download_name='generated_data.csv', mimetype='text/csv')
        elif file_format == 'xlsx':
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                data.to_excel(writer, index=False)
            output.seek(0)
            return send_file(output, as_attachment=True, download_name='generated_data.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    return render_template('generate.html')


#kullanıcıları listeleme
@app.route('/users')
@login_required
def list_users():
    if not current_user.is_admin:  # Eğer kullanıcı admin değilse
        flash("Bu sayfaya erişiminiz yok!", "danger")
        return redirect(url_for('home'))  # Ana sayfaya yönlendir
    
    search_query = request.args.get('query', '')

    if search_query:
        users = User.query.filter(User.email.ilike(f"%{search_query}%")).all()
    else:
        users = User.query.all()

    return render_template('users.html', users=users, query=search_query)

@app.route('/make_admin/<email>')
@login_required
def make_admin(email):
    if not current_user.is_admin:
        flash("Bu işlemi yapma yetkiniz yok!", "danger")
        return redirect(url_for('home'))

    user = User.query.filter_by(email=email).first()
    if user:
        user.is_admin = True
        db.session.commit()
        flash(f"{email} artık admin!", "success")
    else:
        flash("Kullanıcı bulunamadı!", "warning")
    return redirect(url_for('home'))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
