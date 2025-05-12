from flask import Flask, render_template, request, redirect, url_for, flash, send_file,session,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
from flask_migrate import Migrate
import numpy as np
import os
from io import BytesIO
from datetime import datetime,timedelta
import random
import string
from faker import Faker
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import re  # Regex kütüphanesi
from sdv.single_table import CTGANSynthesizer
import tempfile
import sdv
from sdv.metadata import SingleTableMetadata
from flask_wtf.csrf import CSRFProtect #İzinsiz işlemleri engellemek için
import secrets
from datetime import timedelta
from flask import send_file
import uuid
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from itsdangerous import TimedSerializer
from itsdangerous import URLSafeTimedSerializer


app = Flask(__name__)
csrf = CSRFProtect(app)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config['SESSION_COOKIE_SECURE'] = False        # HTTPS ile çalışır (yayın ortamında aktif)
app.config['SESSION_COOKIE_HTTPONLY'] = True      # JavaScript erişemez
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'     # CSRF koruması
# Profil fotoğrafı yükleme klasörü ve izin verilen dosya türleri
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB sınırı
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'database.db')
app.permanent_session_lifetime = timedelta(minutes=30)  # Oturum süresi (30 dakika)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@app.route('/get_all_data_types')
@login_required
def get_all_data_types():
    """Tüm veri türlerini kategorileriyle birlikte döndürür"""
    return jsonify(DATA_CATEGORIES)

@app.route('/preview_data', methods=['POST'])
@login_required
def preview_data():
    """Form verilerinden örnek veri üretip önizleme olarak döndürür"""
    try:
        #Form verilerini al
        rows=5      #önizleme için sabit 5 satır
        columns=int(request.form['columns'])
        data_types=[]

        #veri türlerini topla
        for i in range(columns):
            data_type_key=f'data_type_{i}'
            if data_type_key not in request.form:
                raise ValueError(f"Sütun {i+1} için veri türü seçilmemiş!")
            data_types.append(request.form[data_type_key])

        # Veri üretimi
        data= generate_fake_data(rows, columns, data_types)

        #DataFrame'i dict'e çevir
        preview_data=data.to_dict('records')

        return jsonify({
            'success': True,
            'preview': preview_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Veri Kategorileri ve Türleri
DATA_CATEGORIES = {
    "Kişisel": [
        {"id": "name", "label": "İsim-Soyisim"},
        {"id": "first_name", "label": "İsim"},
        {"id": "last_name", "label": "Soyisim"},
        {"id": "username", "label": "Kullanıcı Adı"},
        {"id": "email", "label": "E-posta"},
        {"id": "phone", "label": "Telefon"},
        {"id": "ssn", "label": "TC Kimlik No"},
        {"id": "job", "label": "Meslek"},
        {"id": "gender", "label": "Cinsiyet"},
        {"id": "birth_date", "label": "Doğum Tarihi"}
    ],
    "Adres": [
        {"id": "address", "label": "Tam Adres"},
        {"id": "street", "label": "Sokak"},
        {"id": "city", "label": "Şehir"},
        {"id": "state", "label": "İlçe"},
        {"id": "country", "label": "Ülke"},
        {"id": "postcode", "label": "Posta Kodu"},
        {"id": "latitude", "label": "Enlem"},
        {"id": "longitude", "label": "Boylam"},
        {"id": "timezone", "label": "Saat Dilimi"}
    ],
    "Finans": [
        {"id": "credit_card", "label": "Kredi Kartı"},
        {"id": "card_provider", "label": "Kart Sağlayıcı"},
        {"id": "iban", "label": "IBAN"},
        {"id": "bic", "label": "BIC/SWIFT"},
        {"id": "currency", "label": "Para Birimi"},
        {"id": "cryptocurrency", "label": "Kripto Para Birimi"},
        {"id": "amount", "label": "Tutar"},
        {"id": "transaction_type", "label": "İşlem Türü"}
    ],
    "İnternet": [
        {"id": "url", "label": "Web Adresi"},
        {"id": "domain", "label": "Domain"},
        {"id": "ipv4", "label": "IPv4"},
        {"id": "ipv6", "label": "IPv6"},
        {"id": "mac_address", "label": "MAC Adresi"},
        {"id": "user_agent", "label": "User Agent"},
        {"id": "uri", "label": "URI"}
    ],
    "Şirket": [
        {"id": "company", "label": "Şirket Adı"},
        {"id": "company_suffix", "label": "Şirket Türü"},
        {"id": "ein", "label": "Vergi No"},
        {"id": "duns", "label": "DUNS Numarası"},
        {"id": "vat", "label": "KDV No"}
    ],
    "Zaman": [
        {"id": "date", "label": "Tarih"},
        {"id": "time", "label": "Saat"},
        {"id": "datetime", "label": "Tarih-Saat"},
        {"id": "timestamp", "label": "Zaman Damgası"},
        {"id": "century", "label": "Yüzyıl"},
        {"id": "year", "label": "Yıl"}
    ],
    "Temel": [
        {"id": "integer", "label": "Tam Sayı"},
        {"id": "float", "label": "Ondalık Sayı"},
        {"id": "text", "label": "Metin"},
        {"id": "boolean", "label": "Evet/Hayır"},
        {"id": "color", "label": "Renk"},
        {"id": "uuid", "label": "UUID"},
        {"id": "md5", "label": "MD5"},
        {"id": "sha1", "label": "SHA1"},
        {"id": "sha256", "label": "SHA256"}
    ]
}

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

Talisman(app, content_security_policy=None)

class LoginAttempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), nullable=False)
    ip_address = db.Column(db.String(45), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    success = db.Column(db.Boolean, default=False)

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# CSRF hata yakalayıcı ekleyin
@app.errorhandler(400)
def handle_csrf_error(e):
    flash("CSRF doğrulama hatası. Lütfen sayfayı yenileyip tekrar deneyin.", "danger")
    return redirect(url_for('profile'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Kullanıcı Modeli
class User(db.Model, UserMixin):  # UserMixin ekledik
    __table_args__={'extend_existing': True}  # Mevcut tabloyu genişlet
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  
    profile_picture = db.Column(db.String(120), nullable=True)
    session_token = db.Column(db.String(100), 
                              unique=True,
                              nullable=True,
                              name='uq_user_session_token'  
                              )

    def get_session_token(self):
        if not self.session_token:
            self.session_token = secrets.token_urlsafe(32)
            db.session.commit()
        return self.session_token

    def invalidate_session_token(self):
        self.session_token = secrets.token_urlsafe(32)
        db.session.commit()

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password)  # Şifre hashleme

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)  # Şifre doğrulama

    def get_reset_token(self, expires_sec=1800):
        s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
        return s.dumps({'user_id': self.id}, salt='password-reset-salt')

    @staticmethod
    def verify_reset_token(token):
        s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token, salt='password-reset-salt', max_age=1800)['user_id']
        except:
            return None
        return User.query.get(user_id)

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

# Kişisel veri üretimi için özel fonksiyon
def generate_personal_data(fake,data_types,rows):
    if data_types=='name':
        return [fake.name() for _ in range(rows)]
    elif data_types=='first_name':
        return [fake.first_name() for _ in range(rows)]
    elif data_types=='last_name':
        return [fake.last_name() for _ in range(rows)]
    elif data_types=='username':
        return [fake.user_name() for _ in range(rows)]
    elif data_types=='email':
        return [fake.email() for _ in range(rows)]
    elif data_types=='phone':
        return [fake.phone_number() for _ in range(rows)]
    elif data_types=='ssn':
        return [fake.ssn() for _ in range(rows)]
    elif data_types=='job':
        return [fake.job() for _ in range(rows)]
    elif data_types=='gender':
        return[fake.random_element(elements=('Erkek', 'Kadın')) for _ in range(rows)]
    elif data_types=='birth_date':
        return[fake.date_of_birth(minimum_age=18, maximum_age=122).strftime('%Y-%m-%d') for _ in range(rows)]
    else:
        raise ValueError(f"Desteklenmeyen veri türü: {data_types}")

#Adres veri üretimi için özel fonksiyon
def generate_address_data(fake,data_types,rows):
    if data_types=='address':
        return [fake.address() for _ in range(rows)]
    elif data_types=='street':
        return [fake.street_address() for _ in range(rows)]
    elif data_types=='city':
        return [fake.city() for _ in range(rows)]
    elif data_types=='state':
        return [fake.state() for _ in range(rows)]
    elif data_types=='country':
        return [fake.country() for _ in range(rows)]
    elif data_types=='postcode':
        return [fake.postcode() for _ in range(rows)]
    elif data_types=='latitude':
        return [str(fake.latitude()) for _ in range(rows)]
    elif data_types=='longitude':
        return [str(fake.longitude()) for _ in range(rows)]
    elif data_types=='timezone':
        return [fake.timezone() for _ in range(rows)]
    else:
        raise ValueError(f"Desteklenmeyen veri türü: {data_types}")
    
#Finans veri üretimi için özel fonksiyon
def generate_financial_data(fake,data_types,rows):
    if data_types=='credit_card':
        return [fake.credit_card_number() for _ in range(rows)]
    elif data_types=='card_provider':
        return [fake.credit_card_provider() for _ in range(rows)]
    elif data_types=='iban':
        return [fake.iban() for _ in range(rows)]
    elif data_types=='bic':
        return [fake.bic() for _ in range(rows)]
    elif data_types=='currency':
        return [fake.currency()[0] for _ in range(rows)]
    elif data_types=='cryptocurrency':
        return [fake.cryptocurrency()[0] for _ in range(rows)]
    elif data_types=='amount':
        return [round(fake.pyfloat(left_digits=5, right_digits=2, positive=True), 2) for _ in range(rows)]
    elif data_types=='transaction_type':
        return [fake.random_element(elements=('debit', 'credit')) for _ in range(rows)]
    else:
        raise ValueError(f"Desteklenmeyen veri türü: {data_types}")
    
# İnternet veri üretimi için özel fonksiyon
def generate_internet_data(fake,data_types,rows):
    if data_types=='url':
        return [fake.url() for _ in range(rows)]
    elif data_types=='domain':
        return [fake.domain_name() for _ in range(rows)]
    elif data_types=='ipv4':
        return [fake.ipv4() for _ in range(rows)]
    elif data_types=='ipv6':
        return [fake.ipv6() for _ in range(rows)]
    elif data_types=='mac_address':
        return [fake.mac_address() for _ in range(rows)]
    elif data_types=='user_agent':
        return [fake.user_agent() for _ in range(rows)]
    elif data_types=='uri':
        return [fake.uri() for _ in range(rows)]
    else:
        raise ValueError(f"Desteklenmeyen veri türü: {data_types}")

# Şirket veri üretimi için özel fonksiyon
def generate_company_data(fake,data_types,rows):
    if data_types=='company':
        return [fake.company() for _ in range(rows)]
    elif data_types=='company_suffix':
        return [fake.company_suffix() for _ in range(rows)]
    elif data_types=='ein':
        return [fake.ein() for _ in range(rows)]
    elif data_types=='duns':
        return [fake.duns_number() for _ in range(rows)]
    elif data_types=='vat':
        return [f"TR{fake.random_number(digits=10)}" for _ in range(rows)]
    else:
        raise ValueError(f"Desteklenmeyen veri türü: {data_types}")

# Zaman veri üretimi için özel fonksiyon
def generate_time_data(fake,data_types,rows):
    if data_types=='date':
        return [fake.date() for _ in range(rows)]
    elif data_types=='time':
        return [fake.time() for _ in range(rows)]
    elif data_types=='datetime':
        return [fake.date_time() for _ in range(rows)]
    elif data_types=='timestamp':
        return [fake.unix_time() for _ in range(rows)]
    elif data_types=='century':
        return [fake.century() for _ in range(rows)]
    elif data_types=='year':
        return [fake.year() for _ in range(rows)]
    else:
        raise ValueError(f"Desteklenmeyen veri türü: {data_types}")
    
# Temel veri üretimi için özel fonksiyon
def generate_basic_data(fake,data_types,rows):
    if data_types=='integer':
        return [fake.random_int(min=0, max=100) for _ in range(rows)]
    elif data_types=='float':
        return [fake.random.uniform(0, 100) for _ in range(rows)]
    elif data_types=='text':
        return [fake.text(max_nb_chars=20) for _ in range(rows)]
    elif data_types=='boolean':
        return [fake.boolean() for _ in range(rows)]
    elif data_types=='color':
        return [fake.color_name() for _ in range(rows)]
    elif data_types=='uuid':
        return [str(fake.uuid4()) for _ in range(rows)]
    elif data_types=='md5':
        return [fake.md5() for _ in range(rows)]
    elif data_types=='sha1':
        return [fake.sha1() for _ in range(rows)]
    elif data_types=='sha256':
        return [fake.sha256() for _ in range(rows)]
    else:
        raise ValueError(f"Desteklenmeyen veri türü: {data_types}")
    
# Veri üretimi için ana fonksiyon
def generate_fake_data(rows, columns, data_types):
    #fake = Faker(['tr_TR', 'en_US','ja_JP','fr_FR','de_DE'])  # çeşitli dillerde veri üretimi
    fake = Faker(['tr_TR'])  # Türkçe veri üretimi
    data = pd.DataFrame()

    for i in range(columns):
        column_name= f'Sütun_{i+1}'
        data_type = data_types[i]

        try:
            if data_type in [type["id"] for type in DATA_CATEGORIES["Kişisel"]]:
                data[column_name] = generate_personal_data(fake, data_type, rows)
            elif data_type in [type["id"] for type in DATA_CATEGORIES["Adres"]]:
                data[column_name] = generate_address_data(fake, data_type, rows)
            elif data_type in [type["id"] for type in DATA_CATEGORIES["Finans"]]:
                data[column_name] = generate_financial_data(fake, data_type, rows)
            elif data_type in [type["id"] for type in DATA_CATEGORIES["İnternet"]]:
                data[column_name] = generate_internet_data(fake, data_type, rows)
            elif data_type in [type["id"] for type in DATA_CATEGORIES["Şirket"]]:
                data[column_name] = generate_company_data(fake, data_type, rows)
            elif data_type in [type["id"] for type in DATA_CATEGORIES["Zaman"]]:
                data[column_name] = generate_time_data(fake, data_type, rows)
            elif data_type in [type["id"] for type in DATA_CATEGORIES["Temel"]]:
                data[column_name] = generate_basic_data(fake, data_type, rows)
            else:
                raise ValueError(f"Desteklenmeyen veri türü: {data_type}")
        except ValueError as e:
            raise ValueError(f"Sütun {i+1} için veri üretim hatası: {str(e)}")
    return data
#about sayfası
@app.route('/about')
def about():
    return render_template('about.html')

#iletişim bilgileri
@app.route('/contact')
def contact():
    return render_template('contact.html')

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

            # Dosya kaydı
            filename = f"model_{current_user.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{file_format}"
            folder_path = os.path.join('generated_files')
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, filename)

            if file_format == 'csv':
                synthetic_data.to_csv(file_path, index=False)
                mimetype = 'text/csv'
            elif file_format == 'xlsx':
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    synthetic_data.to_excel(writer, index=False)
                mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

            # Veritabanına kayıt: UserActivity
            activity = UserActivity(
                user_id=current_user.id,
                action='veri üretimi',
                data_type='model',
                row_count=num_rows,
                column_count=len(synthetic_data.columns),
                file_format=file_format
            )

            # Veritabanına kayıt: Production
            production = Production(
                user_id=current_user.id,
                date=datetime.utcnow().date(),
                type='model',
                row=num_rows,
                column=len(synthetic_data.columns),
                format=file_format,
                file_path=file_path
            )

            db.session.add(activity)
            db.session.add(production)
            db.session.commit()

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
                return send_file(file_path, as_attachment=True, download_name=filename, mimetype=mimetype)

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
        username = request.form['username']
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
        new_user = User(username=username,email=email, password_hash=hashed_password, is_admin=is_admin)

        # Veritabanına kaydet
        db.session.add(new_user)
        db.session.commit()

        flash("Başarıyla kayıt oldunuz! Giriş yapabilirsiniz.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


# Giriş Yapma
@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")  # 1 dakikada en fazla 5 deneme
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        ip_address = request.remote_addr  # Kullanıcının IP adresini al

        # Başarısız giriş denemelerini kontrol et
        failed_attempts = LoginAttempt.query.filter_by(
            email=email,
            ip_address=ip_address,
            success=False,
        ).filter(
            LoginAttempt.timestamp >= (datetime.utcnow() - timedelta(minutes=15))
        ).count()

        if failed_attempts >= 5:
            flash("Çok fazla başarısız deneme. Lütfen 15 dakika sonra tekrar deneyin.", "danger")
            return redirect(url_for('login'))
        
        user = User.query.filter_by(email=email).first()  # Email ile sorgula
        attempt = LoginAttempt(email=email, ip_address=ip_address, success=False)

        # Form validasyonu
        if not email or not password:
            flash("E-posta ve şifre zorunludur!", "danger")
            db.session.add(attempt)
            db.session.commit()
            return redirect(url_for('login'))
        
        remember = request.form.get('remember', False) == 'on'
        if user and bcrypt.check_password_hash(user.password_hash, password):
            attempt.success = True
            db.session.add(attempt)
            login_user(user, remember=remember)
            session.permanent = True
            flash(f'Hoş geldiniz, {user.username}!', 'success')
            next_page = request.args.get('next')
            db.session.commit()
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            db.session.add(attempt)
            if user:
                flash("Hatalı şifre! Lütfen tekrar deneyin.", "danger")
            else:
                flash("Bu email adresi ile kayıtlı bir hesap bulunamadı!", "danger")
            db.session.commit()
            return redirect(url_for('login'))
        # print("Kullanıcı nesnesi:", user)  # Test çıktısı
        # if user:
        #     print("Veritabanındaki hash:", user.password_hash)  # Kullanıcının hash'lenmiş şifresini gör
        #     print("Girilen şifre:", password)  # Kullanıcının girdiği şifreyi gör
        #     print("Hash Check Result:", bcrypt.check_password_hash(user.password_hash, password))  # Doğrulama sonucunu yazdır
        
        
        # if user and bcrypt.check_password_hash(user.password_hash, password):  # Hash doğrulama
        #     session.clear()               #Session fixation önle
        #     session.permanent = True     #Oturumu zamanlayalım
        #     login_user(user)
        #     flash('Giriş başarılı!', 'success')
        #     return redirect(url_for('generate_data'))
            
        # else:
        #     flash('Giriş başarısız. Lütfen bilgilerinizi kontrol edin.', 'danger')
    
    return render_template('login.html')
login_manager.login_view = 'login'  


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
    session['user_id'] = user.id
    return redirect(url_for('home'))

# Çıkış Yapma
@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()  #Oturumu temizle
    flash('Başarıyla çıkış yaptınız.', 'info')
    return redirect(url_for('home'))

def save_and_send_file(data, file_format):
            output = BytesIO()
            if file_format == 'csv':
                data.to_csv(output, index=False)
            elif file_format == 'xlsx':
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    data.to_excel(writer, index=False)
            output.seek(0)
            return output

# Veri Üretimi
@app.route('/generate', methods=['GET', 'POST'])
@login_required
def generate_data():
    print("Current User:", current_user)  
    print("Is Authenticated:", current_user.is_authenticated)  # Kullanıcı giriş yaptı mı?
    
    if request.method == 'POST':
        try:
            # Form verilerini al ve doğrula
            rows = int(request.form['rows'])
            columns = int(request.form['columns'])
            file_format = request.form['format']

            # Satır ve sütun validasyonu
            if rows <= 0 or columns <= 0:
                flash("Satır ve sütun sayısı 0'dan büyük olmalıdır!", "danger")
                return redirect(url_for('generate_data'))

            if rows > 1000000:  # Maksimum satır limiti
                flash("Maksimum 1.000.000 satır üretebilirsiniz!", "danger")
                return redirect(url_for('generate_data'))

            # Veri türlerini topla
            data_types = []
            for i in range(columns):
                data_type_key = f'data_type_{i}'
                if data_type_key not in request.form:
                    flash(f"Sütun {i+1} için veri türü seçilmemiş!", "danger")
                    return redirect(url_for('generate_data'))
                data_types.append(request.form[data_type_key])

            # Veri üretimi
            data = generate_fake_data(rows, columns, data_types)

            # Önizleme verisi oluştur (ilk 5 satır)
            preview_data = data.head()
            
            # Dosyayı oluştur
            output = save_and_send_file(data, file_format)

            # Dosya kaydı için gerekli bilgiler
            filename = f"random_{current_user.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{file_format}"
            folder_path = os.path.join('generated_files')
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, filename)

            # Dosyayı kaydet
            with open(file_path, 'wb') as f:
                f.write(output.getvalue())

            # Aktivite kaydı
            activity = UserActivity(
                user_id=current_user.id,
                action='veri üretimi',
                data_type='random',
                row_count=rows,
                column_count=columns,
                file_format=file_format
            )

            # Üretim kaydı
            production = Production(
                user_id=current_user.id,
                date=datetime.utcnow().date(),
                type='random',
                row=rows,
                column=columns,
                format=file_format,
                file_path=file_path
            )

            # Veritabanına kaydet
            db.session.add(activity)
            db.session.add(production)
            db.session.commit()

            # Dosyayı kullanıcıya gönder
            mimetype = 'text/csv' if file_format == 'csv' else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            return send_file(
                BytesIO(output.getvalue()),
                as_attachment=True,
                download_name=filename,
                mimetype=mimetype
            )

        except ValueError as e:
            flash(f"Veri üretimi sırasında bir hata oluştu: {str(e)}", "danger")
            return redirect(url_for('generate_data'))
        except Exception as e:
            flash(f"Beklenmeyen bir hata oluştu: {str(e)}", "danger")
            return redirect(url_for('generate_data'))

    # GET isteği için template'i render et
    return render_template('generate.html', categories=DATA_CATEGORIES)

@app.route('/get_data_types/<category>')
@login_required
def get_data_types(category):
    if category in DATA_CATEGORIES:
        return jsonify(DATA_CATEGORIES[category])
    return jsonify([])

@app.route('/advanced_generate', methods=['POST'])
@login_required
def advanced_generate():
    try:
        data_types = request.form.getlist('data_types[]')
        rows = int(request.form.get('rows', 1000))
        columns = len(data_types)
        
        if not data_types:
            raise ValueError("En az bir veri türü seçmelisiniz")
            
        data = generate_fake_data(rows, columns, data_types)
        
        return jsonify({
            'success': True,
            'preview': data.head().to_dict('records')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

#kullanıcı aktiviteleri modeli
class UserActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    action = db.Column(db.String(100), nullable=False)  #kullanıcının yaptığı işlem (örneğin: 'veri üretimi', 'giriş', 'çıkış')
    date_created = db.Column(db.DateTime, default=datetime.utcnow) # Aktivite tarihi
    data_type = db.Column(db.String(50))  # örnek: 'random', 'model'
    row_count = db.Column(db.Integer)
    column_count = db.Column(db.Integer)
    file_format = db.Column(db.String(10))  # örnek: 'csv', 'xlsx'
    user = db.relationship('User', backref=db.backref('activities', lazy=True))


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

class Production(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    date = db.Column(db.Date)
    type = db.Column(db.String(50))
    row = db.Column(db.Integer)
    column = db.Column(db.Integer)
    format = db.Column(db.String(10))
    file_path = db.Column(db.String(200))

    user = db.relationship('User', backref=db.backref('productions', lazy=True))

#profil sayfası
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    # Kullanıcının giriş yapıp yapmadığını kontrol et
    if not current_user.is_authenticated:
        flash("Lütfen giriş yapın!", "danger")
        return redirect(url_for('login'))

    # Üretim verilerini sorgula
    productions_query = Production.query.filter_by(user_id=current_user.id)
    filtered = False  # Filtreleme yapılıp yapılmadığını kontrol etmek için
    
    if request.method == 'POST':
        form_type = request.form.get('form_type')
        
        # Profil fotoğrafı yükleme işlemi
        if form_type == 'profile_photo' and 'profile_picture' in request.files:
            file = request.files['profile_picture']
            if file and file.filename and allowed_file(file.filename):
                try:
                    # Benzersiz dosya adı oluştur
                    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                    # Uploads klasörünü kontrol et ve oluştur
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

                    # Eski fotoğraf varsa sil
                    if current_user.profile_picture:
                        old_path = os.path.join(app.config['UPLOAD_FOLDER'], current_user.profile_picture)
                        try:
                            if os.path.exists(old_path):
                                os.remove(old_path)
                        except Exception as e:
                            flash(f"Eski dosya silinirken bir hata oluştu: {str(e)}", "warning")

                    # Yeni dosyayı kaydet
                    file.save(file_path)
                    current_user.profile_picture = f'uploads/{filename}'
                    try:
                        db.session.commit()
                        flash("Profil fotoğrafınız başarıyla yüklendi.", "success")
                    except Exception as e:
                        flash(f"Veritabanı güncellemesi sırasında bir hata oluştu: {str(e)}", "danger")
                        return redirect(url_for('profile'))
                except Exception as e:
                    flash(f"Dosya yüklenirken bir hata oluştu: {str(e)}", "danger")
            else:
                flash("Geçersiz dosya formatı. Lütfen PNG, JPG, JPEG veya GIF yükleyin.", "warning")

        # Tarih filtreleme işlemi
        elif form_type == 'date_filter':
            try:
                start_date = request.form.get('start_date')
                end_date = request.form.get('end_date')
                if start_date and end_date:
                    start_date = datetime.strptime(start_date, '%Y-%m-%d')
                    end_date = datetime.strptime(end_date, '%Y-%m-%d')
                    # Bitiş tarihine 23:59:59 ekleyerek günün sonunu dahil et
                    end_date = end_date + timedelta(days=1, microseconds=-1)
                    productions_query = productions_query.filter(
                        Production.date >= start_date,
                        Production.date <= end_date
                    )
                    filtered = True  # Filtreleme yapıldığını işaretle
                    print("Filtrelenmiş Veriler:", productions_query.all())
                else:
                    flash("Lütfen geçerli bir tarih aralığı girin.", "warning")
            except ValueError:
                flash("Geçersiz tarih formatı. Lütfen doğru bir tarih girin.", "warning")

    # Üretimleri tarihe göre sırala
    productions = productions_query.order_by(Production.date.desc()).all()
    total_production = len(productions)

    # Filtreleme yapıldıysa ve sonuç boşsa mesaj göster
    if filtered and not productions:
        flash("Bu tarihler arasında veri üretimi yapmadınız.", "info")

    return render_template('profile.html',
                         current_user=current_user,
                         productions=productions,
                         total_production=total_production,
                         start_date=request.form.get('start_date', ''),
                         end_date=request.form.get('end_date', ''))

@app.route('/download/<int:id>')
def download(id):
    production = Production.query.get_or_404(id)
    # Dosya yolu: örnek olarak path'i modeline kaydetmiş olalım
    return send_file(production.file_path, as_attachment=True)

@app.route('/delete/<int:id>')
def delete(id):
    production = Production.query.get_or_404(id)
    db.session.delete(production)
    db.session.commit()
    return redirect(url_for('profile'))

@app.route('/account/settings')
@login_required
def account_settings():
    return render_template('account_settings.html')

@app.route('/account/delete', methods=['GET', 'POST'])
@login_required
def delete_account():
    if request.method == 'POST':
        # Kullanıcının ürettiği verileri sil
        Production.query.filter_by(user_id=current_user.id).delete()
        
        # Kullanıcı profilini sil
        db.session.delete(current_user)
        db.session.commit()
        
        # Oturumu sonlandır
        logout_user()
        flash('Hesabınız başarıyla silindi.', 'success')
        return redirect(url_for('home'))
    
    return render_template('delete_account.html')

@app.route('/account/update-profile', methods=['POST'])
@login_required
def update_profile():
    if request.method == 'POST':
        # Email güncelleme düzeltmesi
        new_email = request.form.get('email')
        if new_email and new_email != current_user.email:
            if User.query.filter_by(email=new_email).first():  # email() yerine email=
                flash('Bu email adresi zaten kullanılıyor!', 'danger')
                return redirect(url_for('account_settings'))
            current_user.email = new_email
            
        # Username güncelleme
        new_username = request.form.get('username')
        if new_username and new_username != current_user.username:
            if User.query.filter_by(username=new_username).first():
                flash('Bu kullanıcı adı zaten kullanılıyor!', 'danger')
                return redirect(url_for('account_settings'))
            current_user.username = new_username

        # Email güncelleme
        new_email = request.form.get('email')
        if new_email and new_email != current_user.email:
            if User.query.filter_by(email=new_email).first():
                flash('Bu email adresi zaten kullanılıyor!', 'danger')
                return redirect(url_for('account_settings'))
            current_user.email = new_email

        # Yeni şifre kontrolü ve güncelleme
        new_password = request.form.get('new_password')
        if new_password:
            is_strong, message = is_password_strong(new_password)
            if not is_strong:
                flash(message, 'danger')
                return redirect(url_for('account_settings'))
            current_user.set_password(new_password)

        try:
            db.session.commit()
            flash('Profil bilgileriniz başarıyla güncellendi!', 'success')
        except Exception as e:
            db.session.rollback()
            flash('Bir hata oluştu: ' + str(e), 'danger')

        return redirect(url_for('account_settings'))
@app.context_processor
def utility_processor():
    return {
        'now': datetime.now(),
        'datetime': datetime,
        'current_year': datetime.now().year
    }

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Şifre sıfırlama token'ı oluştur
            token = user.get_reset_token()
            
            # Mail gönderme işlemi burada yapılacak
            flash('Şifre sıfırlama talimatları email adresinize gönderildi.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Bu email adresi ile kayıtlı bir hesap bulunamadı.', 'danger')
    
    return render_template('forgot_password.html')

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
