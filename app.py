from flask import Flask, render_template, request, redirect, url_for, flash, send_file,session,jsonify, send_from_directory
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
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer, GaussianCopulaSynthesizer
import tempfile
import sdv
from sdv.metadata import SingleTableMetadata
from flask_wtf.csrf import CSRFProtect #İzinsiz işlemleri engellemek için
import secrets
from datetime import timedelta
import uuid
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from itsdangerous import TimedSerializer
from itsdangerous import URLSafeTimedSerializer
import io
import logging
import shutil
from threading import Thread
import time
import json
import sdmetrics
from sdmetrics.single_table import CSTest, TVComplement, BoundaryAdherence, MLPRegressor
from sdmetrics.single_table import LogisticDetection, SVCDetection 
from sdmetrics.single_table import CSTest, CorrelationSimilarity
from sdmetrics.single_table import BinaryMLPClassifier, MulticlassMLPClassifier, MLPRegressor, LinearRegression
import traceback


fake=Faker()
app = Flask(__name__)
csrf = CSRFProtect(app)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

# Klasör yapılandırmaları
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
app.config['GENERATED_FILES_FOLDER'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'generated_files')

# Klasörleri oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_FILES_FOLDER'], exist_ok=True)

app.config['SESSION_COOKIE_SECURE'] = False        # HTTPS ile çalışır (yayın ortamında aktif)
app.config['SESSION_COOKIE_HTTPONLY'] = True      # JavaScript erişemez
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'     # CSRF koruması
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB sınırı
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','csv','xlsx'}
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'database.db')
app.permanent_session_lifetime = timedelta(minutes=30)  # Oturum süresi (30 dakika)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
app.logger.setLevel(logging.DEBUG)

turkish_streets = [
    "Atatürk Caddesi", "Cumhuriyet Mahallesi", "İnönü Bulvarı", "Barış Sokak", "Gazi Mustafa Kemal Paşa Caddesi",
    "Fatih Sultan Mehmet Bulvarı", "Mevlana Sokak", "Yavuz Sultan Selim Caddesi", "Şehitler Caddesi", "Çınar Sokak",
    "Kurtuluş Caddesi", "Papatya Sokak", "Menekşe Sokak", "Gül Sokak", "Lale Sokak", "Yıldız Caddesi", "Deniz Sokak",
    "Bahçelievler Caddesi", "Zafer Caddesi", "Vatan Caddesi"
]

turkish_buildings = [
    "Yıldız Apartmanı", "Güneş Sitesi", "Çamlık Evleri", "Deniz Konakları", "Barış Rezidans", "Gökkuşağı Apartmanı",
    "Pırlanta Sitesi", "Yeşil Vadi Evleri", "Güven Apartmanı", "Mutlu Konutları", "Sevgi Sitesi", "Park Rezidans",
    "Akasya Apartmanı", "Kardelen Sitesi", "Vadi Konakları", "Göl Evleri", "Bahçe Apartmanı", "Kule Rezidans"
]

city_district_postcodes = {
    "İstanbul": {"Kadıköy": "34710", "Beşiktaş": "34353", "Üsküdar": "34662", "Şişli": "34360", "Bakırköy": "34142"},
    "Ankara": {"Çankaya": "06680", "Keçiören": "06280", "Mamak": "06470", "Yenimahalle": "06170"},
    "İzmir": {"Konak": "35250", "Karşıyaka": "35550", "Bornova": "35040", "Buca": "35390"},
    "Bursa": {"Nilüfer": "16110", "Osmangazi": "16010", "Yıldırım": "16300"},
    "Antalya": {"Muratpaşa": "07010", "Konyaaltı": "07070", "Kepez": "07060"},
    "Adana": {"Sarıçam": "01240", "Çukurova": "01170", "Seyhan": "01010", "Yüreğir": "01316"}
}

@app.route('/get_all_data_types')
@login_required
def get_all_data_types():
    """Tüm veri türlerini kategorileriyle birlikte döndürür"""
    try:
        return jsonify({
            'success': True,
            'data': DATA_CATEGORIES
        })
    except Exception as e:
        app.logger.error(f"Veri türleri getirme hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/preview_data', methods=['POST'])
@login_required
def preview_data():
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Kullanıcı oturum açmamış!'}), 401

    try:
        rows = 5  # önizleme için sabit 5 satır
        # columns = int(request.form['columns'])  # BUNU KULLANMA!
        data_types = []

        # Sadece gönderilen data_type_X parametrelerini topla
        i = 0
        while True:
            data_type_key = f'data_type_{i}'
            if data_type_key not in request.form:
                break
            data_types.append(request.form[data_type_key])
            i += 1

        columns = len(data_types)  # columns, seçili veri türü sayısı olmalı

        if rows <= 0 or columns <= 0:
            return jsonify({'success': False, 'error': 'Geçersiz satır veya sütun sayısı!'}), 400

        data = generate_fake_data(rows, columns, data_types)

        if data.isnull().values.any():
            data = data.fillna("")

        preview_data = data.head().to_dict('records')

        return jsonify({
            'success': True,
            'preview': preview_data
        })
    except Exception as e:
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/download_data', methods=['POST'])
@login_required
def download_data():
    try:
        rows = int(request.form.get('rows', 0))
        # columns = int(request.form.get('columns', 0))  # BUNU KULLANMA!
        data_types = []
        i = 0
        while True:
            data_type_key = f'data_type_{i}'
            if data_type_key not in request.form:
                break
            data_types.append(request.form[data_type_key])
            i += 1

        columns = len(data_types)

        if rows <= 0 or columns <= 0:
            return jsonify({'success': False, 'error': 'Geçersiz satır veya sütun sayısı!'}), 400

        data = generate_fake_data(rows, columns, data_types)

        output = io.StringIO()
        data.to_csv(output, index=False)
        output.seek(0)

        # --- SADECE BURADA VERİTABANINA KAYIT EKLE ---
        activity = UserActivity(
            user_id=current_user.id,
            action='veri üretimi',
            data_type='random',
            row_count=rows,
            column_count=columns,
            file_format='csv'
        )
        production = Production(
            user_id=current_user.id,
            date=datetime.utcnow().date(),
            type='random',
            row=rows,
            column=columns,
            format='csv',
            file_path=''  # Dosya kaydediliyorsa yolunu ekle
        )
        db.session.add(activity)
        db.session.add(production)
        db.session.commit()
        # --------------------------------------------

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='generated_basic_data.csv'
        )
    except Exception as e:
        app.logger.error(f"Download data error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Veri Kategorileri ve Türleri
DATA_CATEGORIES = {
    "Kişisel": [
        {"id": "name", "label": "İsim-Soyisim", "description": "Rastgele bir isim ve soyisim üretir."},
        {"id": "first_name", "label": "İsim", "description": "Türkçe rastgele isim üretir."},
        {"id": "last_name", "label": "Soyisim", "description": "Türkçe rastgele soyisim üretir."},
        {"id": "username", "label": "Kullanıcı Adı", "description": "Rastgele kullanıcı adı üretir."},
        {"id": "email", "label": "E-posta", "description": "Geçerli bir e-posta adresi üretir."},
        {"id": "phone", "label": "Telefon", "description": "Türkiye formatında rastgele telefon numarası üretir."},
        {"id": "ssn", "label": "TC Kimlik No", "description": "Geçerli formatta rastgele TC kimlik numarası üretir."},
        {"id": "job", "label": "Meslek", "description": "Rastgele bir meslek üretir."},
        {"id": "gender", "label": "Cinsiyet", "description": "Rastgele cinsiyet üretir."},
        {"id": "birth_date", "label": "Doğum Tarihi", "description": "Rastgele doğum tarihi üretir."},
        {"id": "age", "label": "Yaş", "description": "Rastgele yaş üretir."},
        {"id": "marital_status", "label": "Medeni Durum", "description": "Rastgele medeni durum üretir."},
        {"id": "education", "label": "Eğitim Seviyesi", "description": "Rastgele eğitim seviyesi üretir."},
        {"id": "income", "label": "Gelir Seviyesi", "description": "Eğitime göre mantıklı gelir seviyesi üretir."},
        {"id": "credit_score", "label": "Kredi Notu", "description": "Rastgele kredi notu üretir (300-850 arası)."},
        {"id": "passport_number", "label": "Pasaport Numarası", "description": "Rastgele pasaport numarası üretir."},
        {"id": "driver_license", "label": "Ehliyet Numarası", "description": "Rastgele ehliyet numarası üretir."},
        {"id": "twitter", "label": "Twitter Kullanıcı Adı", "description": "Rastgele Twitter kullanıcı adı üretir."},
        {"id": "linkedin", "label": "LinkedIn Profili", "description": "Rastgele LinkedIn profil adresi üretir."},
        {"id": "hobbies", "label": "Hobiler", "description": "Rastgele 1-3 hobi üretir."},
        {"id": "interests", "label": "İlgi Alanları", "description": "Rastgele 1-3 ilgi alanı üretir."}
    ],
    "Adres": [
        {"id": "address", "label": "Tam Adres", "description": "Tam adres (sokak, mahalle, ilçe, şehir) üretir."},
        {"id": "street", "label": "Sokak", "description": "Rastgele sokak adı üretir."},
        {"id": "neighborhood", "label": "Mahalle", "description": "Rastgele mahalle adı üretir."},
        {"id": "building_name", "label": "Apartman Adı", "description": "Rastgele apartman adı üretir."},
        {"id": "door_number", "label": "Kapı No", "description": "Rastgele kapı numarası üretir."},
        {"id": "district", "label": "İlçe", "description": "Rastgele ilçe adı üretir."},
        {"id": "city", "label": "Şehir", "description": "Rastgele şehir adı üretir."},
        {"id": "state", "label": "İl", "description": "Rastgele il adı üretir."},
        {"id": "country", "label": "Ülke", "description": "Ülke adı (Türkiye)."},
        {"id": "postcode", "label": "Posta Kodu", "description": "Rastgele posta kodu üretir."},
        {"id": "latitude", "label": "Enlem", "description": "Rastgele enlem üretir."},
        {"id": "longitude", "label": "Boylam", "description": "Rastgele boylam üretir."},
        {"id": "gps", "label": "GPS", "description": "Rastgele GPS koordinatı üretir."},
        {"id": "timezone", "label": "Saat Dilimi", "description": "Saat dilimi (Europe/Istanbul)."}
    ],
    "Finans": [
        {"id": "credit_card", "label": "Kredi Kartı", "description": "Rastgele kredi kartı numarası üretir."},
        {"id": "card_provider", "label": "Kart Sağlayıcı", "description": "Rastgele kart sağlayıcı (banka) üretir."},
        {"id": "card_type", "label": "Kart Tipi", "description": "Rastgele kart tipi (Visa, MasterCard vb.) üretir."},
        {"id": "card_expiry", "label": "Kart Son Kullanma", "description": "Rastgele kart son kullanma tarihi üretir."},
        {"id": "card_cvv", "label": "Kart CVV", "description": "Rastgele kart güvenlik kodu (CVV) üretir."},
        {"id": "iban", "label": "IBAN", "description": "Rastgele IBAN numarası üretir."},
        {"id": "bic", "label": "BIC/SWIFT", "description": "Rastgele BIC/SWIFT kodu üretir."},
        {"id": "bank_name", "label": "Banka Adı", "description": "Rastgele banka adı üretir."},
        {"id": "account_number", "label": "Hesap No", "description": "Rastgele banka hesap numarası üretir."},
        {"id": "balance", "label": "Bakiye", "description": "Rastgele bakiye miktarı üretir."},
        {"id": "credit_limit", "label": "Kredi Limiti", "description": "Rastgele kredi limiti üretir."},
        {"id": "currency", "label": "Para Birimi", "description": "Rastgele para birimi üretir."},
        {"id": "cryptocurrency", "label": "Kripto Para Birimi", "description": "Rastgele kripto para birimi üretir."},
        {"id": "crypto_wallet", "label": "Kripto Cüzdan", "description": "Rastgele kripto cüzdan adresi üretir."},
        {"id": "amount", "label": "Tutar", "description": "Rastgele tutar ve para birimi üretir."},
        {"id": "transaction_type", "label": "İşlem Türü", "description": "Rastgele finansal işlem türü üretir (debit, credit, transfer vb.)."}
    ],
    "İnternet": [
        {"id": "url", "label": "Web Adresi", "description": "Rastgele web adresi (URL) üretir."},
        {"id": "domain", "label": "Domain", "description": "Rastgele domain adı üretir."},
        {"id": "email_provider", "label": "E-posta Servisi", "description": "Rastgele e-posta servis sağlayıcı üretir."},
        {"id": "ipv4", "label": "IPv4", "description": "Rastgele IPv4 adresi üretir."},
        {"id": "ipv6", "label": "IPv6", "description": "Rastgele IPv6 adresi üretir."},
        {"id": "mac_address", "label": "MAC Adresi", "description": "Rastgele MAC adresi üretir."},
        {"id": "user_agent", "label": "User Agent", "description": "Rastgele tarayıcı user agent bilgisi üretir."},
        {"id": "browser", "label": "Tarayıcı", "description": "Rastgele tarayıcı adı üretir."},
        {"id": "os", "label": "İşletim Sistemi", "description": "Rastgele işletim sistemi adı üretir."},
        {"id": "port", "label": "Port", "description": "Rastgele port numarası üretir."},
        {"id": "ssl_serial", "label": "SSL Seri No", "description": "Rastgele SSL sertifika seri numarası üretir."},
        {"id": "sosyal_medya", "label": "Sosyal Medya", "description": "Rastgele sosyal medya platformu adı üretir."},
        {"id": "sosyal_medya_kullanıcı", "label": "Sosyal Medya Kullanıcı Adı", "description": "Rastgele sosyal medya kullanıcı adı üretir."},
        {"id": "uri", "label": "URI", "description": "Rastgele URI üretir."}
    ],
    "Şirket": [
        {"id": "company", "label": "Şirket Adı", "description": "Rastgele şirket adı üretir."},
        {"id": "company_suffix", "label": "Şirket Türü", "description": "Rastgele şirket türü/soneki üretir (A.Ş., Ltd. Şti. vb.)."},
        {"id": "ein", "label": "Vergi No", "description": "Rastgele vergi numarası (EIN) üretir."},
        {"id": "duns", "label": "DUNS Numarası", "description": "Rastgele DUNS numarası üretir."},
        {"id": "vat", "label": "KDV No", "description": "Rastgele KDV numarası üretir."},
        {"id": "company_address", "label": "Şirket Adresi", "description": "Rastgele şirket adresi üretir."},
        {"id": "company_phone", "label": "Şirket Telefonu", "description": "Rastgele şirket telefonu üretir."},
        {"id": "company_email", "label": "Şirket E-posta", "description": "Rastgele şirket e-posta adresi üretir."},
        {"id": "year_founded", "label": "Kuruluş Yılı", "description": "Rastgele şirket kuruluş yılı üretir."},
        {"id": "sector", "label": "Sektör", "description": "Rastgele sektör adı üretir."},
        {"id": "employee_count", "label": "Çalışan Sayısı", "description": "Rastgele çalışan sayısı üretir."}
    ],
    "Zaman": [
        {"id": "date", "label": "Tarih", "description": "Rastgele tarih üretir."},
        {"id": "time", "label": "Saat", "description": "Rastgele saat üretir."},
        {"id": "datetime", "label": "Tarih-Saat", "description": "Rastgele tarih ve saat üretir."},
        {"id": "timestamp", "label": "Zaman Damgası", "description": "Rastgele zaman damgası (timestamp) üretir."},
        {"id": "century", "label": "Yüzyıl", "description": "Rastgele yüzyıl üretir."},
        {"id": "year", "label": "Yıl", "description": "Rastgele yıl üretir."},
        {"id": "month", "label": "Ay", "description": "Rastgele ay (sayı) üretir."},
        {"id": "month_name", "label": "Ay Adı", "description": "Rastgele ay adı üretir."},
        {"id": "day_of_week", "label": "Hafta Günü", "description": "Rastgele haftanın günü üretir."},
        {"id": "quarter", "label": "Çeyrek", "description": "Rastgele yılın çeyreği üretir."},
        {"id": "is_holiday", "label": "Tatil mi?", "description": "Rastgele günün tatil olup olmadığını belirtir."}
    ],
    "Temel": [
        {"id": "integer", "label": "Tam Sayı", "description": "Rastgele tam sayı üretir."},
        {"id": "float", "label": "Ondalık Sayı", "description": "Rastgele ondalık sayı üretir."},
        {"id": "text", "label": "Metin", "description": "Rastgele kısa metin üretir."},
        {"id": "random_word", "label": "Kelime", "description": "Rastgele kelime üretir."},
        {"id": "random_sentence", "label": "Cümle", "description": "Rastgele kısa cümle üretir."},
        {"id": "boolean", "label": "Evet/Hayır", "description": "Rastgele doğru/yanlış (evet/hayır) değeri üretir."},
        {"id": "color", "label": "Renk", "description": "Rastgele renk adı üretir."},
        {"id": "uuid", "label": "UUID", "description": "Rastgele UUID üretir."},
        {"id": "md5", "label": "MD5", "description": "Rastgele MD5 hash üretir."},
        {"id": "sha1", "label": "SHA1", "description": "Rastgele SHA1 hash üretir."},
        {"id": "sha256", "label": "SHA256", "description": "Rastgele SHA256 hash üretir."},
        {"id": "plaka", "label": "Araç Plakası", "description": "Rastgele Türk plakası üretir."},
        {"id": "barcode", "label": "Barkod", "description": "Rastgele barkod (EAN-13) üretir."},
        {"id": "emoji", "label": "Emoji", "description": "Rastgele emoji üretir."}
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
def generate_personal_data(fake, rows, selected_columns):
    data = []
    fake = Faker('tr_TR')
    # Sabit Türkçe isim ve soyisim listeleri
    turk_first_names_male = [
        "Ahmet", "Mehmet", "Mustafa", "Ali", "Hüseyin", "Burak", "Deniz", "Cem", "Gökhan", "Yusuf",
        "Emre", "Mert", "Can", "Kaan", "Onur", "Barış", "Serkan", "Fatih", "Eren", "Uğur",
        "Oğuz", "Furkan", "Kerem", "Tuna", "Berk", "Tolga", "Suat", "Halil", "Recep", "Süleyman"
    ]
    turk_first_names_female = [
        "Ayşe", "Fatma", "Zeynep", "Elif", "Merve", "Seda", "Buse", "Rabia", "Ece", "Derya",
        "Gül", "Aylin", "Melis", "Sibel", "Esra", "Yasemin", "Ceren", "Nazlı", "Gizem", "Pelin",
        "Şeyma", "Tuğba", "Büşra", "Sevgi", "Nazan", "Nurgül", "Selin", "İrem", "Simge", "Gonca"
    ]
    turk_last_names = [
        "Yılmaz", "Kaya", "Demir", "Çelik", "Şahin", "Yıldız", "Yıldırım", "Aydın", "Öztürk", "Arslan",
        "Doğan", "Kılıç", "Aslan", "Çetin", "Kara", "Koç", "Kurt", "Özdemir", "Polat", "Aksoy",
        "Erdoğan", "Şimşek", "Avcı", "Taş", "Güneş", "Bozkurt", "Bulut", "Güler", "Acar", "Kurtuluş"
    ]
    # Eğitim-Meslek ilişkisi
    education_job_mapping = {
        "İlkokul": ["İşçi", "Teknisyen", "Satış Temsilcisi"],
        "Ortaokul": ["Teknisyen", "Satış Temsilcisi", "Kasiyer"],
        "Lise": ["Teknisyen", "Satış Temsilcisi", "Muhasebeci", "Hemşire"],
        "Üniversite": ["Mühendis", "Doktor", "Öğretmen", "Avukat", "Muhasebeci"],
        "Yüksek Lisans": ["Uzman Mühendis", "Uzman Doktor", "Akademisyen"],
        "Doktora": ["Profesör", "Uzman Doktor", "Araştırmacı"]
    }
    education_income_range = {
        "İlkokul": (5000, 10000),
        "Ortaokul": (7000, 12000),
        "Lise": (10000, 15000),
        "Üniversite": (15000, 25000),
        "Yüksek Lisans": (20000, 35000),
        "Doktora": (30000, 50000)
    }
    age_education_mapping = [
        ((10, 13), ["İlkokul"]),
        ((14, 17), ["Ortaokul", "Lise"]),
        ((18, 21), ["Lise", "Üniversite"]),
        ((22, 25), ["Üniversite"]),
        ((26, 30), ["Üniversite", "Yüksek Lisans"]),
        ((31, 40), ["Üniversite", "Yüksek Lisans"]),
        ((41, 50), ["Üniversite", "Yüksek Lisans", "Doktora"]),
        ((51, 90), ["Üniversite", "Yüksek Lisans", "Doktora"])
    ]
    hobbies = [
        "Kitap Okuma", "Spor", "Müzik", "Sinema", "Seyahat", "Fotoğrafçılık",
        "Yemek Yapma", "Bahçecilik", "Dans", "Yoga", "Meditasyon", "Resim",
        "El İşi", "Koleksiyon", "Balıkçılık", "Kampçılık", "Bisiklet", "Yüzme"
    ]
    interests = [
        "Teknoloji", "Sanat", "Bilim", "Tarih", "Felsefe", "Psikoloji",
        "Ekonomi", "Politika", "Çevre", "Sağlık", "Eğitim", "Spor",
        "Müzik", "Sinema", "Edebiyat", "Fotoğrafçılık", "Yemek", "Seyahat"
    ]
    city_district_mapping = {
        "İstanbul": ["Kadıköy", "Beşiktaş", "Üsküdar", "Şişli", "Bakırköy", "Pendik", "Kartal", "Maltepe", "Ataşehir"],
        "Ankara": ["Çankaya", "Keçiören", "Mamak", "Yenimahalle", "Etimesgut", "Sincan", "Altındağ"],
        "İzmir": ["Konak", "Karşıyaka", "Bornova", "Buca", "Çiğli", "Gaziemir", "Menemen"],
        "Bursa": ["Nilüfer", "Osmangazi", "Yıldırım", "Mudanya", "Gemlik"],
        "Antalya": ["Muratpaşa", "Konyaaltı", "Kepez", "Manavgat", "Alanya"],
        "Adana": ["Sarıçam", "Çukurova", "Seyhan", "Yüreğir"]
    }
    district_neighborhood_mapping = {
        "Kadıköy": ["Caferağa", "Fenerbahçe", "Göztepe", "Suadiye", "Caddebostan", "Moda", "Erenköy"],
        "Beşiktaş": ["Levent", "Etiler", "Bebek", "Ortaköy", "Arnavutköy", "Ulus", "Nişantaşı"],
        "Çankaya": ["Kızılay", "Çayyolu", "Bahçelievler", "Gaziosmanpaşa", "Dikmen"],
        "Konak": ["Alsancak", "Göztepe", "Bostanlı", "Karşıyaka", "Bornova"],
        "Nilüfer": ["Görükle", "Fethiye", "Beşevler", "Çamlıca", "İhsaniye"]
    }
    for _ in range(rows):
        age = random.randint(10, 90)
        education = "İlkokul"
        for age_range, educations in age_education_mapping:
            if age_range[0] <= age <= age_range[1]:
                education = random.choice(educations)
                break
        possible_jobs = education_job_mapping.get(education, ["Öğrenci"])
        job = random.choice(possible_jobs)
        min_income, max_income = education_income_range.get(education, (0, 0))
        income = round(random.randint(min_income, max_income), -3) if max_income > 0 else 0
        gender = random.choice(["Erkek", "Kadın"])
        # Sabit Türkçe isim ve soyisimlerden seç
        if gender == "Erkek":
            first_name = random.choice(turk_first_names_male)
        else:
            first_name = random.choice(turk_first_names_female)
        last_name = random.choice(turk_last_names)
        full_name = f"{first_name} {last_name}"
        username = f"{first_name.lower()}{last_name.lower()}"
        email = f"{first_name.lower()}.{last_name.lower()}@gmail.com"
        twitter = f"@{first_name.lower()}{last_name.lower()}"
        linkedin = f"linkedin.com/in/{first_name.lower()}-{last_name.lower()}"
        # Adres üretimi: şehir, ilçe, mahalle aynı satırda
        city = random.choice(list(city_district_mapping.keys()))
        district = random.choice(city_district_mapping[city])
        neighborhood = random.choice(district_neighborhood_mapping.get(district, ["Merkez"]))
        street = fake.street_name()
        building_no = random.randint(1, 100)
        flat_no = random.randint(1, 20)
        address = f"{street} No:{building_no} D:{flat_no} {neighborhood} {district}/{city}"
        country = "Türkiye"
        birth_date = datetime.now() - timedelta(days=age*365 + random.randint(0, 364))
        marital_status_prob = {
            (10, 17): {"Bekar": 1.0},
            (18, 25): {"Bekar": 0.8, "Evli": 0.2},
            (26, 35): {"Bekar": 0.4, "Evli": 0.5, "Boşanmış": 0.1},
            (36, 45): {"Bekar": 0.2, "Evli": 0.6, "Boşanmış": 0.2},
            (46, 90): {"Bekar": 0.1, "Evli": 0.7, "Boşanmış": 0.2}
        }
        marital_status = "Bekar"
        for age_range, probs in marital_status_prob.items():
            if age_range[0] <= age <= age_range[1]:
                marital_status = random.choices(
                    list(probs.keys()),
                    weights=list(probs.values())
                )[0]
                break
        def generate_tc_no():
            first_nine = ''.join([str(random.randint(0, 9)) for _ in range(9)])
            tenth = sum(int(digit) for digit in first_nine) % 10
            eleventh = (sum(int(digit) for digit in first_nine) + tenth) % 10
            return f"{first_nine}{tenth}{eleventh}"
        def generate_passport_no():
            letters = ''.join(random.choices(string.ascii_uppercase, k=2))
            numbers = ''.join(random.choices(string.digits, k=7))
            return f"{letters}{numbers}"
        def generate_driver_license():
            return ''.join(random.choices(string.ascii_uppercase + string.digits, k=11))
        def generate_credit_score():
            return random.randint(300, 850)
        def generate_tr_phone():
            return f"+90 5{random.randint(0,9)}{random.randint(0,9)} {random.randint(100,999)} {random.randint(10,99)} {random.randint(10,99)}"
        row = {}
        if "name" in selected_columns:
            row["İsim-Soyisim"] = full_name
        if "first_name" in selected_columns:
            row["İsim"] = first_name
        if "last_name" in selected_columns:
            row["Soyisim"] = last_name
        if "username" in selected_columns:
            row["Kullanıcı Adı"] = username
        if "email" in selected_columns:
            row["E-posta"] = email
        if "phone" in selected_columns:
            row["Telefon"] = generate_tr_phone()
        if "job" in selected_columns:
            row["Meslek"] = job
        if "birth_date" in selected_columns:
            row["Doğum Tarihi"] = birth_date.strftime('%Y-%m-%d')
        if "age" in selected_columns:
            row["Yaş"] = age
        if "gender" in selected_columns:
            row["Cinsiyet"] = gender
        if "income" in selected_columns:
            row["Gelir Seviyesi"] = income
        if "education" in selected_columns:
            row["Eğitim Seviyesi"] = education
        if "marital_status" in selected_columns:
            row["Medeni Durum"] = marital_status
        if "ssn" in selected_columns:
            row["TC Kimlik No"] = generate_tc_no()
        if "credit_score" in selected_columns:
            row["Kredi Notu"] = generate_credit_score()
        if "passport_number" in selected_columns:
            row["Pasaport Numarası"] = generate_passport_no()
        if "driver_license" in selected_columns:
            row["Ehliyet Numarası"] = generate_driver_license()
        if "address" in selected_columns:
            row["Ev Adresi"] = address
        if "city" in selected_columns:
            row["Şehir"] = city
        if "country" in selected_columns:
            row["Ülke"] = country
        if "twitter" in selected_columns:
            row["Twitter Kullanıcı Adı"] = twitter
        if "linkedin" in selected_columns:
            row["LinkedIn Profili"] = linkedin
        if "hobbies" in selected_columns:
            row["Hobiler"] = ', '.join(random.sample(hobbies, random.randint(1, 3)))
        if "interests" in selected_columns:
            row["İlgi Alanları"] = ', '.join(random.sample(interests, random.randint(1, 3)))
        if "neighborhood" in selected_columns:
            row["Mahalle"] = neighborhood
        data.append(row)
    return pd.DataFrame(data)

#Adres veri üretimi için özel fonksiyon
def generate_address_data(fake, data_type, rows, city=None):
    # Şehir-İlçe ilişkisi genişletilmiş
    city_district_mapping = {
        "İstanbul": ["Kadıköy", "Beşiktaş", "Üsküdar", "Şişli", "Bakırköy", "Pendik", "Kartal", "Maltepe", "Ataşehir"],
        "Ankara": ["Çankaya", "Keçiören", "Mamak", "Yenimahalle", "Etimesgut", "Sincan", "Altındağ"],
        "İzmir": ["Konak", "Karşıyaka", "Bornova", "Buca", "Çiğli", "Gaziemir", "Menemen"],
        "Bursa": ["Nilüfer", "Osmangazi", "Yıldırım", "Mudanya", "Gemlik"],
        "Antalya": ["Muratpaşa", "Konyaaltı", "Kepez", "Manavgat", "Alanya"],
        "Adana": ["Sarıçam", "Çukurova", "Seyhan", "Yüreğir"]
    }
    
    # İlçe-Mahalle ilişkisi genişletilmiş
    district_neighborhood_mapping = {
        "Kadıköy": ["Caferağa", "Fenerbahçe", "Göztepe", "Suadiye", "Caddebostan", "Moda", "Erenköy"],
        "Beşiktaş": ["Levent", "Etiler", "Bebek", "Ortaköy", "Arnavutköy", "Ulus", "Nişantaşı"],
        "Çankaya": ["Kızılay", "Çayyolu", "Bahçelievler", "Gaziosmanpaşa", "Dikmen"],
        "Konak": ["Alsancak", "Göztepe", "Bostanlı", "Karşıyaka", "Bornova"],
        "Nilüfer": ["Görükle", "Fethiye", "Beşevler", "Çamlıca", "İhsaniye"]
    }
    
    # Posta kodu formatı
    def generate_postcode(city):
        city_codes = {
            "İstanbul": "34",
            "Ankara": "06",
            "İzmir": "35",
            "Bursa": "16",
            "Antalya": "07",
            "Adana": "01"
        }
        return f"{city_codes.get(city, '00')}{random.randint(100, 999)}"
    
    # GPS koordinatları (şehirlere göre yaklaşık merkezler)
    city_coordinates = {
        "İstanbul": {"lat": (40.9, 41.1), "lon": (28.7, 29.1)},
        "Ankara": {"lat": (39.8, 40.0), "lon": (32.6, 32.8)},
        "İzmir": {"lat": (38.3, 38.5), "lon": (27.0, 27.2)},
        "Bursa": {"lat": (40.1, 40.3), "lon": (29.0, 29.2)},
        "Antalya": {"lat": (36.8, 37.0), "lon": (30.6, 30.8)},
        "Adana": {"lat": (36.9, 37.1), "lon": (35.2, 35.4)}
    }
    
    data = []
    for _ in range(rows):
        # Şehir ve ilçe seçimi
        city = random.choice(list(city_district_mapping.keys()))
        district = random.choice(city_district_mapping[city])
        neighborhood = random.choice(district_neighborhood_mapping.get(district, ["Merkez"]))

        if data_type == 'address':
            street = random.choice(turkish_streets)
            building_no = random.randint(1, 100)
            flat_no = random.randint(1, 20)
            full_address = f"{street} No:{building_no} D:{flat_no} {neighborhood} {district}/{city}"
            data.append(full_address)
        elif data_type == 'street':
            data.append(random.choice(turkish_streets))
        elif data_type == 'building_name':
            data.append(random.choice(turkish_buildings))
        elif data_type == 'door_number':
            data.append(f"{random.randint(1, 100)}/{random.randint(1, 20)}")
        elif data_type == 'district':
            data.append(district)
        elif data_type == 'city':
            data.append(city)
        elif data_type == 'state':
            data.append(city)
        elif data_type == 'country':
            data.append("Türkiye")
        elif data_type == 'postcode':
            # Şehir ve ilçe için özel posta kodu
            postcode = city_district_postcodes.get(city, {}).get(district)
            if not postcode:
                postcode = f"{random.randint(10000, 99999)}"
            data.append(postcode)
        elif data_type == 'latitude':
            lat_range = city_coordinates[city]["lat"]
            data.append(f"{random.uniform(lat_range[0], lat_range[1]):.6f}")
        elif data_type == 'longitude':
            lon_range = city_coordinates[city]["lon"]
            data.append(f"{random.uniform(lon_range[0], lon_range[1]):.6f}")
        elif data_type == 'gps':
            lat_range = city_coordinates[city]["lat"]
            lon_range = city_coordinates[city]["lon"]
            lat = random.uniform(lat_range[0], lat_range[1])
            lon = random.uniform(lon_range[0], lon_range[1])
            data.append(f"{lat:.6f},{lon:.6f}")
        elif data_type == 'timezone':
            data.append("Europe/Istanbul")
        elif data_type == 'neighborhood':
            data.append(neighborhood)
        else:
            raise ValueError(f"Desteklenmeyen veri türü: {data_type}")
    return data
    
#Finans veri üretimi için özel fonksiyon
def generate_financial_data(fake, data_type, rows, bank=None, card_type=None, income=None):
    # Banka-Kart ilişkisi genişletilmiş
    bank_card_mapping = {
        "Ziraat Bankası": ["Visa", "MasterCard"],
        "Garanti BBVA": ["Visa", "MasterCard", "American Express"],
        "İş Bankası": ["Visa", "MasterCard"],
        "Akbank": ["Visa", "MasterCard", "American Express"],
        "Yapı Kredi": ["Visa", "MasterCard"],
        "Halkbank": ["Visa", "MasterCard"],
        "Vakıfbank": ["Visa", "MasterCard"],
        "Denizbank": ["Visa", "MasterCard", "American Express"]
    }
    
    # Kart türü ve limit ilişkisi
    card_type_limit_multiplier = {
        "Visa": (2.0, 3.0),
        "MasterCard": (2.5, 3.5),
        "American Express": (3.0, 4.0)
    }
    
    # Banka-IBAN ilişkisi
    bank_iban_prefix = {
        "Ziraat Bankası": "TR33",
        "Garanti BBVA": "TR62",
        "İş Bankası": "TR64",
        "Akbank": "TR46",
        "Yapı Kredi": "TR67",
        "Halkbank": "TR12",
        "Vakıfbank": "TR15",
        "Denizbank": "TR69"
    }
    
    # Para birimleri ve kurları
    currencies = {
        "TRY": 1.0,
        "USD": 31.5,
        "EUR": 34.2,
        "GBP": 39.8
    }
    
    # Kripto para birimleri ve fiyatları
    cryptocurrencies = {
        "BTC": 65000,
        "ETH": 3500,
        "BNB": 400,
        "XRP": 0.5
    }
    
    data = []
    for _ in range(rows):
        if data_type == 'credit_card':
            bank = random.choice(list(bank_card_mapping.keys()))
            card_type = random.choice([ct for ct in bank_card_mapping[bank] if ct.lower() in ['visa', 'mastercard']])
            data.append(fake.credit_card_number(card_type=card_type.lower()))
        elif data_type == 'card_provider':
            data.append(random.choice(list(bank_card_mapping.keys())))
        elif data_type == 'card_type':
            bank = random.choice(list(bank_card_mapping.keys()))
            data.append(random.choice(bank_card_mapping[bank]))
        elif data_type == 'card_expiry':
            data.append(fake.credit_card_expire())
        elif data_type == 'card_cvv':
            data.append(fake.credit_card_security_code())
        elif data_type == 'iban':
            bank = random.choice(list(bank_iban_prefix.keys()))
            prefix = bank_iban_prefix[bank]
            rest = ''.join([str(random.randint(0, 9)) for _ in range(24)])
            data.append(f"{prefix}{rest}")
        elif data_type == 'bic':
            # Rastgele 8-11 karakterli BIC kodu üret
            bic = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            data.append(bic)
        elif data_type == 'ein':
            # ABD EIN formatı: 2 rakam - 7 rakam
            ein = f"{random.randint(10,99)}-{random.randint(1000000,9999999)}"
            data.append(ein)
        elif data_type == 'duns':
            duns = ''.join(random.choices(string.digits, k=9))
            data.append(duns)
        elif data_type == 'bank_name':
            data.append(random.choice(list(bank_card_mapping.keys())))
        elif data_type == 'account_number':
            data.append(str(fake.random_number(digits=12)))
        elif data_type == 'balance':
            currency = random.choice(list(currencies.keys()))
            amount = round(random.uniform(1000, 100000), 2)
            data.append(f"{amount} {currency}")
        elif data_type == 'credit_limit':
            income = random.randint(5000, 50000)
            card_type = random.choice(["Visa", "MasterCard", "American Express"])
            min_mult, max_mult = card_type_limit_multiplier[card_type]
            limit = int(income * random.uniform(min_mult, max_mult))
            data.append(limit)
        elif data_type == 'currency':
            data.append(random.choice(list(currencies.keys())))
        elif data_type == 'cryptocurrency':
            data.append(random.choice(list(cryptocurrencies.keys())))
        elif data_type == 'crypto_wallet':
            data.append(fake.sha256())
        elif data_type == 'amount':
            currency = random.choice(list(currencies.keys()))
            amount = round(random.uniform(100, 10000), 2)
            data.append(f"{amount} {currency}")
        elif data_type == 'transaction_type':
            data.append(random.choice(['debit', 'credit', 'transfer', 'payment']))
        elif data_type == 'vat':
            data.append(f"TR{random.randint(1000000000, 9999999999)}")
        elif data_type == 'duns':
            duns = ''.join(random.choices(string.digits, k=9))
            data.append(duns)
        else:
            raise ValueError(f"Desteklenmeyen veri türü: {data_type}")
    return data
    
# İnternet veri üretimi için özel fonksiyon
def generate_internet_data(fake, data_type, rows, name=None, email=None):
    """
    İnternet verisi üretimi için özel fonksiyon.
    
    Args:
        fake: Faker nesnesifi
        data_type: Üretilecek veri türü
        rows: Üretilecek satır sayısı
        name: Kişi adı (opsiyonel)
        email: E-posta adresi (opsiyonel)
    """
    # İşletim Sistemi-Browser ilişkisi
    os_browser_mapping = {
        "Windows 10": ["Chrome", "Firefox", "Edge", "Opera"],
        "Windows 11": ["Chrome", "Firefox", "Edge", "Opera"],
        "macOS": ["Safari", "Chrome", "Firefox"],
        "Linux": ["Firefox", "Chrome", "Opera"],
        "Android": ["Chrome", "Samsung Internet", "Firefox"],
        "iOS": ["Safari", "Chrome", "Firefox"]
    }
    
    # Browser-User Agent ilişkisi
    browser_versions = {
        "Chrome": ["120.0.0.0", "119.0.0.0", "118.0.0.0"],
        "Firefox": ["120.0", "119.0", "118.0"],
        "Safari": ["17.0", "16.0", "15.0"],
        "Edge": ["120.0.0.0", "119.0.0.0", "118.0.0.0"]
    }
    
    # E-posta sağlayıcıları ve domain ilişkisi
    email_domain_mapping = {
        "gmail.com": ["google.com", "googlemail.com"],
        "hotmail.com": ["outlook.com", "live.com", "msn.com"],
        "yahoo.com": ["yahoo.co.uk", "yahoo.fr", "yahoo.de"],
        "outlook.com": ["microsoft.com", "hotmail.com"]
    }
    
    # Sosyal medya platformları ve kullanıcı adı formatları
    social_media_formats = {
        "Twitter": lambda name: f"@{name.lower().replace(' ', '')}",
        "Instagram": lambda name: f"@{name.lower().replace(' ', '_')}",
        "LinkedIn": lambda name: f"linkedin.com/in/{name.lower().replace(' ', '-')}" ,
        "Facebook": lambda name: f"facebook.com/{name.lower().replace(' ', '.')}"
    }
    
    data = []
    for _ in range(rows):
        if data_type == 'os':
            os = random.choice(list(os_browser_mapping.keys()))
            data.append(os)
        elif data_type == 'browser':
            os = random.choice(list(os_browser_mapping.keys()))
            browser = random.choice(os_browser_mapping[os])
            data.append(browser)
        elif data_type == 'user_agent':
            os = random.choice(list(os_browser_mapping.keys()))
            browser = random.choice(os_browser_mapping[os])
            version = random.choice(browser_versions.get(browser, ["1.0"]))
            user_agent = f"Mozilla/5.0 ({os}) {browser}/{version}"
            data.append(user_agent)
        elif data_type == 'email_provider':
            provider = random.choice(list(email_domain_mapping.keys()))
            data.append(provider)
        elif data_type == 'domain':
            provider = random.choice(list(email_domain_mapping.keys()))
            domain = random.choice(email_domain_mapping[provider])
            data.append(domain)
        elif data_type == 'sosyal_medya':
            platform = random.choice(list(social_media_formats.keys()))
            data.append(platform)
        elif data_type == 'sosyal_medya_kullanıcı':
            platform = random.choice(list(social_media_formats.keys()))
            if name:
                username = social_media_formats[platform](name)
            else:
                username = social_media_formats[platform](fake.name())
            data.append(username)
        elif data_type == 'ipv4':
            data.append(fake.ipv4())
        elif data_type == 'ipv6':
            data.append(fake.ipv6())
        elif data_type == 'mac_address':
            data.append(fake.mac_address())
        elif data_type == 'port':
            data.append(str(random.randint(1, 65535)))
        elif data_type == 'ssl_serial':
            data.append(fake.sha1())
        elif data_type == 'uri':
            data.append(fake.uri())
        elif data_type == 'url':
            data.append(fake.url())
        else:
            raise ValueError(f"Desteklenmeyen veri türü: {data_type}")
    return data

# Şirket veri üretimi için özel fonksiyon
def generate_company_data(fake, data_type, rows):
    # Sektör-Şirket türü ilişkisi
    sector_company_type = {
        "Teknoloji": ["Ltd. Şti.", "A.Ş.", "Teknoloji A.Ş."],
        "Finans": ["A.Ş.", "Bankası", "Sigorta A.Ş."],
        "Sağlık": ["Hastanesi", "Tıp Merkezi", "Sağlık Hizmetleri A.Ş."],
        "Eğitim": ["Okulu", "Eğitim Kurumu", "Akademi"],
        "Üretim": ["Fabrikası", "Üretim A.Ş.", "Sanayi Ltd. Şti."],
        "Perakende": ["Mağazaları", "Market", "Alışveriş Merkezi"]
    }
    
    # Sektör-Çalışan sayısı ilişkisi
    sector_employee_range = {
        "Teknoloji": (10, 1000),
        "Finans": (50, 5000),
        "Sağlık": (20, 2000),
        "Eğitim": (5, 500),
        "Üretim": (100, 10000),
        "Perakende": (5, 1000)
    }
    
    # Sektör-Kuruluş yılı ilişkisi
    sector_founding_year_range = {
        "Teknoloji": (2000, 2023),
        "Finans": (1950, 2023),
        "Sağlık": (1980, 2023),
        "Eğitim": (1990, 2023),
        "Üretim": (1970, 2023),
        "Perakende": (1990, 2023)
    }
    
    data = []
    for _ in range(rows):
        # Önce sektör belirle
        sector = random.choice(list(sector_company_type.keys()))
        if data_type == 'company':
            company_type = random.choice(sector_company_type[sector])
            company_name = fake.company()
            data.append(f"{company_name} {company_type}")
        elif data_type == 'company_suffix':
            data.append(random.choice(sector_company_type[sector]))
        elif data_type == 'ein':
            # ABD EIN formatı: 2 rakam - 7 rakam
            ein = f"{random.randint(10,99)}-{random.randint(1000000,9999999)}"
            data.append(ein)
        elif data_type == 'duns':
            duns = ''.join(random.choices(string.digits, k=9))
            data.append(duns)
        elif data_type == 'vat':
            data.append(f"TR{random.randint(1000000000, 9999999999)}")
        elif data_type == 'company_address':
            data.append(fake.address().replace("\n", " "))
        elif data_type == 'company_phone':
            data.append(fake.phone_number())
        elif data_type == 'company_email':
            company_name = fake.company().lower().replace(" ", "")
            data.append(f"info@{company_name}.com")
        elif data_type == 'year_founded':
            min_year, max_year = sector_founding_year_range[sector]
            data.append(random.randint(min_year, max_year))
        elif data_type == 'sector':
            data.append(sector)
        elif data_type == 'employee_count':
            min_emp, max_emp = sector_employee_range[sector]
            data.append(random.randint(min_emp, max_emp))
        else:
            raise ValueError(f"Desteklenmeyen veri türü: {data_type}")
    
    return data

# Zaman veri üretimi için özel fonksiyon
def generate_time_data(fake, data_type, rows):
    # Tatil günleri (Türkiye için örnek)
    holidays = [
        "2024-01-01",  # Yılbaşı
        "2024-04-23",  # Ulusal Egemenlik ve Çocuk Bayramı
        "2024-05-01",  # İşçi Bayramı
        "2024-05-19",  # Gençlik ve Spor Bayramı
        "2024-07-15",  # Demokrasi Bayramı
        "2024-08-30",  # Zafer Bayramı
        "2024-10-29",  # Cumhuriyet Bayramı
    ]
    
    # Mevsim-Ay ilişkisi
    season_month_mapping = {
        "İlkbahar": [3, 4, 5],
        "Yaz": [6, 7, 8],
        "Sonbahar": [9, 10, 11],
        "Kış": [12, 1, 2]
    }
    
    # Ay-Gün sayısı ilişkisi
    month_days = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }
    
    data = []
    for _ in range(rows):
        if data_type == 'date':
            date = fake.date_between(start_date='-1y', end_date='+1y')
            data.append(date.strftime('%Y-%m-%d'))
        elif data_type == 'time':
            data.append(fake.time())
        elif data_type == 'datetime':
            datetime_val = fake.date_time_between(start_date='-1y', end_date='+1y')
            data.append(datetime_val.strftime('%Y-%m-%d %H:%M:%S'))
        elif data_type == 'timestamp':
            data.append(int(fake.date_time_between(start_date='-1y', end_date='+1y').timestamp()))
        elif data_type == 'century':
            data.append(fake.century())
        elif data_type == 'year':
            data.append(random.randint(1900, 2024))
        elif data_type == 'month':
            data.append(random.randint(1, 12))
        elif data_type == 'month_name':
            month = random.randint(1, 12)
            data.append(datetime(2024, month, 1).strftime('%B'))
        elif data_type == 'day_of_week':
            data.append(fake.day_of_week())
        elif data_type == 'quarter':
            data.append(random.randint(1, 4))
        elif data_type == 'is_holiday':
            date = fake.date_between(start_date='-1y', end_date='+1y')
            data.append("Evet" if date.strftime('%Y-%m-%d') in holidays else "Hayır")
        else:
            raise ValueError(f"Desteklenmeyen veri türü: {data_type}")
    return data
    
# Temel veri üretimi için özel fonksiyon
def generate_basic_data(fake, data_type, rows):
    emojis = ["😀", "😂", "😍", "😎", "😢", "😡", "👍", "🙏", "🎉", "🚀"]
    if data_type == 'integer':
        return [fake.random_int(min=0, max=100) for _ in range(rows)]
    elif data_type == 'float':
        return [fake.pyfloat(left_digits=3, right_digits=2, positive=True) for _ in range(rows)]
    elif data_type == 'text':
        return [fake.text(max_nb_chars=20) for _ in range(rows)]
    elif data_type == 'random_word':
        return [fake.word() for _ in range(rows)]
    elif data_type == 'random_sentence':
        return [fake.sentence(nb_words=6) for _ in range(rows)]
    elif data_type == 'boolean':
        return [fake.boolean() for _ in range(rows)]
    elif data_type == 'color':
        return [fake.color_name() for _ in range(rows)]
    elif data_type == 'uuid':
        return [str(fake.uuid4()) for _ in range(rows)]
    elif data_type == 'md5':
        return [fake.md5() for _ in range(rows)]
    elif data_type == 'sha1':
        return [fake.sha1() for _ in range(rows)]
    elif data_type == 'sha256':
        return [fake.sha256() for _ in range(rows)]
    elif data_type == 'plaka':
        return [f"{random.randint(1,81):02d} {fake.random_uppercase_letter()}{fake.random_uppercase_letter()} {random.randint(100,9999)}" for _ in range(rows)]
    elif data_type == 'barcode':
        return [fake.ean(length=13) for _ in range(rows)]
    elif data_type == 'emoji':
        return [random.choice(emojis) for _ in range(rows)]
    else:
        raise ValueError(f"Desteklenmeyen veri türü: {data_type}")
    
#yardımcı fonksiyon
def get_label(category, id_):
    for t in DATA_CATEGORIES[category]:
        if t["id"] == id_:
            return t["label"]
    return id_  # fallback

# id'den label'a eşleştirme fonksiyonu
def get_label_by_id(data_type):
    for category in DATA_CATEGORIES.values():
        for t in category:
            if t["id"] == data_type:
                return t["label"]
    return data_type

# Veri üretimi için ana fonksiyon
def generate_fake_data(rows, columns, data_types):
    fake = Faker(['tr_TR'])
    data = {data_type: [] for data_type in data_types}
    valid_data_types = []
    for category in DATA_CATEGORIES.values():
        valid_data_types.extend([t["id"] for t in category])
    invalid_types = [dt for dt in data_types if dt not in valid_data_types]
    if invalid_types:
        raise ValueError(f"Geçersiz veri türleri: {invalid_types}")

    unsupported_types = []
    for dt in data_types:
        if not check_data_type_supported(fake, dt):
            unsupported_types.append(dt)
    if unsupported_types:
        raise ValueError(
            "Aşağıdaki veri türleri üretilemiyor: " +
            ", ".join([get_label_by_id(dt) for dt in unsupported_types])
        )

    for _ in range(rows):
        # Ortak değişkenler
        common_data = {
            'city': None, 'bank': None, 'card_type': None, 'income': None,
            'name': None, 'email': None, 'phone': None, 'company': None, 'sector': None
        }
        # Kişisel verileri önceden üret
        personal_types = [dt for dt in data_types if dt in [t["id"] for t in DATA_CATEGORIES["Kişisel"]]]
        personal_data = {}
        if personal_types:
            pd_df = generate_personal_data(fake, 1, personal_types)
            id_to_label = {t["id"]: t["label"] for t in DATA_CATEGORIES["Kişisel"]}
            for id_ in personal_types:
                label = id_to_label[id_]
                personal_data[id_] = pd_df[label].iloc[0]
                # Ortak değişkenleri güncelle
                if id_ == 'income':
                    common_data['income'] = pd_df[label].iloc[0]
                elif id_ == 'name':
                    common_data['name'] = pd_df[label].iloc[0]
                elif id_ == 'email':
                    common_data['email'] = pd_df[label].iloc[0]
                elif id_ == 'phone':
                    common_data['phone'] = pd_df[label].iloc[0]
        # Her veri türü için değer üret
        for data_type in data_types:
            value = ""
            try:
                print(f"Üretiliyor: {data_type}")
                category = get_data_type_category(data_type)
                if category == "Kişisel":
                    value = personal_data.get(data_type, "")
                elif category == "Adres":
                    value = generate_address_data(fake, data_type, 1, common_data['city'])[0]
                elif category == "Finans":
                    value = generate_financial_data(fake, data_type, 1, bank=common_data['bank'], card_type=common_data['card_type'], income=common_data['income'])[0]
                elif category == "İnternet":
                    value = generate_internet_data(fake, data_type, 1, name=common_data['name'], email=common_data['email'])[0]
                elif category == "Şirket":
                    value = generate_company_data(fake, data_type, 1)[0]
                elif category == "Zaman":
                    value = generate_time_data(fake, data_type, 1)[0]
                elif category == "Temel":
                    value = generate_basic_data(fake, data_type, 1)[0]
                else:
                    value = ""
            except Exception as e:
                app.logger.error(f"Veri üretimi hatası - Tür: {data_type}, Hata: {str(e)}")
            data[data_type].append(value)
    # Her sütunun uzunluğunu rows'a tamamla
    for k, v in data.items():
        if len(v) < rows:
            v.extend([""] * (rows - len(v)))
    for k, v in data.items():
        print(f"{k}: {len(v)}")
    print("rows:", rows)
    # Sütun adlarını Türkçeleştir
    turkish_columns = {col: get_label_by_id(col) for col in data.keys()}
    df = pd.DataFrame(data)
    df.rename(columns=turkish_columns, inplace=True)
    return df

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
            real_data = turkce_sutun_adlari(real_data)
            meta_data=SingleTableMetadata()
            meta_data.detect_from_dataframe(data=real_data)
            meta_data = remove_pii_from_metadata(real_data)
            model = CTGANSynthesizer(
                metadata=meta_data,
                epochs=300,
                batch_size=500,
                generator_dim=(256, 256),
                discriminator_dim=(256, 256),
                generator_lr=0.0002,
                discriminator_lr=0.0002,
                verbose=True,
                enforce_min_max_values=False
            )
            model.fit(real_data)

            # Yeni veri üret
            synthetic_data = model.sample(num_rows)
            synthetic_data.replace([np.nan, np.inf, -np.inf], None, inplace=True)

            # Dosya kaydı
            print(synthetic_data.isnull().sum())
            print(synthetic_data.dtypes)

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
        # Username üret
        username = user_info.get('name')
        if not username:
            username = user_info['email'].split('@')[0]
        # Geçici şifre oluştur (kullanıcı tarafından asla kullanılmayacak)
        temp_password = generate_password_hash("google_login_placeholder")
        user = User(
            email=user_info['email'],
            username=username,
            password_hash=temp_password,
            is_admin=False
        )
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
            is_download = request.form.get('download') == 'true'

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

            if is_download:
                #dosya indirme işlemi için
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
            else:
                # Önizleme için JSON response
                return jsonify({
                    'success': True,
                    'preview': data.head().to_dict('records')
                })

        except ValueError as e:
            flash(f"Veri üretimi sırasında bir hata oluştu: {str(e)}", "danger")
            return redirect(url_for('generate_data'))
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })

    # GET isteği için template'i render et
    return render_template('generate.html', 
                         categories=DATA_CATEGORIES.keys(),
                         DATA_CATEGORIES=DATA_CATEGORIES)

@app.route('/get_data_types/<category>')
@login_required
def get_data_types(category):
    """Belirli bir kategoriye ait veri türlerini döndürür"""
    try:
        if category in DATA_CATEGORIES:
            return jsonify({
                'success': True,
                'data': DATA_CATEGORIES[category]
            })
        return jsonify({
            'success': False,
            'error': 'Kategori bulunamadı'
        })
    except Exception as e:
        app.logger.error(f"Veri türleri hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
        # Veri türlerinin geçerliliğini kontrol et
     
#kullanıcı aktiviteleri modeli
class UserActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    action = db.Column(db.String(100), nullable=False)  #kullanıcının yaptığı işlem (örneğin: 'veri üretimi', 'giriş', 'çıkış')
    date_created = db.Column(db.DateTime, default=datetime.utcnow) # Aktivite tarihi
    data_type = db.Column(db.String(50))  # örneğin: 'random', 'model'
    row_count = db.Column(db.Integer)
    column_count = db.Column(db.Integer)
    file_format = db.Column(db.String(10))  # örneğin: 'csv', 'xlsx'
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

# Demografik veri üretimi için özel fonksiyon
def generate_demographic_data(fake, rows):
    data = []
    marital_statuses = ["Bekar", "Evli", "Boşanmış", "Dul"]
    education_levels = ["İlkokul", "Ortaokul", "Lise", "Üniversite", "Yüksek Lisans", "Doktora"]

    for _ in range(rows):
        marital_status = random.choice(marital_statuses)
        education = random.choice(education_levels)
        data.append({
            "Medeni Durum": marital_status,
            "Eğitim Seviyesi": education
        })
    return pd.DataFrame(data)

def check_data_type_supported(fake, data_type):
    try:
        if data_type in [t["id"] for t in DATA_CATEGORIES["Kişisel"]]:
            generate_personal_data(fake, 1, [data_type])
        elif data_type in [t["id"] for t in DATA_CATEGORIES["Adres"]]:
            generate_address_data(fake, data_type, 1)
        elif data_type in [t["id"] for t in DATA_CATEGORIES["Finans"]]:
            generate_financial_data(fake, data_type, 1)
        elif data_type in [t["id"] for t in DATA_CATEGORIES["İnternet"]]:
            generate_internet_data(fake, data_type, 1)
        elif data_type in [t["id"] for t in DATA_CATEGORIES["Şirket"]]:
            generate_company_data(fake, data_type, 1)
        elif data_type in [t["id"] for t in DATA_CATEGORIES["Zaman"]]:
            generate_time_data(fake, data_type, 1)
        elif data_type in [t["id"] for t in DATA_CATEGORIES["Temel"]]:
            generate_basic_data(fake, data_type, 1)
        else:
            return False
        return True
    except Exception:
        return False

def get_data_type_category(data_type):
    for category, types in DATA_CATEGORIES.items():
        for t in types:
            if t["id"] == data_type:
                return category
    return None

SHARED_FOLDER = os.path.join('generated_files', 'shared')
os.makedirs(SHARED_FOLDER, exist_ok=True)

# Paylaşılan dosyaları temizleyen fonksiyon (1 saatten eski dosyaları siler)
def cleanup_shared_files():
    while True:
        now = time.time()
        for fname in os.listdir(SHARED_FOLDER):
            fpath = os.path.join(SHARED_FOLDER, fname)
            if os.path.isfile(fpath):
                created = os.path.getctime(fpath)
                if now - created > 3600:  # 1 saat
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
        time.sleep(600)  # 10 dakikada bir kontrol et

# Temizleyici thread'i başlat
Thread(target=cleanup_shared_files, daemon=True).start()

@app.route('/share_data', methods=['POST'])
@login_required
def share_data():
    try:
        rows = int(request.form.get('rows', 0))
        file_format = request.form.get('format', 'csv')

        data_types = []
        i = 0
        while True:
            data_type_key = f'data_type_{i}'
            if data_type_key not in request.form:
                break
            data_types.append(request.form[data_type_key])
            i += 1
        columns = len(data_types)
        
        if rows <= 0 or columns <= 0:
            return jsonify({'success': False, 'error': 'Geçersiz satır veya sütun sayısı!'}), 400

        data = generate_fake_data(rows, columns, data_types)
        uid = str(uuid.uuid4())
        ext = 'csv' if file_format == 'csv' else 'xlsx'
        filename = f"shared_{uid}.{ext}"
        file_path = os.path.join(SHARED_FOLDER, filename)
        if file_format == 'csv':
            data.to_csv(file_path, index=False)
        else:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                data.to_excel(writer, index=False)
        share_url = url_for('download_shared', uid=uid, _external=True)
        return jsonify({'success': True, 'url': share_url})
    except Exception as e:
        app.logger.error(f"Paylaşılabilir dosya hatası: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/share/<uid>')
def download_shared(uid):
    # Dosya adını bul
    for ext in ['csv', 'xlsx']:
        filename = f"shared_{uid}.{ext}"
        file_path = os.path.join(SHARED_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
    return "Dosya bulunamadı veya süresi doldu.", 404

# Model tabanlı veri üretimi route'ları
@app.route('/model_generate')
@login_required
def model_generate():
    return render_template('generate_model.html')

def detect_pii_columns(df):
    # Basit PII tespiti: isim, soyisim, email, telefon, adres, ssn, id, title, imdb gibi anahtar kelimeler
    pii_keywords = ['isim', 'name', 'soyisim', 'surname', 'email', 'mail', 'telefon', 'phone', 'adres', 'address', 'ssn', 'id', 'title', 'imdb']
    pii_cols = [col for col in df.columns if any(kw in col.lower() for kw in pii_keywords)]
    return pii_cols

# model_preview fonksiyonunda PII kolonlarını tespit et ve frontend'e gönder
@app.route('/model_preview', methods=['POST'])
@login_required
def model_preview():
    try:
        file = request.files['file']
        model_type = request.form.get('model_type', 'ctgan')
        epochs = int(request.form.get('epochs', 100))
        batch_size = int(request.form.get('batch_size', 500))
        learning_rate = float(request.form.get('learning_rate', 0.0002))
        num_rows = int(request.form.get('num_rows', 1000))
        generator_dims_str = request.form.get('generator_dims', '256,256')
        generator_dims_str = generator_dims_str.replace('[', '').replace(']', '').replace(' ', '')
        generator_dims = [int(x) for x in generator_dims_str.split(',')]
        exclude_pii_cols = request.form.getlist('exclude_pii_cols')
        exclude_pii = request.form.get('exclude_pii', 'false') == 'true'
        if file and allowed_file(file.filename):
            df = pd.read_csv(file)
            pii_cols = detect_pii_columns(df)
            # Eğer exclude_pii_cols gelmişse sadece işaretli olmayanları bırak
            if exclude_pii_cols:
                df = df.drop(columns=exclude_pii_cols)
            elif exclude_pii and pii_cols:
                df = df.drop(columns=pii_cols)
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df)
            model_params = {
                'epochs': epochs,
                'batch_size': batch_size,
                'generator_dim': generator_dims,
                'discriminator_dim': generator_dims,
                'generator_lr': learning_rate,
                'discriminator_lr': learning_rate
            }
            if model_type == 'gaussiancopula':
                model_params = {}
            model = train_model(df, metadata, model_type=model_type, **model_params)
            synthetic_data = model.sample(num_rows=num_rows)
            synthetic_data.replace([np.nan, np.inf, -np.inf], None, inplace=True)
            turkish_columns = {col: get_label_by_id(col) for col in synthetic_data.columns}
            synthetic_data.rename(columns=turkish_columns, inplace=True)
            metrics = calculate_quality_metrics(df, synthetic_data)
            preview = synthetic_data.head().to_dict('records')
            return jsonify({
                'success': True,
                'preview': preview,
                'metrics': metrics,
                'pii_columns': pii_cols
            })
        else:
            return jsonify({'success': False, 'error': 'Dosya yüklenmedi veya formatı desteklenmiyor.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def train_model(df, metadata, model_type='ctgan', **kwargs):
    try:
        if model_type == 'ctgan':
            allowed = ['epochs', 'batch_size', 'generator_dim', 'discriminator_dim', 'generator_lr', 'discriminator_lr', 'verbose', 'enforce_min_max_values']
            params = {k: v for k, v in kwargs.items() if k in allowed}
            model = CTGANSynthesizer(metadata, **params)
        elif model_type == 'tvae':
            allowed = ['epochs', 'batch_size', 'embedding_dim', 'compress_dims', 'decompress_dims', 'lr', 'verbose', 'enforce_min_max_values']
            params = {k: v for k, v in kwargs.items() if k in allowed}
            model = TVAESynthesizer(metadata, **params)
        elif model_type == 'copulagan':
            allowed = ['epochs', 'batch_size', 'generator_lr', 'discriminator_lr', 'verbose', 'enforce_min_max_values']
            params = {k: v for k, v in kwargs.items() if k in allowed}
            model = CopulaGANSynthesizer(metadata, **params)
        elif model_type == 'gaussiancopula':
            model = GaussianCopulaSynthesizer(metadata)
        else:
            raise ValueError(f"Desteklenmeyen model tipi: {model_type}")
        model.fit(df)
        return model
    except Exception as e:
        import traceback
        raise ValueError(f"Model eğitimi sırasında hata oluştu: {str(e)}\n{traceback.format_exc()}")

def calculate_quality_metrics(real_data, synthetic_data, target_col=None):
    scores = {}
    for col in real_data.columns:
        if pd.api.types.is_numeric_dtype(real_data[col]):
            try:
                val1 = CSTest.compute(real_data, synthetic_data, column_name=col)
                val2 = BoundaryAdherence.compute(real_data, synthetic_data, column_name=col)
                scores[f'CSTest_{col}'] = float(val1) if val1 is not None and not pd.isna(val1) else None
                scores[f'BoundaryAdherence_{col}'] = float(val2) if val2 is not None and not pd.isna(val2) else None
                app.logger.info(f"CSTest_{col}: {val1}, BoundaryAdherence_{col}: {val2}")
            except Exception as e:
                scores[f'CSTest_{col}'] = None
                scores[f'BoundaryAdherence_{col}'] = None
                app.logger.error(f"CSTest/BoundaryAdherence hata ({col}): {str(e)}")
        else:
            try:
                val3 = TVComplement.compute(real_data, synthetic_data, column_name=col)
                scores[f'TVComplement_{col}'] = float(val3) if val3 is not None and not pd.isna(val3) else None
                app.logger.info(f"TVComplement_{col}: {val3}")
            except Exception as e:
                scores[f'TVComplement_{col}'] = None
                app.logger.error(f"TVComplement hata ({col}): {str(e)}")
    try:
        val4 = CorrelationSimilarity.compute(real_data, synthetic_data)
        scores['CorrelationSimilarity'] = float(val4) if val4 is not None and not pd.isna(val4) else None
        app.logger.info(f"CorrelationSimilarity: {val4}")
    except Exception as e:
        scores['CorrelationSimilarity'] = None
        app.logger.error(f"CorrelationSimilarity hata: {str(e)}")
    try:
        val5 = LogisticDetection.compute(real_data, synthetic_data)
        val6 = SVCDetection.compute(real_data, synthetic_data)
        scores['LogisticDetection'] = float(val5) if val5 is not None and not pd.isna(val5) else None
        scores['SVCDetection'] = float(val6) if val6 is not None and not pd.isna(val6) else None
        app.logger.info(f"LogisticDetection: {val5}, SVCDetection: {val6}")
    except Exception as e:
        scores['LogisticDetection'] = None
        scores['SVCDetection'] = None
        app.logger.error(f"Logistic/SVCDetection hata: {str(e)}")
    if target_col and target_col in real_data.columns:
        try:
            val7 = BinaryMLPClassifier.compute(real_data, synthetic_data, target=target_col)
            scores['BinaryMLPClassifier'] = float(val7) if val7 is not None and not pd.isna(val7) else None
            app.logger.info(f"BinaryMLPClassifier: {val7}")
        except Exception as e:
            scores['BinaryMLPClassifier'] = None
            app.logger.error(f"BinaryMLPClassifier hata: {str(e)}")
    return scores

@app.route('/model_download', methods=['POST'])
@login_required
def model_download():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Dosya yüklenmedi'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Dosya seçilmedi'}), 400
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Desteklenmeyen dosya formatı'}), 400
    try:
        model_type = request.form.get('model_type', 'ctgan')
        if model_type not in ['ctgan', 'tvae', 'copulagan', 'gaussian', 'gaussiancopula']:
            return jsonify({'success': False, 'error': 'Geçersiz model tipi'}), 400
        try:
            epochs = int(request.form.get('epochs', 300))
            batch_size = int(request.form.get('batch_size', 500))
            num_rows = int(request.form.get('num_rows', 1000))
            gen_dims_raw = request.form.get('generator_dims', '[256, 256]')
            try:
                if '[' in gen_dims_raw:
                    generator_dims = json.loads(gen_dims_raw)
                else:
                    generator_dims = [int(x) for x in gen_dims_raw.split(',') if x.strip()]
            except Exception:
                generator_dims = [256, 256]
            learning_rate = float(request.form.get('learning_rate', 0.0002))
        except (ValueError, json.JSONDecodeError) as e:
            return jsonify({'success': False, 'error': 'Geçersiz parametre değerleri'}), 400
        if not (1 <= epochs <= 1000):
            return jsonify({'success': False, 'error': 'Epochs 1-1000 arası olmalı'}), 400
        if not (1 <= batch_size <= 10000):
            return jsonify({'success': False, 'error': 'Batch size 1-10000 arası olmalı'}), 400
        if not (1 <= num_rows <= 1000000):
            return jsonify({'success': False, 'error': 'Satır sayısı 1-1.000.000 arası olmalı'}), 400
        filename = generate_unique_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        try:
            real_data = pd.read_csv(temp_path)
            real_data = turkce_sutun_adlari(real_data)
            real_data.columns = [str(col) for col in real_data.columns]
            # PII checkbox'ları ile gelen sütunları çıkar
            exclude_pii_cols = request.form.getlist('exclude_pii_cols')
            if exclude_pii_cols:
                real_data = real_data.drop(columns=exclude_pii_cols)
            # Metadata hazırla ve PII sütunlarını işle
            meta_data = SingleTableMetadata()
            meta_data.detect_from_dataframe(real_data)
            model_params = {
                'epochs': epochs,
                'batch_size': batch_size,
                'generator_dim': generator_dims,
                'discriminator_dim': generator_dims,
                'generator_lr': learning_rate,
                'discriminator_lr': learning_rate
            }
            model = train_model(real_data, meta_data, model_type=model_type, **model_params)
            synthetic_data = model.sample(num_rows=num_rows)
            synthetic_data.replace([np.nan, np.inf, -np.inf], None, inplace=True)
            turkish_columns = {col: get_label_by_id(col) for col in synthetic_data.columns}
            synthetic_data.rename(columns=turkish_columns, inplace=True)
            metrics = None
            try:
                metrics = calculate_quality_metrics(real_data, synthetic_data)
            except Exception as met_ex:
                app.logger.error(f"Kalite metrikleri hesaplanamadı: {str(met_ex)}")
                metrics = {'error': str(met_ex)}
            file_format = request.form.get('format', 'csv')
            output_filename = f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_format}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            if file_format == 'csv':
                synthetic_data.to_csv(output_path, index=False)
            else:
                synthetic_data.to_excel(output_path, index=False)
            activity = UserActivity(
                user_id=current_user.id,
                action='model_data_generation',
                data_type=model_type,
                row_count=num_rows,
                column_count=len(synthetic_data.columns),
                file_format=file_format
            )
            db.session.add(activity)
            db.session.commit()
            os.remove(temp_path)
            return send_file(
                output_path,
                as_attachment=True,
                download_name=output_filename,
                mimetype='text/csv' if file_format == 'csv' else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            app.logger.error(f"Model download error: {str(e)}")
            return jsonify({'success': False, 'error': f'İndirme sırasında hata: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f"Model download error: {str(e)}")
        return jsonify({'success': False, 'error': f'İndirme sırasında hata: {str(e)}'}), 500

@app.route('/download_file/<filename>')
@login_required
def download_file(filename):
    """Üretilen dosyayı indir"""
    try:
        return send_from_directory(
            app.config['UPLOAD_FOLDER'],
            filename,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        flash('Dosya indirilemedi', 'error')
        return redirect(url_for('model_generate'))

@app.route('/model_share', methods=['POST'])
@login_required
def model_share():
    try:
        file = request.files['file']
        if file.filename == '':
            app.logger.error('Paylaş: Dosya seçilmedi')
            return jsonify({'success': False, 'error': 'Dosya seçilmedi'}), 400
        if not allowed_file(file.filename):
            app.logger.error('Paylaş: Desteklenmeyen dosya formatı')
            return jsonify({'success': False, 'error': 'Desteklenmeyen dosya formatı'}), 400
        model_type = request.form.get('model_type', 'ctgan')
        epochs = int(request.form.get('epochs', 300))
        batch_size = int(request.form.get('batch_size', 500))
        num_rows = int(request.form.get('num_rows', 1000))
        gen_dims_raw = request.form.get('generator_dims', '[256, 256]')
        try:
            if '[' in gen_dims_raw:
                generator_dims = json.loads(gen_dims_raw)
            else:
                generator_dims = [int(x) for x in gen_dims_raw.split(',') if x.strip()]
        except Exception:
            generator_dims = [256, 256]
        learning_rate = float(request.form.get('learning_rate', 0.0002))
        exclude_pii_cols = request.form.getlist('exclude_pii_cols')
        exclude_pii = request.form.get('exclude_pii', 'false') == 'true'
        filename = generate_unique_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        try:
            real_data = pd.read_csv(temp_path)
            pii_cols = detect_pii_columns(real_data)
            if exclude_pii_cols:
                real_data = real_data.drop(columns=exclude_pii_cols)
            elif exclude_pii and pii_cols:
                real_data = real_data.drop(columns=pii_cols)
            real_data = turkce_sutun_adlari(real_data)
            real_data.columns = [str(col) for col in real_data.columns]
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(real_data)
            model_params = {
                'epochs': epochs,
                'batch_size': batch_size,
                'generator_dim': generator_dims,
                'discriminator_dim': generator_dims,
                'generator_lr': learning_rate,
                'discriminator_lr': learning_rate
            }
            if model_type == 'gaussiancopula':
                model_params = {}
            model = train_model(real_data, metadata, model_type=model_type, **model_params)
            synthetic_data = model.sample(num_rows=num_rows)
            synthetic_data.replace([np.nan, np.inf, -np.inf], None, inplace=True)
            turkish_columns = {col: get_label_by_id(col) for col in synthetic_data.columns}
            synthetic_data.rename(columns=turkish_columns, inplace=True)
            metrics = None
            metrics_debug = None
            try:
                metrics = calculate_quality_metrics(real_data, synthetic_data)
                metrics_debug = metrics
            except Exception as met_ex:
                app.logger.error(f"Paylaş: Kalite metrikleri hesaplanamadı: {str(met_ex)}\n{traceback.format_exc()}")
                metrics = {'error': str(met_ex)}
                metrics_debug = traceback.format_exc()
            share_id = str(uuid.uuid4())
            file_format = request.form.get('format', 'csv')
            output_filename = f"synthetic_data_{share_id}.csv"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            synthetic_data.to_csv(output_path, index=False)
            shared_data = SharedData(
                user_id=current_user.id,
                share_id=share_id,
                data_type='model',
                parameters={
                    'model_type': model_type,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'num_rows': num_rows,
                    'generator_dims': generator_dims,
                    'learning_rate': learning_rate,
                    'metrics': metrics,
                    'original_columns': list(real_data.columns),
                    'synthetic_columns': list(synthetic_data.columns),
                    'pii_columns': pii_cols
                }
            )
            db.session.add(shared_data)
            db.session.commit()
            os.remove(temp_path)
            share_url = url_for('shared_model_data', share_id=share_id, _external=True)
            return jsonify({
                'success': True,
                'message': 'Veri başarıyla paylaşıldı',
                'share_url': share_url,
                'metrics': metrics,
                'metrics_debug': metrics_debug,
                'preview': synthetic_data.head(5).to_dict('records'),
                'pii_columns': pii_cols
            })
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            app.logger.error(f"Paylaş: Hata: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'success': False, 'error': f'Paylaşım sırasında hata: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f"Paylaş: Hata: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Paylaşım sırasında hata: {str(e)}'}), 500

@app.route('/shared/model/<share_id>')
def shared_model_data(share_id):
    """Paylaşılan model verisini görüntüle"""
    try:
        shared_data = SharedData.query.filter_by(share_id=share_id).first_or_404()
        
        # Dosya yolunu oluştur
        filename = f"synthetic_data_{share_id}.csv"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            flash('Paylaşılan dosya bulunamadı', 'error')
            return redirect(url_for('home'))
        
        # Veriyi oku
        synthetic_data = pd.read_csv(file_path)
        
        return render_template(
            'shared_model.html',
            shared_data=shared_data,
            preview=synthetic_data.head(10).to_dict('records'),
            columns=synthetic_data.columns.tolist(),
            metrics=shared_data.parameters.get('metrics', {})
        )
        
    except Exception as e:
        app.logger.error(f"Paylaşılan model görüntüleme hatası: {str(e)}")
        flash('Paylaşılan veri görüntülenemedi', 'error')
        return redirect(url_for('home'))

def generate_unique_filename(original_filename):
    """Benzersiz dosya adı oluşturur"""
    ext = os.path.splitext(original_filename)[1]
    return f"{uuid.uuid4().hex}{ext}"

def remove_pii_from_metadata(df, primary_key=None):
    # PII sütunlarını metadata'dan çıkar
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    return metadata

def turkce_sutun_adlari(dataframe):
    # İngilizce PII sütun adlarını Türkçeye çevir
    mapping = {
        'name': 'isim',
        'full_name': 'tam_isim',
        'first_name': 'isim',
        'last_name': 'soyisim',
        'email': 'eposta',
        'e_mail': 'eposta',
        'email_address': 'eposta',
        'phone': 'telefon',
        'phone_number': 'telefon',
        'tel': 'telefon',
        'address': 'adres',
        'street': 'sokak',
        'city': 'sehir',
        'zipcode': 'posta_kodu',
        'ssn': 'tckn',
        'id_number': 'kimlik_no',
        'national_id': 'kimlik_no',
        'dob': 'dogum_tarihi',
        'date_of_birth': 'dogum_tarihi',
        'ip': 'ip_adresi',
        'ip_address': 'ip_adresi'
    }
    # Küçük harfe çevirerek eşleştir
    new_columns = {col: mapping.get(col.lower(), col) for col in dataframe.columns}
    return dataframe.rename(columns=new_columns)

def clean_dataframe(df):
    """
    DataFrame'i temizler:
    1. Hatalı sütunları kaldırır (dict, list, DataFrame içerenler)
    2. Object tiplerini string'e çevirir
    3. NaN değerleri boş string ile değiştirir
    """
    df = df.copy()
    # Hatalı sütunları kaldır
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (dict, list, pd.DataFrame))).any():
            print(f"Hatalı sütun: {col} - kaldırılıyor")
            df = df.drop(columns=[col])
    # Object tiplerini string'e çevir
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    # NaN değerleri boş string ile değiştir
    df = df.fillna('')
    return df

class SharedData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    share_id = db.Column(db.String(36), unique=True, nullable=False)
    data_type = db.Column(db.String(50), nullable=False)  # 'model' veya 'basic'
    parameters = db.Column(db.JSON, nullable=False)  # Model parametreleri
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('shared_data', lazy=True))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
