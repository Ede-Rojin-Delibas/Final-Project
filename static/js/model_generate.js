document.addEventListener('DOMContentLoaded', function() {
    // Model açıklamaları
    const modelDescriptions = {
        'ctgan': 'CTGAN (Conditional Tabular GAN), kategorik ve sürekli değişkenleri işleyebilen bir GAN modelidir. Karmaşık tablo verileri için idealdir.',
        'tvae': 'TVAE (Tabular Variational Autoencoder), varyasyonel otokodlayıcı tabanlı bir modeldir. Daha hızlı eğitim süresi sunar.',
        'copulagan': 'CopulaGAN, korelasyon yapısını koruyan bir GAN modelidir. İlişkili veriler için uygundur.',
        'gaussiancopula': 'Gaussian Copula, basit ve hızlı bir modeldir. Temel istatistiksel özellikleri korur.'
    };

    // Model seçimi değiştiğinde
    const modelTypeSelect = document.getElementById('modelType');
    const modelDescription = document.getElementById('modelDescription');
    const modelParams = document.getElementById('modelParams');

    if (modelTypeSelect) {
        modelTypeSelect.addEventListener('change', function() {
            const selectedModel = this.value;
            if (selectedModel) {
                modelDescription.textContent = modelDescriptions[selectedModel];
                modelParams.classList.remove('hidden');
            } else {
                modelDescription.textContent = 'Model seçiniz...';
                modelParams.classList.add('hidden');
            }
        });
    }

    // Generator dims input kontrolü
    const generatorDimsInput = document.querySelector('input[name="generator_dims"]');
    if (generatorDimsInput) {
        generatorDimsInput.addEventListener('input', function() {
            // Sadece sayılar, virgüller ve köşeli parantezlere izin ver
            this.value = this.value.replace(/[^0-9,\[\]]/g, '');
        });
    }
});

// Veri üretimi fonksiyonu
function generateModelData() {
    const form = document.getElementById('modelForm');
    const formData = new FormData(form);
    const generateButton = document.getElementById('generateButton');
    const originalButtonHtml = generateButton.innerHTML;

    // Generator dims değerini diziye çevir
    const generatorDims = formData.get('generator_dims');
    try {
        formData.set('generator_dims', JSON.stringify(eval(generatorDims)));
    } catch (e) {
        showToast('Generator dims formatı geçersiz. Örnek: [256, 256]', 'error');
        return;
    }

    // Yükleniyor animasyonu
    generateButton.innerHTML = `
        <svg class="animate-spin h-5 w-5 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        Veri Üretiliyor...
    `;
    generateButton.disabled = true;

    fetch('/model_preview', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Sonuçları göster
            document.getElementById('results').classList.remove('hidden');
            
            // Önizleme tablosunu oluştur
            createPreviewTable(data.preview);
            
            // İndirme ve paylaşma butonlarını göster
            document.getElementById('downloadButton').classList.remove('hidden');
            document.getElementById('shareButton').classList.remove('hidden');
            
            // Buton event listener'larını ekle
            setupDownloadButton(data.downloadUrl);
            setupShareButton();
            
            showPIIColumns(data.pii_columns);
            
            // Debug: metriklerin ham JSON'unu göster
            if (data.metrics_debug) {
                const debugBox = document.getElementById('metricsDebugBox');
                if (debugBox) {
                    debugBox.classList.remove('hidden');
                    debugBox.textContent = JSON.stringify(data.metrics_debug, null, 2);
                }
            }
            
            // Kalite metriklerini göster
            if (data.metrics) {
                updateQualityMetrics(data.metrics);
                document.getElementById('qualityMetrics').classList.remove('hidden');
            }
            
            showToast('Veri başarıyla üretildi!', 'success');
        } else {
            showToast(data.error || 'Veri üretimi başarısız oldu.', 'error');
        }
    })
    .catch(error => {
        showToast('Bir hata oluştu: ' + error.message, 'error');
    })
    .finally(() => {
        // Yükleniyor animasyonunu kaldır
        generateButton.innerHTML = originalButtonHtml;
        generateButton.disabled = false;
    });
}

// Önizleme tablosunu oluştur
function createPreviewTable(data) {
    const container = document.getElementById('previewTable');
    container.innerHTML = '';
    
    if (!data || data.length === 0) return;

    const table = document.createElement('table');
    table.className = 'min-w-full divide-y divide-gray-200 border border-gray-300 rounded-lg shadow';
    
    // Tablo başlığı
    const thead = document.createElement('thead');
    thead.className = 'bg-gray-50';
    const headerRow = document.createElement('tr');
    
    Object.keys(data[0]).forEach(col => {
        const th = document.createElement('th');
        th.className = 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
        th.textContent = col;
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Tablo gövdesi
    const tbody = document.createElement('tbody');
    data.forEach((row, idx) => {
        const tr = document.createElement('tr');
        tr.className = idx % 2 === 0 ? 'bg-white' : 'bg-gray-50';
        
        Object.values(row).forEach(val => {
            const td = document.createElement('td');
            td.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-900';
            td.textContent = val;
            tr.appendChild(td);
        });
        
        tbody.appendChild(tr);
    });
    
    table.appendChild(tbody);
    container.appendChild(table);
}

// İndirme butonu ayarları
function getCheckedPIIColumns() {
    const checkboxes = document.querySelectorAll('#piiColumnsBox input[type="checkbox"][name="exclude_pii_cols"]');
    const checked = new Set(); // Tekrarları önlemek için Set kullan
    checkboxes.forEach(cb => { 
        if (cb.checked) checked.add(cb.value); 
    });
    return Array.from(checked); // Set'i array'e çevir
}

function setupDownloadButton(downloadUrl) {
    const downloadButton = document.getElementById('downloadButton');
    downloadButton.onclick = function() {
        const form = document.getElementById('modelForm');
        const formData = new FormData(form);
        // PII seçimlerini ekle
        getCheckedPIIColumns().forEach(col => formData.append('exclude_pii_cols', col));
        // Yükleniyor animasyonu
        const originalButtonHtml = this.innerHTML;
        this.innerHTML = 'İndiriliyor...';
        this.disabled = true;
        
        fetch('/model_download', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                // JSON hata mesajını almaya çalış
                return response.json().then(errorData => {
                    throw new Error(errorData.error || 'İndirme başarısız. Sunucu yanıtı alınamadı.');
                }).catch(() => {
                    throw new Error('İndirme başarısız. Sunucu yanıtı alınamadı.');
                });
            }
            
            // Content-Type kontrolü
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                // JSON yanıt geldi, hata olabilir
                return response.json().then(data => {
                    if (!data.success) {
                        throw new Error(data.error || 'İndirme başarısız.');
                    }
                    throw new Error('Beklenmeyen JSON yanıtı.');
                });
            }
            
            // Dosya adını al
            const contentDisposition = response.headers.get('content-disposition');
            let filename = 'synthetic_data.csv';
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
                if (filenameMatch && filenameMatch[1]) {
                    filename = filenameMatch[1].replace(/['"]/g, '');
                }
            }
            
            // Dosya indirme işlemi
            return response.blob().then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                a.style.display = 'none';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                showToast('Dosya başarıyla indirildi!', 'success');
            });
        })
        .catch(error => {
            console.error('İndirme hatası:', error);
            showToast('Dosya indirilemedi: ' + error.message, 'error');
        })
        .finally(() => {
            this.innerHTML = originalButtonHtml;
            this.disabled = false;
        });
    };
}

// Paylaşma butonu ayarları
function setupShareButton() {
    const shareButton = document.getElementById('shareButton');
    shareButton.onclick = function() {
        const form = document.getElementById('modelForm');
        const formData = new FormData(form);
        // PII seçimlerini ekle
        getCheckedPIIColumns().forEach(col => formData.append('exclude_pii_cols', col));
        // Yükleniyor animasyonu
        const originalButtonHtml = this.innerHTML;
        this.innerHTML = 'Paylaşım Linki Oluşturuluyor...';
        this.disabled = true;
        
        fetch('/model_share', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.error || 'Paylaşım başarısız. Sunucu yanıtı alınamadı.');
                }).catch(() => {
                    throw new Error('Paylaşım başarısız. Sunucu yanıtı alınamadı.');
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                const shareLinkBox = document.getElementById('shareLinkBox');
                shareLinkBox.classList.remove('hidden');
                const url = data.share_url || data.url;
                shareLinkBox.innerHTML = `
                    <div class="flex items-center gap-2 w-full">
                        <input type="text" readonly value="${url}" 
                            class="flex-1 px-3 py-2 rounded-lg border border-gray-300 bg-gray-50 text-gray-700 font-mono text-sm">
                        <button onclick="navigator.clipboard.writeText('${url}'); showToast('Link kopyalandı!', 'success')"
                            class="px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded-lg font-medium transition">
                            Kopyala
                        </button>
                    </div>
                `;
                showToast('Paylaşım linki oluşturuldu!', 'success');
            } else {
                showToast(data.error || 'Paylaşım linki oluşturulamadı.', 'error');
            }
        })
        .catch(error => {
            console.error('Paylaşım hatası:', error);
            showToast('Paylaşım linki oluşturulamadı: ' + error.message, 'error');
        })
        .finally(() => {
            this.innerHTML = originalButtonHtml;
            this.disabled = false;
        });
    };
}

// PII kolonları gösterimi
function showPIIColumns(piiColumns) {
    const container = document.getElementById('piiColumnsBox');
    if (!container) return;
    container.innerHTML = '';
    if (!piiColumns || piiColumns.length === 0) {
        container.classList.add('hidden');
        return;
    }
    container.classList.remove('hidden');
    container.innerHTML = `<div class="mb-2 text-yellow-700 font-semibold">Kişisel veri olabilecek sütunlar tespit edildi:</div><div class="mb-1 text-xs text-gray-600">İşaretli olanlar veri setinden çıkarılır. Görmek istediğiniz sütunların işaretini kaldırın.</div>`;
    piiColumns.forEach(col => {
        const div = document.createElement('div');
        div.className = 'flex items-center gap-2 mb-1';
        div.innerHTML = `<input type="checkbox" name="exclude_pii_cols" value="${col}" checked class="accent-yellow-600"> <span>${col}</span>`;
        container.appendChild(div);
    });
    container.innerHTML += `<div class="text-xs text-gray-500 mt-1">Bu sütunları hariç bırakmak istemiyorsanız işaretini kaldırabilirsiniz.</div>`;
}

// Kalite metriklerini güncelle
function updateQualityMetrics(metrics) {
    if (!metrics) return;
    
    const formatScore = (score) => {
        if (score === undefined || score === null || score === '' || score === 'Veri yok' || score === 'Hesaplanamadı') {
            return '-';
        }
        return score.toString();
    };
    
    // ML Efficacy metrikleri
    document.getElementById('mlEfficacyScore').textContent = formatScore(metrics['ML Etkinliği (MLPRegressor)']);
    document.getElementById('linearRegressionScore').textContent = formatScore(metrics['ML Etkinliği (LinearRegression)']);
    
    // İstatistiksel metrikler
    document.getElementById('statisticalScore').textContent = formatScore(metrics['İstatistiksel Benzerlik (CSTest)']);
    document.getElementById('correlationScore').textContent = formatScore(metrics['Korelasyon Benzerliği (CorrelationSimilarity)']);
    document.getElementById('categoryCoverageScore').textContent = formatScore(metrics['Kategori Kapsamı (TVComplement)']);
    document.getElementById('boundaryScore').textContent = formatScore(metrics['Sınır Uyumu (BoundaryAdherence)']);
    
    // Detection metrikleri
    document.getElementById('logisticDetectionScore').textContent = formatScore(metrics['Logistic Detection']);
    document.getElementById('svcDetectionScore').textContent = formatScore(metrics['SVC Detection']);
    
    // Metrik renklerini ayarla (performansa göre)
    updateMetricColors();
}

// Metrik renklerini performansa göre ayarla
function updateMetricColors() {
    const metricElements = document.querySelectorAll('[id$="Score"]');
    
    metricElements.forEach(element => {
        const score = element.textContent;
        if (score === '-' || score === 'Hesaplanamadı' || score === 'Veri yok') {
            element.className = 'text-2xl font-bold text-gray-400';
            return;
        }
        
        // Sayısal değeri çıkar
        const numericValue = parseFloat(score.replace(/[^\d.-]/g, ''));
        if (isNaN(numericValue)) {
            element.className = 'text-2xl font-bold text-gray-500';
            return;
        }
        
        // Detection metrikleri için (düşük değer = iyi)
        if (element.id.includes('Detection')) {
            if (numericValue <= 30) {
                element.className = 'text-2xl font-bold text-green-600';
            } else if (numericValue <= 60) {
                element.className = 'text-2xl font-bold text-yellow-600';
            } else {
                element.className = 'text-2xl font-bold text-red-600';
            }
        } else {
            // Diğer metrikler için (yüksek değer = iyi)
            if (numericValue >= 70) {
                element.className = 'text-2xl font-bold text-green-600';
            } else if (numericValue >= 40) {
                element.className = 'text-2xl font-bold text-yellow-600';
            } else {
                element.className = 'text-2xl font-bold text-red-600';
            }
        }
    });
}

// Toast bildirimi
function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `mb-2 px-6 py-4 rounded-lg shadow-lg text-white font-semibold flex items-center gap-2 transition-all duration-300 ${
        type === 'success' ? 'bg-green-600' : 'bg-red-600'
    }`;
    
    toast.innerHTML = `
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                d="${type === 'success' ? 'M5 13l4 4L19 7' : 'M6 18L18 6M6 6l12 12'}"/>
        </svg>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    // 2.5 saniye sonra kaldır
    setTimeout(() => {
        toast.classList.add('opacity-0');
        setTimeout(() => toast.remove(), 500);
    }, 2500);
}