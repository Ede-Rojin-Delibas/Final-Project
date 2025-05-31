document.addEventListener('DOMContentLoaded', function() {
    let currentStep = 1;
    let selectedCategory = null;
    let columns = 0;
    let rows = 0;

    // Tümünü Seç checkbox'ı için event listener
    const allCategoriesCheckbox = document.getElementById('allCategoriesCheckbox');
    if (allCategoriesCheckbox) {
        allCategoriesCheckbox.addEventListener('change', function() {
            const checkboxes = document.querySelectorAll('input[name="data_types[]"]');
            checkboxes.forEach(checkbox => {
                checkbox.checked = this.checked;
            });
        });
    }

    function showStep(step) {
        for (let i = 1; i <= 4; i++) {
            const el = document.getElementById('step' + i);
            if (el) el.classList.add('hidden');
        }
        const el = document.getElementById('step' + step);
        if (el) el.classList.remove('hidden');
        currentStep = step;
    }

    window.nextStep = function(step) {
        if (step === 1) {
            rows = parseInt(document.getElementById('rowsInput').value);
            columns = parseInt(document.getElementById('columnsInput').value);
            if (!rows || !columns || rows < 1 || columns < 1) {
                alert('Lütfen geçerli satır ve sütun sayısı girin.');
                return;
            }
        }
        if (step === 2) {
            if (!selectedCategory) {
                alert('Lütfen bir kategori seçin.');
                return;
            }
            loadDataTypes(selectedCategory, columns);
        }
        if (step === 3) {
            const checked = document.querySelectorAll('.data-type-checkbox:checked');
            if (checked.length < 1) {
                alert('Lütfen en az bir veri türü seçin.');
                return;
            }
        }
        showStep(step + 1);
    };

    window.prevStep = function(step) {
        showStep(step - 1);
    };

    // Kategori seçimi
    document.querySelectorAll('.category-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.category-btn').forEach(b => {
                b.classList.remove('border-blue-500', 'bg-blue-50', 'ring-2', 'ring-blue-200');
                b.classList.add('bg-gray-100', 'border-gray-300');
            });
            this.classList.add('border-blue-500', 'bg-blue-50', 'ring-2', 'ring-blue-200');
            this.classList.remove('bg-gray-100', 'border-gray-300');
            selectedCategory = this.dataset.category;
        });
    });

    // Veri türü seçimlerini oluştur
    window.loadDataTypes = function(category, columns) {
        fetch(`/get_data_types/${category}`)
            .then(response => response.json())
            .then(data => {
                const types = data.data;
                const container = document.getElementById('dataTypeSelections');
                container.innerHTML = '';

                // Sayaç/uyarıyı başta göster
                updateColumnInfo();

                // Tümünü Seç butonu
                const selectAllBtn = document.createElement('button');
                selectAllBtn.type = 'button';
                selectAllBtn.textContent = 'Tümünü Seç';
                selectAllBtn.className = 'mb-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 font-semibold shadow transition';
                selectAllBtn.onclick = function() {
                    document.querySelectorAll('.data-type-checkbox').forEach(cb => cb.checked = true);
                    updateColumnInfo();
                };
                container.appendChild(selectAllBtn);

                // Checkbox listesi
                types.forEach(type => {
                    const div = document.createElement('div');
                    div.className = 'mb-2 flex items-center gap-2 relative';

                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.className = 'data-type-checkbox accent-blue-600';
                    checkbox.value = type.id;
                    checkbox.id = `data_type_${type.id}`;
                    checkbox.addEventListener('change', updateColumnInfo);

                    const label = document.createElement('label');
                    label.htmlFor = checkbox.id;
                    label.className = 'text-base text-gray-700 cursor-pointer hover:text-blue-700 transition';
                    label.textContent = type.label;

                    // Tooltip info icon
                    const infoIcon = document.createElement('span');
                    infoIcon.className = 'ml-2 text-blue-500 cursor-pointer relative hover:scale-110 transition-transform';
                    infoIcon.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor"><title>Bilgi</title><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M12 20a8 8 0 100-16 8 8 0 000 16z" /></svg>`;
                    const tooltip = document.createElement('div');
                    tooltip.className = 'hidden absolute left-8 top-1 bg-blue-900 text-white text-xs rounded py-1 px-2 z-50 shadow-lg';
                    tooltip.style.minWidth = '200px';
                    tooltip.style.maxWidth = '300px';
                    tooltip.style.whiteSpace = 'normal';
                    tooltip.textContent = type.description || 'Açıklama yok';

                    infoIcon.addEventListener('mouseenter', function() {
                        tooltip.classList.remove('hidden');
                    });
                    infoIcon.addEventListener('mouseleave', function() {
                        tooltip.classList.add('hidden');
                    });

                    div.appendChild(checkbox);
                    div.appendChild(label);
                    div.appendChild(infoIcon);
                    div.appendChild(tooltip);
                    container.appendChild(div);
                });
            });
    };

    // Sütun sayısı değiştiğinde veri türü seçimlerini güncelle
    document.getElementById('columnsInput').addEventListener('change', function() {
        columns = parseInt(this.value);
        if (columns > 0 && selectedCategory) {
            loadDataTypes(selectedCategory, columns);
        }
    });

    // Sayfa yüklendiğinde ilk adımı göster
    showStep(1);

    // Veri üretimi ve önizleme
    window.generateData = function() {
        const form = document.getElementById('generateForm');
        const formData = new FormData(form);
        formData.append('rows', rows);

        // Seçili veri türlerini ekle
        const checked = document.querySelectorAll('.data-type-checkbox:checked');
        formData.append('columns', checked.length);
        let i = 0;
        checked.forEach(cb => {
            formData.append(`data_type_${i}`, cb.value);
            i++;
        });
        formData.append('format', document.getElementById('formatSelect').value);

        // Yükleniyor animasyonu
        const generateBtn = document.getElementById('generateButton');
        const originalBtnHtml = generateBtn.innerHTML;
        generateBtn.innerHTML = '<svg class="animate-spin h-5 w-5 mr-2 inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Yükleniyor...';
        generateBtn.disabled = true;

        fetch('/preview_data', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Önizleme tablosunu doldur
                const previewTable = document.getElementById('previewTable');
                previewTable.innerHTML = '';
                if (data.preview && data.preview.length > 0) {
                    const table = document.createElement('table');
                    table.className = 'min-w-full divide-y divide-gray-200 border border-gray-300 rounded-lg shadow';
                    const thead = document.createElement('thead');
                    thead.className = 'bg-gray-50';
                    const headerRow = document.createElement('tr');
                    Object.keys(data.preview[0]).forEach(col => {
                        const th = document.createElement('th');
                        th.className = 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
                        th.textContent = col;
                        headerRow.appendChild(th);
                    });
                    thead.appendChild(headerRow);
                    table.appendChild(thead);
                    const tbody = document.createElement('tbody');
                    data.preview.forEach((row, idx) => {
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
                    previewTable.appendChild(table);
                }
                document.getElementById('downloadButton').classList.remove('hidden');
                showToast('Veri başarıyla üretildi!', 'success');
                showAfterGenerate();
            } else {
                alert(data.error || 'Veri önizlemesi alınamadı.');
            }
        })
        .catch(error => {
            alert('Veri üretimi sırasında bir hata oluştu: ' + error.message);
        })
        .finally(() => {
            generateBtn.innerHTML = originalBtnHtml;
            generateBtn.disabled = false;
        });
    };

    // İndir butonu için event listener
    document.getElementById('downloadButton').addEventListener('click', function() {
        const form = document.getElementById('generateForm');
        const formData = new FormData(form);
        formData.append('rows', rows);
        const checked = document.querySelectorAll('.data-type-checkbox:checked');
        formData.append('columns', checked.length);
        let i = 0;
        checked.forEach(cb => {
            formData.append(`data_type_${i}`, cb.value);
            i++;
        });
        formData.append('format', document.getElementById('formatSelect').value);

        // Yükleniyor animasyonu
        const downloadBtn = this;
        const originalBtnHtml = downloadBtn.innerHTML;
        downloadBtn.innerHTML = 'İndiriliyor...';
        downloadBtn.disabled = true;

        fetch('/download_data', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) throw new Error('İndirme başarısız.');
                return response.blob();
            })
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
            a.download = 'generated_data.csv';
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
            })
            .catch(error => {
                alert('Dosya indirilemedi: ' + error.message);
            })
            .finally(() => {
                downloadBtn.innerHTML = originalBtnHtml;
                downloadBtn.disabled = false;
            });
    });

    function updateColumnInfo() {
        const checked = document.querySelectorAll('.data-type-checkbox:checked').length;
        const infoDiv = document.getElementById('columnInfo');
        infoDiv.textContent = `Toplam sütun: ${columns} | Seçili veri türü: ${checked}`;
        if (checked < columns) {
            infoDiv.classList.remove('text-green-700');
            infoDiv.classList.add('text-blue-700');
        } else if (checked === columns) {
            infoDiv.classList.remove('text-blue-700');
            infoDiv.classList.add('text-green-700');
        }
    }

    // Toast bildirimi fonksiyonu
    function showToast(message, type = 'success') {
        const container = document.getElementById('toastContainer');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = `mb-2 px-6 py-4 rounded-lg shadow-lg text-white font-semibold flex items-center gap-2 transition-all duration-300 ${type === 'success' ? 'bg-green-600' : 'bg-red-600'}`;
        toast.innerHTML = `<svg class='w-6 h-6' fill='none' stroke='currentColor' stroke-width='2' viewBox='0 0 24 24'><path stroke-linecap='round' stroke-linejoin='round' d='M5 13l4 4L19 7'/></svg> <span>${message}</span>`;
        container.appendChild(toast);
        setTimeout(() => {
            toast.classList.add('opacity-0');
            setTimeout(() => toast.remove(), 500);
        }, 2500);
    }

    // Tabloyu CSV olarak panoya kopyala
    function copyTableToClipboard(tableId) {
        const table = document.getElementById(tableId)?.querySelector('table');
        if (!table) return;
        let csv = '';
        // Başlıklar
        const headers = Array.from(table.querySelectorAll('thead th')).map(th => th.textContent);
        csv += headers.join(',') + '\n';
        // Satırlar
        table.querySelectorAll('tbody tr').forEach(tr => {
            const row = Array.from(tr.querySelectorAll('td')).map(td => td.textContent);
            csv += row.join(',') + '\n';
        });
        navigator.clipboard.writeText(csv).then(() => {
            showToast('Tablo panoya kopyalandı!', 'success');
        });
    }

    // Kopyala butonunu göster ve işlevini bağla
    function showCopyButton() {
        const btn = document.getElementById('copyTableButton');
        btn.classList.remove('hidden');
        btn.onclick = () => copyTableToClipboard('previewTable');
    }

    // Paylaş butonunu göster ve işlevini bağla
    function showShareButton() {
        const btn = document.getElementById('shareButton');
        btn.classList.remove('hidden');
        btn.onclick = function() {
            btn.disabled = true;
            btn.innerHTML = 'Oluşturuluyor...';
            const form = document.getElementById('generateForm');
            const formData = new FormData(form);

            formData.set('rows', document.getElementById('rowsInput').value);
            formData.set('columns', document.getElementById('columnsInput').value);

            const checked = document.querySelectorAll('.data-type-checkbox:checked');
            let i = 0;
            checked.forEach(cb => {
                formData.append(`data_type_${i}`, cb.value);
                i++;
            });

            formData.set('format', document.getElementById('formatSelect').value);

            fetch('/share_data', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const box = document.getElementById('shareLinkBox');
                    box.classList.remove('hidden');
                    box.innerHTML = `
                        <input type='text' readonly value='${data.url}' class='w-72 px-2 py-1 rounded border border-gray-300 text-gray-800 font-mono text-sm'>
                        <button type="button" class='px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 text-gray-700 font-semibold'
                            onclick='navigator.clipboard.writeText("${data.url}");showToast("Link panoya kopyalandı!","success")'>
                            Kopyala
                        </button>
                    `;
                    showToast('Paylaşılabilir link oluşturuldu!', 'success');
                } else {
                    showToast(data.error || 'Link oluşturulamadı.', 'error');
                }
            })
            .catch(() => showToast('Link oluşturulamadı.', 'error'))
            .finally(() => {
                btn.disabled = false;
                btn.innerHTML = 'Paylaş';
            });
        };
    }

    // Veri üretimi sonrası gösterilecek butonlar
    function showAfterGenerate() {
        showCopyButton();
        showShareButton();
    }

    // Local storage işlemleri
    function saveSelectionsToLocal() {
        const rows = document.getElementById('rowsInput')?.value;
        const format = document.getElementById('formatSelect')?.value;
        const dataTypes = [];
        document.querySelectorAll('.data-type-checkbox:checked')?.forEach(cb => {
            dataTypes.push(cb.value);
        });
        localStorage.setItem('rows', rows);
        localStorage.setItem('format', format);
        localStorage.setItem('data_types', JSON.stringify(dataTypes));
    }

    function loadSelectionsFromLocal() {
        const rows = localStorage.getItem('rows');
        const format = localStorage.getItem('format');
        const dataTypes = JSON.parse(localStorage.getItem('data_types') || '[]');
        if (rows) document.getElementById('rowsInput').value = rows;
        if (format) document.getElementById('formatSelect').value = format;
        if (dataTypes.length) {
            document.querySelectorAll('.data-type-checkbox')?.forEach(cb => {
                cb.checked = dataTypes.includes(cb.value);
            });
        }
    }

    // Sayfa yüklendiğinde seçimleri yükle
    loadSelectionsFromLocal();

    // Seçim değiştiğinde kaydet
    document.getElementById('rowsInput')?.addEventListener('input', saveSelectionsToLocal);
    document.getElementById('formatSelect')?.addEventListener('change', saveSelectionsToLocal);
    document.querySelectorAll('.data-type-checkbox')?.forEach(cb => {
        cb.addEventListener('change', saveSelectionsToLocal);
    });
});



