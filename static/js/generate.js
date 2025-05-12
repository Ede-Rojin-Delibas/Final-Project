document.addEventListener('DOMContentLoaded', function() {
    // DOM elementleri
    const columnsInput = document.getElementById('columnsInput');
    const categorySelect = document.getElementById('categorySelect');
    const generateForm = document.getElementById('generateForm');
    const previewTable = document.getElementById('previewTable');
    const previewHeader = document.getElementById('previewHeader');
    const previewBody = document.getElementById('previewBody');
    const advancedModeToggle = document.getElementById('advancedMode');
    const simpleMode = document.getElementById('simpleMode');
    const advancedMode = document.getElementById('advancedMode');

    // DATA_CATEGORIES'i backend'den al
    let DATA_CATEGORIES = {};
    fetch('/get_all_data_types')
        .then(response => response.json())
        .then(data => {
            DATA_CATEGORIES = data;
        });

    // Form submit işlemi
    generateForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        const submitButton = e.submitter;
        // İndirme butonu için direkt submit
        if (submitButton && submitButton.classList.contains('download-btn')) {
            return true; // Normal form submit işleminin devam etmesini sağlar
        }

        try {
            const response = await fetch('/preview_data', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            if (data.success) {
                updatePreviewTable(data.preview);
                previewTable.classList.remove('hidden');
             
            } else {
                throw new Error(data.error);
            }
        } catch (error) {
            alert('Hata: ' + error.message);
        }
    });
    
    function loadAllDataTypes() {
        const container = document.getElementById('dataTypeSelections');
        const columns = parseInt(columnsInput.value) || 1;
        
        container.innerHTML = '';
        
        for (let i = 0; i < columns; i++) {
            const columnDiv = document.createElement('div');
            columnDiv.className = 'mb-4';
            
            const label = document.createElement('label');
            label.className = 'block text-sm font-medium text-gray-700 mb-1';
            label.textContent = `Sütun ${i + 1} Veri Türü`;
            
            const select = document.createElement('select');
            select.name = `data_type_${i}`;
            select.className = 'w-full border border-gray-300 rounded-lg px-3 py-2';
            select.required = true;
            select.innerHTML = generateDataTypeOptions();
            
            columnDiv.appendChild(label);
            columnDiv.appendChild(select);
            container.appendChild(columnDiv);
        }
    }
    function generateDataTypeOptions() {
        let options = '';
        for(const category in DATA_CATEGORIES) {
            options += `<optgroup label="${category}">`;
            DATA_CATEGORIES[category].forEach(type => {
                options += `<option value="${type.id}">${type.label}</option>`;
            });
            options += '</optgroup>';
        }
        return options;
    }
    // Gelişmiş mod geçiş işlemi
    advancedModeToggle.addEventListener('change', function() {
        if (this.checked) {
            simpleMode.classList.add('hidden');
            advancedMode.classList.remove('hidden');
            loadAllDataTypes();
        } else {
            simpleMode.classList.remove('hidden');
            advancedMode.classList.add('hidden');
        }
    });

    // Kategori değiştiğinde veri türlerini güncelle
    categorySelect.addEventListener('change', function() {
        const selectedCategory = this.value;
        if (selectedCategory) {
            fetch(`/get_data_types/${selectedCategory}`)
                .then(response => response.json())
                .then(types => {
                    updateDataTypeInputs(types);
                });
        }
    });

    // Önizleme tablosunu güncelleme fonksiyonu
    function updatePreviewTable(previewData) {
        if (!previewData || previewData.length === 0) return;

        // Başlıkları oluştur
        const headerRow = document.createElement('tr');
        headerRow.className = 'bg-gray-50';
        
        Object.keys(previewData[0]).forEach(columnName => {
            const th = document.createElement('th');
            th.className = 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
            th.textContent = columnName;
            headerRow.appendChild(th);
        });
        
        previewHeader.innerHTML = '';
        previewHeader.appendChild(headerRow);

        // Tablo gövdesini oluştur
        previewBody.innerHTML = '';
        previewData.forEach((row, index) => {
            const tr = document.createElement('tr');
            tr.className = index % 2 === 0 ? 'bg-white' : 'bg-gray-50';

            Object.values(row).forEach(value => {
                const td = document.createElement('td');
                td.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-900';
                td.textContent = value;
                tr.appendChild(td);
            });

            previewBody.appendChild(tr);
        });
    }

    // Veri türü seçimlerini güncelleme fonksiyonu
    function updateDataTypeInputs(types) {
        const container = document.getElementById('dataTypeSelections');
        container.innerHTML = '';

        const columns = parseInt(columnsInput.value) || 1;
        
        for (let i = 0; i < columns; i++) {
            const columnDiv = document.createElement('div');
            columnDiv.className = 'mb-4';
            
            const label = document.createElement('label');
            label.className = 'block text-sm font-medium text-gray-700 mb-1';
            label.textContent = `Sütun ${i + 1} Veri Türü`;
            
            const select = document.createElement('select');
            select.name = `data_type_${i}`;
            select.className = 'w-full border border-gray-300 rounded-lg px-3 py-2';
            select.required = true;

            // Veri türü seçeneklerini ekle
            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = 'Veri türü seçin';
            select.appendChild(defaultOption);

            types.forEach(type => {
                const option = document.createElement('option');
                option.value = type.id;
                option.textContent = type.label;
                select.appendChild(option);
            });

            columnDiv.appendChild(label);
            columnDiv.appendChild(select);
            container.appendChild(columnDiv);
        }
    }
});




