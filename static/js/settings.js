/**
 * Settings Page JavaScript
 * Handles file upload, training triggers, and status updates
 */

// Global state
let selectedFile = null;
let statusPollInterval = null;

// Initialize page
document.addEventListener('DOMContentLoaded', function () {
    loadDatasetInfo();
    loadTrainingStatus();
    setupUploadZone();
});

/**
 * Load current dataset information
 */
async function loadDatasetInfo() {
    try {
        const response = await fetch('/api/dataset-info');
        const data = await response.json();

        document.getElementById('dataset-name').textContent = data.filename || 'No dataset loaded';
        document.getElementById('dataset-rows').textContent = data.rows ? data.rows.toLocaleString() : '--';

        if (data.date_range && data.date_range.start) {
            const start = new Date(data.date_range.start).toLocaleDateString();
            const end = new Date(data.date_range.end).toLocaleDateString();
            document.getElementById('dataset-range').textContent = `${start} - ${end}`;
        } else {
            document.getElementById('dataset-range').textContent = '--';
        }

        if (data.last_trained) {
            const lastTrained = new Date(data.last_trained);
            document.getElementById('last-trained').textContent = lastTrained.toLocaleString();
        } else if (data.last_modified) {
            const lastMod = new Date(data.last_modified);
            document.getElementById('last-trained').textContent = lastMod.toLocaleString();
        } else {
            document.getElementById('last-trained').textContent = 'Never';
        }
    } catch (error) {
        console.error('Error loading dataset info:', error);
    }
}

/**
 * Load training status
 */
async function loadTrainingStatus() {
    try {
        const response = await fetch('/api/training-status');
        const status = await response.json();

        updateTrainingUI(status);

        // Update history
        if (status.history && status.history.length > 0) {
            renderTrainingHistory(status.history);
        }

        // If training is in progress, start polling
        if (status.is_training && !statusPollInterval) {
            startStatusPolling();
        }
    } catch (error) {
        console.error('Error loading training status:', error);
    }
}

/**
 * Update training UI based on status
 */
function updateTrainingUI(status) {
    const indicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const progressContainer = document.getElementById('progress-container');
    const progressFill = document.getElementById('progress-fill');
    const progressStep = document.getElementById('progress-step');
    const progressPercent = document.getElementById('progress-percent');
    const message = document.getElementById('training-message');

    // Remove all status classes
    indicator.className = 'status-indicator';

    if (status.is_training) {
        indicator.classList.add('training');
        statusText.textContent = 'Training in Progress';
        progressContainer.style.display = 'block';
        progressFill.style.width = `${status.progress}%`;
        progressStep.textContent = status.current_step;
        progressPercent.textContent = `${status.progress}%`;
        message.textContent = '';
    } else if (status.error) {
        indicator.classList.add('error');
        statusText.textContent = 'Error';
        progressContainer.style.display = 'none';
        message.innerHTML = `<div class="error-message">❌ ${status.error}</div>`;
    } else if (status.progress === 100) {
        indicator.classList.add('complete');
        statusText.textContent = 'Complete';
        progressContainer.style.display = 'block';
        progressFill.style.width = '100%';
        progressStep.textContent = status.current_step;
        progressPercent.textContent = '100%';
        message.innerHTML = '<div class="success-message">✅ Models trained successfully!</div>';
    } else {
        indicator.classList.add('idle');
        statusText.textContent = 'Idle';
        progressContainer.style.display = 'none';
        message.textContent = '';
    }
}

/**
 * Setup drag and drop upload zone
 */
function setupUploadZone() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');

    // Click to upload
    uploadZone.addEventListener('click', () => fileInput.click());

    // File selected
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');

        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.name.endsWith('.csv')) {
                handleFileSelect(file);
            } else {
                showMessage('Please upload a CSV file', 'error');
            }
        }
    });
}

/**
 * Handle file selection
 */
function handleFileSelect(file) {
    if (!file.name.endsWith('.csv')) {
        showMessage('Please upload a CSV file', 'error');
        return;
    }

    selectedFile = file;

    // Update UI
    document.getElementById('upload-zone').style.display = 'none';
    document.getElementById('upload-info').style.display = 'block';
    document.getElementById('selected-file-name').textContent = file.name;
    document.getElementById('selected-file-size').textContent = formatFileSize(file.size);
    document.getElementById('upload-btn').disabled = false;
}

/**
 * Remove selected file
 */
function removeFile() {
    selectedFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('upload-zone').style.display = 'block';
    document.getElementById('upload-info').style.display = 'none';
    document.getElementById('upload-btn').disabled = true;
}

/**
 * Upload file and start training
 */
async function uploadAndTrain() {
    if (!selectedFile) {
        showMessage('Please select a file first', 'error');
        return;
    }

    const uploadBtn = document.getElementById('upload-btn');
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<span>⏳</span> Uploading...';

    try {
        // Upload file
        const formData = new FormData();
        formData.append('file', selectedFile);

        const uploadResponse = await fetch('/api/upload-dataset', {
            method: 'POST',
            body: formData
        });

        const uploadResult = await uploadResponse.json();

        if (uploadResult.error) {
            showMessage(uploadResult.error, 'error');
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<span>📤</span> Upload & Train Models';
            return;
        }

        // Start training
        uploadBtn.innerHTML = '<span>🔄</span> Starting Training...';

        const trainResponse = await fetch('/api/train-models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ filepath: uploadResult.filepath })
        });

        const trainResult = await trainResponse.json();

        if (trainResult.error) {
            showMessage(trainResult.error, 'error');
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<span>📤</span> Upload & Train Models';
            return;
        }

        // Training started - start polling for status
        showMessage('Training started! Please wait...', 'success');
        startStatusPolling();

        // Reset UI
        removeFile();
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span>📤</span> Upload & Train Models';

    } catch (error) {
        console.error('Error:', error);
        showMessage('An error occurred: ' + error.message, 'error');
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<span>📤</span> Upload & Train Models';
    }
}

/**
 * Start polling for training status
 */
function startStatusPolling() {
    if (statusPollInterval) {
        clearInterval(statusPollInterval);
    }

    statusPollInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/training-status');
            const status = await response.json();

            updateTrainingUI(status);

            // Update history if available
            if (status.history && status.history.length > 0) {
                renderTrainingHistory(status.history);
            }

            // Stop polling when training is done
            if (!status.is_training) {
                clearInterval(statusPollInterval);
                statusPollInterval = null;

                // Reload dataset info after training completes
                if (status.progress === 100) {
                    loadDatasetInfo();
                }
            }
        } catch (error) {
            console.error('Error polling status:', error);
        }
    }, 1000);
}

/**
 * Render training history
 */
function renderTrainingHistory(history) {
    const container = document.getElementById('training-history');

    if (!history || history.length === 0) {
        container.innerHTML = '<div class="empty-state"><p>No training history yet</p></div>';
        return;
    }

    const html = history.map(item => {
        const date = new Date(item.timestamp);
        return `
            <div class="history-item">
                <div class="history-header">
                    <span class="history-date">${date.toLocaleString()}</span>
                    <span class="history-file">${item.dataset}</span>
                </div>
                <div class="history-details">
                    <span class="history-rows">${item.rows.toLocaleString()} rows</span>
                    <span class="history-metric">LR R²: ${item.lr_r2}</span>
                    <span class="history-metric">LSTM R²: ${item.lstm_r2}</span>
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = html;
}

/**
 * Show message to user
 */
function showMessage(text, type = 'info') {
    const message = document.getElementById('training-message');
    const className = type === 'error' ? 'error-message' : 'success-message';
    const icon = type === 'error' ? '❌' : '✅';
    message.innerHTML = `<div class="${className}">${icon} ${text}</div>`;

    // Clear message after 5 seconds for non-error messages
    if (type !== 'error') {
        setTimeout(() => {
            message.innerHTML = '';
        }, 5000);
    }
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
