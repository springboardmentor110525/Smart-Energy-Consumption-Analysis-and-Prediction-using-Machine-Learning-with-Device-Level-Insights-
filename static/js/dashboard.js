// Smart Energy Dashboard - JavaScript

// API base URL
const API_BASE = '';

// Chart.js default configuration
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#334155';

// Load all dashboard data
async function loadDashboard() {
    try {
        await Promise.all([
            loadOverview(),
            loadHourlyChart(),
            loadDailyChart(),
            loadDeviceData(),
            loadModelComparison(),
            loadSuggestions()
        ]);
    } catch (error) {
        console.error('Error loading dashboard:', error);
    }
}

// Load overview statistics
async function loadOverview() {
    try {
        const response = await fetch(`${API_BASE}/api/overview`);
        const data = await response.json();
        
        document.getElementById('total-usage').textContent = `${data.total_usage_kwh} kWh`;
        document.getElementById('avg-usage').textContent = `${data.avg_usage_kw} kW`;
        document.getElementById('solar-gen').textContent = `${data.solar_generation_kwh} kWh`;
        document.getElementById('max-usage').textContent = `${data.max_usage_kw} kW`;
        document.getElementById('date-range').textContent = 
            `Data: ${data.date_range.start.split(' ')[0]} to ${data.date_range.end.split(' ')[0]}`;
    } catch (error) {
        console.error('Error loading overview:', error);
    }
}

// Load hourly consumption chart
async function loadHourlyChart() {
    try {
        const response = await fetch(`${API_BASE}/api/hourly`);
        const data = await response.json();
        
        const ctx = document.getElementById('hourlyChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels.map(h => `${h}:00`),
                datasets: [{
                    label: 'Avg. Consumption (kW)',
                    data: data.data,
                    backgroundColor: createGradient(ctx, '#6366f1', '#10b981'),
                    borderRadius: 4,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: '#334155'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading hourly chart:', error);
    }
}

// Load daily consumption chart
async function loadDailyChart() {
    try {
        const response = await fetch(`${API_BASE}/api/daily`);
        const data = await response.json();
        
        const ctx = document.getElementById('dailyChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Daily Consumption (kWh)',
                    data: data.data,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 3,
                    pointBackgroundColor: '#10b981'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: '#334155'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            maxTicksLimit: 7
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading daily chart:', error);
    }
}

// Load device consumption data
async function loadDeviceData() {
    try {
        const response = await fetch(`${API_BASE}/api/devices`);
        const data = await response.json();
        
        const devices = data.devices.slice(0, 10); // Top 10 devices
        
        // Device pie chart
        const ctx = document.getElementById('deviceChart').getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: devices.map(d => d.name),
                datasets: [{
                    data: devices.map(d => d.percentage),
                    backgroundColor: [
                        '#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
                        '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16'
                    ],
                    borderWidth: 2,
                    borderColor: '#1e293b'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            padding: 15,
                            usePointStyle: true
                        }
                    }
                }
            }
        });
        
        // Device list
        const deviceList = document.getElementById('device-list');
        deviceList.innerHTML = devices.slice(0, 5).map(device => `
            <div class="device-item">
                <div>
                    <div class="device-name">${device.name}</div>
                    <div class="device-bar">
                        <div class="device-bar-fill" style="width: ${device.percentage}%"></div>
                    </div>
                </div>
                <div class="device-usage">${device.percentage}%</div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading device data:', error);
    }
}

// Load model comparison
async function loadModelComparison() {
    try {
        const response = await fetch(`${API_BASE}/api/model-comparison`);
        const data = await response.json();
        
        if (!data.error) {
            document.getElementById('lr-mae').textContent = data.baseline.mae;
            document.getElementById('lr-rmse').textContent = data.baseline.rmse;
            document.getElementById('lr-r2').textContent = data.baseline.r2;
            
            document.getElementById('lstm-mae').textContent = data.lstm.mae;
            document.getElementById('lstm-rmse').textContent = data.lstm.rmse;
            document.getElementById('lstm-r2').textContent = data.lstm.r2;
        }
    } catch (error) {
        console.error('Error loading model comparison:', error);
    }
}

// Load smart suggestions
async function loadSuggestions() {
    try {
        const response = await fetch(`${API_BASE}/api/suggestions`);
        const data = await response.json();
        
        const suggestionsList = document.getElementById('suggestions-list');
        
        if (data.suggestions && data.suggestions.length > 0) {
            suggestionsList.innerHTML = data.suggestions.map(suggestion => `
                <div class="suggestion-item ${suggestion.priority}">
                    <div class="suggestion-header">
                        <span class="suggestion-device">${suggestion.device}</span>
                        <span class="suggestion-savings">💰 Save: ${suggestion.potential_savings}</span>
                    </div>
                    <p class="suggestion-message">${suggestion.message}</p>
                </div>
            `).join('');
        } else {
            suggestionsList.innerHTML = '<p style="padding: 1rem; color: #94a3b8;">No suggestions available.</p>';
        }
    } catch (error) {
        console.error('Error loading suggestions:', error);
    }
}

// Helper: Create gradient for charts
function createGradient(ctx, color1, color2) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, color1);
    gradient.addColorStop(1, color2);
    return gradient;
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', loadDashboard);
