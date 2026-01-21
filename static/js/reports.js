// Smart Energy Dashboard - Reports Page

const API_BASE = '';
let selectedReportType = 'daily';
let reportData = [];
let dataDateRange = null;

// Initialize page
document.addEventListener('DOMContentLoaded', async () => {
    // First, get the actual date range from the data
    await loadDataDateRange();

    // Report type selection
    document.querySelectorAll('.report-type-card').forEach(card => {
        card.addEventListener('click', () => {
            document.querySelectorAll('.report-type-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            selectedReportType = card.dataset.type;
        });
    });

    // Auto-generate report on load
    generateReport();
});

// Load the actual date range from the dataset
async function loadDataDateRange() {
    try {
        const response = await fetch(`${API_BASE}/api/overview`);
        const data = await response.json();

        if (data.date_range) {
            // Parse the dates from the data
            const startStr = data.date_range.start.split(' ')[0];
            const endStr = data.date_range.end.split(' ')[0];

            dataDateRange = { start: startStr, end: endStr };

            // Set date inputs to match actual data
            document.getElementById('start-date').value = startStr;
            document.getElementById('end-date').value = endStr;

            console.log('Data range:', startStr, 'to', endStr);
        }
    } catch (error) {
        console.error('Error loading date range:', error);
        // Fallback to defaults
        const today = new Date();
        const lastWeek = new Date(today);
        lastWeek.setDate(lastWeek.getDate() - 7);

        document.getElementById('end-date').value = today.toISOString().split('T')[0];
        document.getElementById('start-date').value = lastWeek.toISOString().split('T')[0];
    }
}

// Generate report
async function generateReport() {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    const includeSummary = document.getElementById('inc-summary').checked;
    const includeDevices = document.getElementById('inc-devices').checked;
    const includeHourly = document.getElementById('inc-hourly').checked;
    const includeSuggestions = document.getElementById('inc-suggestions').checked;

    if (!startDate || !endDate) {
        alert('Please select both start and end dates');
        return;
    }

    // Show loading state
    const tbody = document.getElementById('report-table-body');
    tbody.innerHTML = `
        <tr>
            <td colspan="5" style="text-align: center; color: var(--text-secondary);">
                <div class="spinner" style="margin: 1rem auto;"></div>
                Generating report...
            </td>
        </tr>
    `;

    // Hide suggestions section initially
    const suggestionsSection = document.getElementById('suggestions-section');
    if (suggestionsSection) {
        suggestionsSection.style.display = 'none';
    }

    try {
        const response = await fetch(`${API_BASE}/api/generate-report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: selectedReportType,
                start_date: startDate,
                end_date: endDate,
                include: {
                    summary: includeSummary,
                    devices: includeDevices,
                    hourly: includeHourly,
                    suggestions: includeSuggestions
                }
            })
        });

        const data = await response.json();

        if (data.error) {
            console.error('API Error:', data.error);
            tbody.innerHTML = `
                <tr>
                    <td colspan="5" style="text-align: center; color: var(--danger);">
                        Error: ${data.error}
                    </td>
                </tr>
            `;
            return;
        }

        reportData = data.report || [];

        if (reportData.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="5" style="text-align: center; color: var(--text-secondary);">
                        No data available for the selected date range.<br>
                        <small>Try selecting dates within the dataset range${dataDateRange ? ': ' + dataDateRange.start + ' to ' + dataDateRange.end : ''}</small>
                    </td>
                </tr>
            `;
            return;
        }

        displayReport(reportData);

        // Fetch and display suggestions if checkbox is checked
        if (includeSuggestions) {
            await loadSuggestions();
        }

    } catch (error) {
        console.error('Error generating report:', error);
        tbody.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; color: var(--danger);">
                    Failed to generate report. Please try again.
                </td>
            </tr>
        `;
    }
}

// Load and display energy-saving suggestions
async function loadSuggestions() {
    try {
        const response = await fetch(`${API_BASE}/api/suggestions`);
        const data = await response.json();

        const suggestions = data.suggestions || [];
        const suggestionsSection = document.getElementById('suggestions-section');

        if (suggestions.length > 0 && suggestionsSection) {
            suggestionsSection.style.display = 'block';
            const suggestionsBody = document.getElementById('suggestions-body');

            suggestionsBody.innerHTML = suggestions.map(s => `
                <div class="suggestion-item" style="padding: 1rem; border-left: 4px solid ${getPriorityColor(s.priority)}; background: var(--bg-dark); margin-bottom: 0.75rem; border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div>
                            <strong style="color: var(--text-primary);">${s.device}</strong>
                            <span style="font-size: 0.75rem; padding: 0.2rem 0.5rem; border-radius: 4px; background: ${getPriorityColor(s.priority)}20; color: ${getPriorityColor(s.priority)}; margin-left: 0.5rem;">${s.priority}</span>
                        </div>
                        <span style="color: var(--secondary); font-weight: 500;">Save ${s.potential_savings}</span>
                    </div>
                    <p style="margin-top: 0.5rem; color: var(--text-secondary); font-size: 0.9rem;">${s.message}</p>
                </div>
            `).join('');
        }
    } catch (error) {
        console.error('Error loading suggestions:', error);
    }
}

// Get color based on priority
function getPriorityColor(priority) {
    switch (priority) {
        case 'high': return '#ef4444';
        case 'medium': return '#f59e0b';
        case 'low': return '#10b981';
        default: return '#6366f1';
    }
}


// Display report in table
function displayReport(data) {
    const tbody = document.getElementById('report-table-body');

    if (!data || data.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; color: var(--text-secondary);">
                    No data available for the selected period
                </td>
            </tr>
        `;
        document.getElementById('report-summary').style.display = 'none';
        return;
    }

    // Calculate summary stats
    const totalSum = data.reduce((acc, r) => acc + parseFloat(r.total || 0), 0);
    const avgDaily = totalSum / data.length;
    const maxPeak = Math.max(...data.map(r => parseFloat(r.peak || 0)));
    const totalCost = data.reduce((acc, r) => acc + parseFloat(r.cost || 0), 0);

    // Update summary display
    document.getElementById('summary-total').textContent = `${totalSum.toFixed(2)} kWh`;
    document.getElementById('summary-avg').textContent = `${avgDaily.toFixed(2)} kWh`;
    document.getElementById('summary-peak').textContent = `${maxPeak.toFixed(4)} kW`;
    document.getElementById('summary-cost').textContent = `$${totalCost.toFixed(2)}`;
    document.getElementById('report-summary').style.display = 'block';

    // Update print header
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    document.getElementById('print-date-range').textContent = `Report Period: ${startDate} to ${endDate}`;
    document.getElementById('print-generated-date').textContent = new Date().toLocaleString();

    // Render table rows
    tbody.innerHTML = data.map(row => `
        <tr>
            <td>${row.date}</td>
            <td>${row.total} kWh</td>
            <td>${row.average} kW</td>
            <td>${row.peak} kW</td>
            <td>$${row.cost}</td>
        </tr>
    `).join('');

    // Add summary row
    tbody.innerHTML += `
        <tr style="font-weight: bold; background: var(--bg-card-hover);">
            <td>Total</td>
            <td>${totalSum.toFixed(2)} kWh</td>
            <td>-</td>
            <td>${maxPeak.toFixed(4)} kW</td>
            <td>$${totalCost.toFixed(2)}</td>
        </tr>
    `;
}

// Download report as CSV
function downloadReport(format) {
    if (!reportData || reportData.length === 0) {
        alert('Please generate a report first');
        return;
    }

    if (format === 'csv') {
        const headers = ['Date', 'Total (kWh)', 'Average (kW)', 'Peak (kW)', 'Est. Cost ($)'];
        const rows = reportData.map(r => [r.date, r.total, r.average, r.peak, r.cost]);

        // Add summary
        const totalSum = reportData.reduce((acc, r) => acc + parseFloat(r.total || 0), 0).toFixed(2);
        const maxPeak = Math.max(...reportData.map(r => parseFloat(r.peak || 0))).toFixed(4);
        const totalCost = reportData.reduce((acc, r) => acc + parseFloat(r.cost || 0), 0).toFixed(2);
        rows.push(['Total', totalSum, '-', maxPeak, totalCost]);

        const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `energy_report_${selectedReportType}_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        window.URL.revokeObjectURL(url);
    }
}

// Print report with proper formatting
function printReport() {
    if (!reportData || reportData.length === 0) {
        alert('Please generate a report first');
        return;
    }

    // Show print header before printing
    const printHeader = document.getElementById('print-header');
    printHeader.style.display = 'block';

    // Trigger print
    window.print();

    // Hide print header after printing
    setTimeout(() => {
        printHeader.style.display = 'none';
    }, 1000);
}

// Set date range to match data
function useDataRange() {
    if (dataDateRange) {
        document.getElementById('start-date').value = dataDateRange.start;
        document.getElementById('end-date').value = dataDateRange.end;
        generateReport();
    }
}
