async function predictEnergy() {
    const current_energy = document.getElementById("value").value;
    const device = document.getElementById("device").value;
    const time_feature = document.getElementById("time_feature").value;

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ current_energy, device, time_feature })
    });

    const data = await response.json();
    const resultsDiv = document.getElementById("results");

    if (data.error) {
        resultsDiv.innerHTML = `<p>⚠️ ${data.error}</p>`;
        return;
    }

    resultsDiv.innerHTML = `
        <p>🔮 Predicted ${time_feature} usage for ${device}: ${data.prediction_kWh} kWh</p>
        <p>💡 ${data.smart_tip}</p>
    `;

    // Render trend chart
    const ctx = document.getElementById("chart-trend").getContext("2d");
    const labels = data.trend.map(t => t.time);
    const values = data.trend.map(t => t.value);

    if (window.energyChart) window.energyChart.destroy();

    window.energyChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Energy Usage (kWh)",
                data: values,
                fill: true,
                borderColor: "#3498db",
                backgroundColor: "rgba(52,152,219,0.2)",
                tension: 0.2
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: true } }
        }
    });
}
