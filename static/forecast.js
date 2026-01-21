// ==========================
// HELPERS
// ==========================
function formatFullDateTime(isoStr) {
    const d = new Date(isoStr);

    const date = d.toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "short",
        year: "numeric"
    });

    const time = d.toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false
    });

    return `${date} ${time}`;
}

function formatShortDate(isoStr) {
    const d = new Date(isoStr);
    return d.toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "short"
    });
}

function getWeekdayName(isoStr) {
    return new Date(isoStr).toLocaleDateString("en-US", { weekday: "short" });
}

function getMonthName(isoStr) {
    return new Date(isoStr).toLocaleDateString("en-US", { month: "short" });
}

// Soft pastel palette
const COLOR_PALETTE = [
    "#8E9AAF", "#FF9F68", "#6EC6A8",
    "#F2A1C2", "#A3D65C", "#FFD84D",
    "#BDBDBD", "#A8DADC", "#CDB4DB"
];

// ==========================
// MAIN PREDICT FUNCTION
// ==========================
function predictEnergy() {

    const energy = document.getElementById("energy").value;
    const device = document.getElementById("device").value;
    const horizon = document.getElementById("horizon").value;

    if (!energy || !device || !horizon) {
        alert("Please fill all fields");
        return;
    }

    document.getElementById("result-box").classList.remove("hidden");
    document.getElementById("prediction-list").innerHTML =
        "<p>⏳ Generating prediction...</p>";

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ energy, device, horizon })
    })
    .then(res => {
        if (!res.ok) throw new Error("Prediction failed");
        return res.json();
    })
    .then(data => renderData(data, horizon))
    .catch(err => {
        console.error(err);
        alert("Prediction failed");
    });
}

// ==========================
// RENDER DATA
// ==========================
function renderData(data, horizon) {

    const dates = data.dates;
    const values = data.predictions;

    // --------------------------
    // TEXT OUTPUT (ONCE ONLY)
    // --------------------------
    let html = "";
    dates.forEach((d, i) => {
        html += `<p>${formatFullDateTime(d)} : <b>${values[i].toFixed(3)} kWh</b></p>`;
    });
    document.getElementById("prediction-list").innerHTML = html;

    // --------------------------
    // SMART TIP
    // --------------------------
    document.getElementById("tip").innerHTML = data.tip;

    // --------------------------
    // X AXIS LABELS
    // --------------------------
    const labels = dates.map(d => formatShortDate(d));

    // --------------------------
    // Y AXIS RANGE
    // --------------------------
    const min = Math.min(...values);
    const max = Math.max(...values);
    const pad = (max - min) * 0.6 || 0.002;

    // ==========================
    // LINE CHART (ALWAYS)
    // ==========================
    Plotly.react("lineChart", [{
        x: labels,
        y: values,
        type: "scatter",
        mode: "lines+markers",
        line: { color: "#6D597A", width: 3 },
        marker: { size: 7 },
        hovertemplate: "%{y:.3f} kWh<extra></extra>"
    }], {
        title: "📈 Energy Trend",
        yaxis: {
            title: "kWh",
            range: [min - pad, max + pad],
            gridcolor: "#E5E7EB"
        },
        paper_bgcolor: "#F8FAFC",
        plot_bgcolor: "#F8FAFC"
    });

    // ==========================
    // BAR CHART (WEEK / MONTH)
    // ==========================
    if (horizon === "week" || horizon === "month") {

        Plotly.react("barChart", [{
            x: labels,
            y: values,
            type: "bar",
            text: values.map(v => v.toFixed(3)),
            textposition: "outside",
            marker: {
                color: labels.map((_, i) =>
                    COLOR_PALETTE[i % COLOR_PALETTE.length]
                )
            }
        }], {
            title: "📊 Consumption Comparison",
            yaxis: {
                title: "kWh",
                range: [min - pad, max + pad],
                gridcolor: "#E5E7EB"
            },
            xaxis: { tickangle: -30 },
            paper_bgcolor: "#F8FAFC",
            plot_bgcolor: "#F8FAFC"
        });

    } else {
        Plotly.purge("barChart");
    }

    // ==========================
    // PIE CHART (WEEK / MONTH)
    // ==========================
    if (horizon === "week") {

        Plotly.react("pieChart", [{
            labels: dates.map(getWeekdayName),
            values: values,
            type: "pie",
            hole: 0.5,
            marker: { colors: COLOR_PALETTE },
            textinfo: "percent"
        }], {
            title: "🔄 Weekly Energy Distribution",
            paper_bgcolor: "#F8FAFC"
        });

    } else if (horizon === "month") {

        const monthlyTotals = {};
        dates.forEach((d, i) => {
            const m = getMonthName(d);
            monthlyTotals[m] = (monthlyTotals[m] || 0) + values[i];
        });

        Plotly.react("pieChart", [{
            labels: Object.keys(monthlyTotals),
            values: Object.values(monthlyTotals),
            type: "pie",
            hole: 0.5,
            marker: { colors: COLOR_PALETTE },
            textinfo: "percent"
        }], {
            title: "🔄 Monthly Energy Distribution",
            paper_bgcolor: "#F8FAFC"
        });

    } else {
        Plotly.purge("pieChart");
    }
}
