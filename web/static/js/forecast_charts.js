const dashboardDiv = document.getElementById("dashboardContainer");
if (dashboardDiv) {
  const dashboardData = JSON.parse(dashboardDiv.getAttribute("data-dashboard"));
  renderDashboard(dashboardData);
}

function renderKPI(total, perDevice) {
  const kpiContainer = document.getElementById("kpiCards");
  if (!kpiContainer) return;

  // Clear existing cards
  kpiContainer.innerHTML = "";

  // Per-device cards
  for (const device in perDevice) {
    const card = document.createElement("div");
    card.className = "card flex-fill";
    card.style.backgroundColor = "transparent";
    card.innerHTML = `
      <div class="card-body" style="font-size: 16px">
        <h6 class="card-title">${device.charAt(0).toUpperCase() + device.slice(1)}</h6>
        <p class="card-text">${perDevice[device].toFixed(2)} kWh</p>
      </div>
    `;
    kpiContainer.appendChild(card);
  }
}

// Donut chart for device-wise energy split
function renderDonut(perDevice, totalEnergy) {
  const labels = Object.keys(perDevice);
  const values = Object.values(perDevice);

  Plotly.newPlot("donutChart", [{
    type: "pie",
    labels: labels,
    values: values,
    hole: 0.5,
    textinfo: "label+percent",  
    hoverinfo: "label+value"
  }], {
    title: "Energy Distribution by Device",
    showlegend: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    annotations: [
    {
      text: `${totalEnergy.toFixed(2)} kWh`,
      font: {
        size: 10,
        color: "black"
      },
      showarrow: false,
      x: 0.5,
      y: 0.5
    }
  ]
  });
}

// Line chart for daily/hourly
function renderLineChart(series, range, containerId) {
  const traces = [];
  for (const device in series) {
    const modeData =
      range === "day_range" ? series[device].hourly :
      series[device].daily;

    traces.push({
      x: [...Array(modeData.length).keys()],
      y: modeData,
      mode: "lines+markers",
      name: device
    });
  }

  Plotly.newPlot(containerId, traces, {
    title: `Daily Energy Forecast`,
    xaxis: { title: "Time" },
    yaxis: { title: "Energy (kWh)" },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)"
  });
}

// Weekly chart for 30-day monthly
function renderWeeklyChart(series) {
  const weeklyTraces = [];
  for (const device in series) {
    weeklyTraces.push({
      x: ["Week 1", "Week 2", "Week 3", "Week 4"],
      y: series[device].weekly,
      mode: "lines+markers",
      name: device,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)"
    });
  }

  Plotly.newPlot("weeklyChart", weeklyTraces, {
    title: "Weekly Energy Forecast (Next 30 Days)",
    yaxis: { title: "Energy (kWh)" },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)"
  });
}

function renderDashboard(dashboard) {
  renderKPI(dashboard.kpis.total_energy, dashboard.kpis.per_device);
  renderDonut(dashboard.kpis.per_device, dashboard.kpis.total_energy);
  renderLineChart(dashboard.series, dashboard.metadata.range, "lineChartDaily");

  if (dashboard.metadata.range === "month_range") {
    renderWeeklyChart(dashboard.series);
  }
}

renderDashboard(dashboard);
