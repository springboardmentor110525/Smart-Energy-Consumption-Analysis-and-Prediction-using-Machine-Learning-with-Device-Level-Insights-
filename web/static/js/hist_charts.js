const dashboardDiv = document.getElementById("dashboardContainer");
if (dashboardDiv) {
  const dashboardData = JSON.parse(
    dashboardDiv.getAttribute("data-dashboard")
  );
  renderDashboard(dashboardData);
}

function renderKPI(totalEnergy, peakLoad) {
  const container = document.getElementById("kpiCards");
  if (!container) return;

  container.innerHTML = `
    <div class="card flex-fill" style="background: transparent">
      <div class="card-body">
        <h6>Total Energy</h6>
        <p>${totalEnergy.toFixed(2)} kWh</p>
      </div>
    </div>
    <div class="card flex-fill" style="background: transparent">
      <div class="card-body">
        <h6>Peak Load</h6>
        <p>${peakLoad.toFixed(2)} kWh</p>
      </div>
    </div>
  `;
}

function renderPie(pieData) {
  const labels = Object.keys(pieData);
  const values = Object.values(pieData);
  const data = [{
    type: 'pie',
    labels: labels,
    values: values,
    textinfo: 'label+percent',
    hoverinfo: 'label+value+percent'
  }];
  Plotly.newPlot('pieChart', data, { responsive: true, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)" });
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

function renderTopDevices(topDevices) {
  Plotly.newPlot("top-devices", [{
    x: Object.keys(topDevices),
    y: Object.values(topDevices),
    type: "bar",
    text: Object.values(topDevices).map(v => v.toFixed(2)),
    textposition: "auto"
  }], {
    title: "Top Energy Consuming Devices",
    xaxis: { title: "Device" },
    yaxis: { title: "Energy (kWh)" },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)"
  });
}

function renderDashboard(dashboard) {
  renderKPI(
    dashboard.kpis.total_energy,
    dashboard.kpis.peak
  );
  renderPie(dashboard.devices.pie);
  renderLineChart(dashboard.series, dashboard.metadata.range, "lineChartDaily");
  console.log(dashboard.metadata.range)
  if (dashboard.metadata.range === "month_range") {
    renderWeeklyChart(dashboard.series);
  }
  renderTopDevices(dashboard.devices.top);
}
