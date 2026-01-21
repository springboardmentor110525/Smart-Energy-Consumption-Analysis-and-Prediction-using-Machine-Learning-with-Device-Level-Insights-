function loadHourlyChart() {
    fetch("/api/hourly")
        .then(res => res.json())
        .then(data => {
            new Chart(document.getElementById("hourlyChart"), {
                type: "line",
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: "Hourly Energy (kWh)",
                        data: data.values,
                        borderWidth: 2
                    }]
                }
            });
        });
}

function loadDeviceChart() {
    fetch("/api/device")
        .then(res => res.json())
        .then(data => {
            new Chart(document.getElementById("deviceChart"), {
                type: "pie",
                data: {
                    labels: data.labels,
                    datasets: [{ data: data.values }]
                }
            });
        });
}
