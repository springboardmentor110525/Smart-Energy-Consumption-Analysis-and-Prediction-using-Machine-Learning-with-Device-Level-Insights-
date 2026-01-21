document.getElementById("energyForm").addEventListener("submit", function (e) {
    e.preventDefault();

    const data = {
        ac: document.getElementById("ac").value,
        fan: document.getElementById("fan").value,
        fridge: document.getElementById("fridge").value,
        tv: document.getElementById("tv").value,
        wm: document.getElementById("wm").value
    };

    fetch("/analyze", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(result => {
        document.getElementById("results").style.display = "block";

        document.getElementById("highDevice").innerText =
            `${result.max_device} (${result.devices[result.max_device]} W)`;

        document.getElementById("lowDevice").innerText =
            `${result.min_device} (${result.devices[result.min_device]} W)`;

        let table = "";
        for (let d in result.devices) {
            table += `<tr><td>${d}</td><td>${result.devices[d]}</td></tr>`;
        }
        document.getElementById("deviceTable").innerHTML = table;

        document.getElementById("tips").innerHTML =
            result.tips.map(t => `<p>${t}</p>`).join("");
    });
});
