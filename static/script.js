document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("predictBtn");

    btn.addEventListener("click", async () => {
        const input = document.getElementById("energyValues").value.trim();

        if (!input) {
            alert("Please enter 24 hourly values");
            return;
        }

        // Convert input string to array of numbers
        const values = input.split(",").map(v => parseFloat(v.trim()));

        if (values.length !== 24 || values.some(isNaN)) {
            alert("Please enter EXACTLY 24 numeric values separated by commas");
            return;
        }

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ values })
            });

            const data = await response.json();

            if (data.error) {
                alert(data.error);
                return;
            }

            // Show result
            document.getElementById("resultBox").innerHTML = `
                <h3>🔮 Predicted Energy Usage</h3>
                <p><strong>${data.predicted_units} kWh</strong></p>
            `;
            document.getElementById("resultBox").style.display = "block";

        } catch (err) {
            console.error(err);
            alert("Prediction failed. Check console.");
        }
    });
});
