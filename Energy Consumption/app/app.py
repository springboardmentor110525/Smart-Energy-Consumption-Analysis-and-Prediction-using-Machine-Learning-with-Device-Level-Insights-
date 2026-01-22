from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    mode = None

    if request.method == "POST":
        value = float(request.form["value"])
        mode = request.form["mode"]

        if mode == "hourly":
            prediction = round(value * 1.05, 3)
        elif mode == "daily":
            prediction = round(value * 24, 3)
        elif mode == "weekly":
            prediction = round(value * 24 * 7, 3)
        elif mode == "monthly":
            prediction = round(value * 24 * 30, 3)
        elif mode == "yearly":
            prediction = round(value * 24 * 365, 3)

    return render_template("index.html", prediction=prediction, mode=mode)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
