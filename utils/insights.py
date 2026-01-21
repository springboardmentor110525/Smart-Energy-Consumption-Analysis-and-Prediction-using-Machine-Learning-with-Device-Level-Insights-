import numpy as np
from datetime import datetime

def generate_tip(predictions, device):
    avg = np.mean(predictions)
    day = datetime.today().weekday()

    if device == "AC":
        if avg > 5:
            return "🌡 High AC usage detected. Setting temperature to 24–26°C can reduce energy by up to 20%."
        return "❄ Use inverter mode and clean filters monthly to save energy."

    if device == "TV":
        if day >= 5:
            return "📺 TV usage is usually higher on weekends. Avoid long idle screen time."
        return "🔌 Turning off at the socket avoids standby power loss."

    if device == "Fridge":
        return "🧊 Reducing fridge temperature by 1°C can save up to 5% electricity."

    if device == "Washer":
        return "🧺 Washing full loads with cold water saves both energy and water."

    if device == "Light":
        return "💡 Replacing bulbs with LEDs reduces lighting energy by nearly 80%."

    return "⚡ Monitor usage trends to avoid peak-hour consumption."
