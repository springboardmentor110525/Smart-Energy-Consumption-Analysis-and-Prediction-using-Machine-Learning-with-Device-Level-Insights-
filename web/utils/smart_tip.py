import random

SMART_TIPS = {
    "fridge_high": {
        "title": "Fridge Energy Spike",
        "messages": [
            "Your fridge is consuming unusually high energy. Check door seals and avoid frequent opening.",
            "Fridge load looks heavy — reducing temperature by 1°C can save noticeable energy."
        ]
    },
    "tv_high": {
        "title": "TV Usage Alert",
        "messages": [
            "TV usage is high. Switching off standby mode can reduce phantom power draw.",
            "Long TV hours detected — consider enabling auto-sleep settings."
        ]
    },
    "ac_high": {
        "title": "AC Usage Alert",
        "messages": [
            "AC is the major energy driver. Setting it to 24–26°C can cut consumption by ~10–15%."
        ]
    },
    "overall_high": {
        "title": "High Energy Consumption",
        "messages": [
            "Overall energy usage is high for the selected duration. Energy-heavy devices dominate the load."
        ]
    },
    "efficient": {
        "title": "Efficient Energy Use",
        "messages": [
            "Nice. Your energy usage looks efficient for the selected devices."
        ]
    }
}

def generate_smart_tip(predicted_kwh):
    total = sum(predicted_kwh.values())
    if total == 0:
        return None

    dominant_device = max(predicted_kwh, key=predicted_kwh.get)
    dominant_value = predicted_kwh[dominant_device]
    dominance_ratio = dominant_value / total

    # thresholds (tune later)
    HIGH_DEVICE_KWH = 25
    DOMINANT_RATIO = 0.5
    HIGH_TOTAL_KWH = 45

    if dominant_value >= HIGH_DEVICE_KWH and dominance_ratio >= DOMINANT_RATIO:
        tip_key = f"{dominant_device}_high"
    elif total >= HIGH_TOTAL_KWH:
        tip_key = "overall_high"
    else:
        tip_key = "efficient"

    tip = SMART_TIPS.get(tip_key)
    if not tip:
        return None

    return {
        "title": tip["title"],
        "message": random.choice(tip["messages"])
    }
