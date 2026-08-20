# backend/alerts.py

ALERT_TEMPLATES = {
    "Red": {
        "en": "EMERGENCY: Critical flood risk in {zone} ({depth}m expected within {lead_time}h). Avoid travel. Safe center: {shelter}.",
        "ta": "அவசர எச்சரிக்கை: {zone} பகுதியில் அதிக வெள்ள அபாயம் ({lead_time} மணி நேரத்தில் {depth}m நீர் மட்டம்). பாதுகாப்பான இடம்: {shelter}."
    },
    "Orange": {
        "en": "WARNING: Moderate waterlogging in {zone} ({depth}m expected within {lead_time}h). Low-lying roads impassable.",
        "ta": "எச்சரிக்கை: {zone} பகுதியில் மிதமான நீர் தேக்கம் ({lead_time} மணி நேரத்தில் {depth}m). தாழ்வான சாலைகளை தவிர்க்கவும்."
    },
    "Green": {
        "en": "ADVISORY: Normal conditions in {zone}. Monitoring river discharge.",
        "ta": "அறிவிப்பு: {zone} பகுதியில் நிலைமை சீராக உள்ளது. நீர்நிலைகள் கண்காணிக்கப்படுகின்றன."
    }
}

def generate_localized_alert(
    zone_name: str, 
    risk_level: str, 
    depth_m: float, 
    lead_time: int = 72, 
    shelter: str = "Anna University Shelter (Guindy)"
) -> dict:
    """Generates localized warning text in English and Tamil."""
    template = ALERT_TEMPLATES.get(risk_level, ALERT_TEMPLATES["Green"])
    return {
        "en": template["en"].format(zone=zone_name, depth=round(depth_m, 2), lead_time=lead_time, shelter=shelter),
        "ta": template["ta"].format(zone=zone_name, depth=round(depth_m, 2), lead_time=lead_time, shelter=shelter)
    }