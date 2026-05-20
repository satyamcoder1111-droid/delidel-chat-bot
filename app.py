"""
Delidel WhatsApp Bot — Flask Application
Production-ready with CRM passthrough, auth, and clean routing.
"""

import json
import re
import requests
from flask import Flask, request, jsonify, render_template

from config.settings import (
    ALLOWED_NUMBERS, BOT_PHONE,
    OGA_CRM_BASE_URL, OGA_CRM_BEARER_TOKEN, OGA_CRM_INSTANCE_NAME,
    VERIFY_TOKEN,
)
from chains.orchestrator import process_message

app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def clean_number(raw: str) -> str:
    return re.sub(r"^\+?(91|971)", "", str(raw))


def is_allowed_number(raw: str) -> bool:
    cleaned = clean_number(raw)
    print(f"[AUTH] raw={raw} -> cleaned={cleaned}")
    return cleaned in ALLOWED_NUMBERS


def forward_raw_to_crm(raw_body: dict):
    """Mirror n8n 'HTTP Request' — forward Evolution event to CRM."""
    url     = f"{OGA_CRM_BASE_URL}/api/webhook/evolution"
    payload = raw_body.get("body", raw_body)
    try:
        res = requests.post(url, json=payload,
                            headers={"Content-Type": "application/json"}, timeout=10)
        print(f"[CRM PASSTHROUGH] {res.status_code}")
    except Exception as e:
        print(f"[CRM PASSTHROUGH ERROR] {e}")


def send_reply_via_crm(sender_number: str, message: str):
    """Mirror n8n 'Send Reply via OGA CRM'."""
    url      = f"{OGA_CRM_BASE_URL}/api/v1/send"
    clean_to = str(sender_number).lstrip("+").strip()
    payload  = {
        "instanceName": OGA_CRM_INSTANCE_NAME,
        "number":       clean_to,
        "type":         "text",
        "message":      str(message).strip(),
    }
    headers = {
        "Authorization": f"Bearer {OGA_CRM_BEARER_TOKEN}",
        "Content-Type":  "application/json",
    }
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        print(f"[CRM SEND] {res.status_code}")
    except Exception as e:
        print(f"[CRM SEND ERROR] {e}")


def transform_evolution_to_standard(data: dict) -> dict:
    """
    Evolution API sends { data: { key: { remoteJid }, message: { conversation } } }.
    Transform to the standard WhatsApp Cloud API envelope so the rest of the code
    can use a single path.
    """
    raw    = data.get("data", {})
    sender = raw.get("key", {}).get("remoteJid", "").replace("@s.whatsapp.net", "")
    text   = (
        raw.get("message", {}).get("conversation")
        or raw.get("message", {}).get("extendedTextMessage", {}).get("text")
        or ""
    )
    return {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": sender,
                        "type": "text",
                        "text": {"body": text},
                    }]
                }
            }]
        }]
    }


# ─────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return "Delidel WhatsApp Bot ✅", 200


@app.route("/ui")
def web_ui():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "delidel-bot"}), 200


# ── Web Chat (for testing UI) ─────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    body          = request.get_json(force=True)
    user_input    = body.get("message", "").strip()
    sender_number = body.get("sender_number", BOT_PHONE)

    # if not is_allowed_number(sender_number):
    #     return jsonify({"reply": "Unauthorized"}), 403
    # if not user_input:
    #     return jsonify({"reply": ""}), 200

    try:
        reply = process_message(user_input, sender_number)
    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        reply = "Something went wrong. Please try again. 🙏"

    return jsonify({"reply": reply})


# ── WhatsApp Webhook — Verify (GET) ──────────────────────────────────

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode      = request.args.get("hub.mode")
    token     = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("[WEBHOOK] Verified!")
        return challenge, 200
    return "Forbidden", 403


# ── WhatsApp Webhook — Receive (POST) ────────────────────────────────

@app.route("/webhook", methods=["POST"])
def receive_webhook():
    data = request.get_json(force=True)
    print("[WEBHOOK IN]", json.dumps(data, indent=2)[:500])

    # 1. Forward to CRM immediately (fire-and-forget)
    forward_raw_to_crm(data)

    # 2. Transform payload to standard format
    data = transform_evolution_to_standard(data)

    try:
        entry   = data["entry"][0]
        changes = entry["changes"][0]["value"]

        if "messages" not in changes:
            return "OK", 200

        msg    = changes["messages"][0]
        sender = msg["from"]

        if msg.get("type") != "text":
            return "OK", 200

        user_input = msg["text"]["body"].strip()
        print(f"[MSG] {sender}: {user_input}")

        if not is_allowed_number(sender):
            print(f"[BLOCKED] {sender}")
            return "OK", 200

        # 3. Generate reply
        reply = process_message(user_input, sender)

        # 4. Send via CRM
        # send_reply_via_crm(sender, reply)

    except Exception as e:
        print(f"[WEBHOOK ERROR] {e}")

    return "OK", 200


# ─────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
