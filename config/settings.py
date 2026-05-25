"""
Delidel WhatsApp Bot — Configuration
All secrets should come from environment variables in production.
"""

import os

# ── LLM ──────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = "llama-3.1-8b-instant"          # fast + cheap
GROQ_MODEL_ADV = "llama-3.3-70b-versatile"       # for order parsing if needed

# ── PrestaShop / Delidel API ─────────────────────────────────────────
PRESTA_BASE_URL    = os.getenv("PRESTA_BASE_URL", "https://uae.delidel.in")
PRESTA_API_KEY     = os.getenv("PRESTA_API_KEY",  "your_prestashop_key")
CHATBOT_API_URL    = f"{PRESTA_BASE_URL}/module/ogachatbotapi/ogachatbotapi"
PRESTA_REST_URL    = f"{PRESTA_BASE_URL}/api"     # native PrestaShop REST

# ── OGA CRM ──────────────────────────────────────────────────────────
OGA_CRM_BASE_URL      = os.getenv("OGA_CRM_BASE_URL","https://crm.ogaapps.in")
OGA_CRM_BEARER_TOKEN  = os.getenv("OGA_CRM_BEARER_TOKEN", "")
OGA_CRM_INSTANCE_NAME = os.getenv("OGA_CRM_INSTANCE_NAME","Delidel Support")

# ── WhatsApp (fallback direct API) ───────────────────────────────────
WHATSAPP_TOKEN   = os.getenv("WHATSAPP_TOKEN",   "")
PHONE_NUMBER_ID  = os.getenv("PHONE_NUMBER_ID",  "")
VERIFY_TOKEN     = os.getenv("VERIFY_TOKEN",     "")

# ── Redis Session Store ───────────────────────────────────────────────
REDIS_URL   = os.getenv("REDIS_URL",   "https://true-giraffe-105851.upstash.io")
REDIS_TOKEN = os.getenv("REDIS_TOKEN", "")

# ── Auth ──────────────────────────────────────────────────────────────
ALLOWED_NUMBERS = os.getenv("ALLOWED_NUMBERS", "9354906215,9759145356,7988149282").split(",")
BOT_PHONE       = os.getenv("BOT_PHONE", "9759145356")

# ── Session / Cart Expiration ─────────────────────────────────────────
CART_EXPIRATION_MINUTES = int(os.getenv("CART_EXPIRATION_MINUTES", "10"))

# ── Bot Persona ───────────────────────────────────────────────────────
BOT_NAME       = "Deli"
STORE_NAME     = "Delidel"
STORE_CURRENCY = "AED"
STORE_TIMEZONE = "Asia/Dubai"
