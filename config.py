"""
Central configuration for Delidel WhatsApp Bot
"""
import os

# ─── LLM ───────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "gsk_your_key_here")
GROQ_MODEL        = "llama-3.1-8b-instant"   # fast & cheap
GROQ_SLOW_MODEL   = "llama-3.3-70b-versatile"  # for complex extraction

# ─── PrestaShop ────────────────────────────────────────────────────────────
PS_BASE_URL       = os.getenv("PS_BASE_URL", "https://stguae.delidel.in")
PS_API_KEY        = os.getenv("PS_API_KEY", "your_prestashop_key")
PS_LANG_ID        = 1    # English
PS_CURRENCY_ID    = 3    # AED
PS_SHOP_ID        = 1
PS_DEFAULT_ADDRESS_ID = None   # populated per customer

# ─── OGA CRM ───────────────────────────────────────────────────────────────
OGA_CRM_BASE_URL      = os.getenv("OGA_CRM_BASE_URL", "https://crm.ogaapps.in")
OGA_CRM_BEARER_TOKEN  = os.getenv("OGA_CRM_BEARER_TOKEN", "")
OGA_CRM_INSTANCE_NAME = "Delidel Support"

# ─── Redis (Session) ───────────────────────────────────────────────────────
REDIS_URL   = os.getenv("REDIS_URL", "https://true-giraffe-105851.upstash.io")
REDIS_TOKEN = os.getenv("REDIS_TOKEN", "your_token")
SESSION_TTL = 86400   # 24 hours
MAX_HISTORY = 8       # turns kept per session

# ─── WhatsApp ──────────────────────────────────────────────────────────────
PHONE_NUMBER_ID  = os.getenv("PHONE_NUMBER_ID", "")
WHATSAPP_TOKEN   = os.getenv("WHATSAPP_TOKEN", "")
VERIFY_TOKEN     = os.getenv("VERIFY_TOKEN", "")

# ─── Access Control ────────────────────────────────────────────────────────
ALLOWED_NUMBERS  = os.getenv("ALLOWED_NUMBERS", "9354906215,9759145356,7988149282").split(",")

# ─── App ───────────────────────────────────────────────────────────────────
DEFAULT_PHONE    = "9759145356"
PORT             = int(os.getenv("PORT", 10000))
