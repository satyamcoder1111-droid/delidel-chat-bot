"""
Core chain orchestrator.
No agents — pure deterministic chain dispatch based on classified intent.

Flow:
  User Message
      ↓
  [Intent Chain]   → classify intent (Groq LLM)
      ↓
  Dispatch Router  → pick the right handler
      ↓
  Handler          → call tools → format reply (Groq LLM)
      ↓
  Reply String
"""

import json
import re
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from config.settings import GROQ_API_KEY, GROQ_MODEL, CART_EXPIRATION_MINUTES
from config.session import (
    get_session, save_session, append_history,
    clear_cart, set_stage, clean_number,
)
from chains.prompts import (
    INTENT_PROMPT, ORDER_PARSER_PROMPT, GREETING_PROMPT,
    PRODUCT_REPLY_PROMPT, CART_SUMMARY_PROMPT,
    ORDER_CONFIRM_PROMPT, GENERAL_REPLY_PROMPT,
)
from tools.presta_tools import (
    search_products, validate_cart,
    place_order_via_module, get_order_status,
)


# ─────────────────────────────────────────────────────────────────────
# LLM  (shared across all chains)
# ─────────────────────────────────────────────────────────────────────

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0,
    max_tokens=600,
)

str_parser = StrOutputParser()


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Strip markdown fences and extract first JSON object or array."""
    text = re.sub(r"`{3}[a-z]*", "", text).strip()
    # Prefer a top-level object first so nested arrays inside valid JSON
    # responses do not get extracted on their own.
    match = re.search(r'\{[\s\S]*\}', text) or re.search(r'\[[\s\S]*\]', text)
    return match.group(0) if match else text


def _safe_json(text: str, fallback):
    try:
        return json.loads(_extract_json(text))
    except Exception:
        return fallback


def _is_yes_confirmation(text: str) -> bool:
    text_clean = re.sub(r"[^\w\s]", "", text.lower().strip())
    tokens = text_clean.split()
    if not tokens:
        return False
    
    YES_WORDS = {
        "yes", "y", "ok", "okay", "confirm", "haan", "ha", "theek hai", 
        "sure", "go ahead", "yas", "yess", "yesss", "yep", "yeah", "yup", "ys",
        "done", "correct", "perfect", "thik", "theek", "thk", "sahi", "haji",
        "ji", "han", "yuss", "yus"
    }
    
    first_token = tokens[0]
    if first_token in YES_WORDS:
        return True
        
    for yes_phrase in (
        "go ahead", "theek hai", "thik hai", "thk hai", "ok hai", 
        "sahi hai", "ji haan", "ji han", "yes please", "yes of course"
    ):
        if text_clean.startswith(yes_phrase):
            return True
            
    return False


def _is_no_discard(text: str) -> bool:
    text_clean = re.sub(r"[^\w\s]", "", text.lower().strip())
    tokens = text_clean.split()
    if not tokens:
        return False
        
    NO_WORDS = {
        "no", "nahi", "nhi", "cancel", "nevermind", "nope", "band kar", 
        "nah", "na", "naa", "discard", "stop", "band"
    }
    
    first_token = tokens[0]
    if first_token in NO_WORDS:
        # A token like "no", "nahi", "nhi", "na", "naa", "nah", "nope" is a discard word,
        # but if there are other tokens, we only count it as a discard if it is part of a 
        # known pure negative phrase. Otherwise, it could be a correction/update (e.g., "no fries 9mm").
        if first_token in {"no", "nahi", "nhi", "na", "naa", "nah", "nope"}:
            if len(tokens) == 1:
                return True
            pure_no_phrases = {
                "no thanks", "no thank you", "dont want", "don't want", 
                "cancel order", "cancel it", "nahi chahiye", "nhi chahiye", 
                "not now", "no need", "no please", "no no", "nahi nahi", "nhi nhi",
                "never", "never mind", "discard", "discard it", "no cancel", "no cancel order",
                "cancel this order", "cancel my order", "please cancel", "pls cancel"
            }
            return any(text_clean.startswith(phrase) for phrase in pure_no_phrases) or text_clean in pure_no_phrases
        return True
        
    for no_phrase in (
        "band kar", "never mind", "no thanks", "dont want", "don't want", 
        "cancel order", "cancel it", "nahi chahiye", "nhi chahiye", 
        "not now", "no need", "no no", "nahi nahi", "nhi nhi", "cancel this order",
        "cancel my order", "please cancel", "pls cancel"
    ):
        if text_clean.startswith(no_phrase):
            return True
            
    return False


def _extract_negated_replacement(text: str) -> str:
    """
    If the message starts with a negative/discard word (e.g. 'no', 'nahi', 'nhi', 'cancel')
    followed by other words, and it is NOT classified as a pure discard,
    returns the remaining text. Otherwise returns ''.
    """
    text_clean = text.strip()
    match = re.match(r"^(no|nahi|nhi|na|naa|nah|nope|cancel)[\s,.:;!_-]+(?P<rest>.+)$", text_clean, re.IGNORECASE)
    if match:
        rest = match.group("rest").strip()
        if rest and re.search(r"[a-zA-Z0-9]", rest):
            return rest
    return ""



def _normalize_order_product_name(name: str) -> str:
    """Apply a few lightweight normalizations before API validation."""
    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    cleaned = re.sub(
        r"^(add|need|want|send|give me|i need|i want|please add)\s+",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\s+(also|too|pls|please)$", "", cleaned)
    replacements = {
        "frys": "fries",
        "frice": "fries",
        "frize": "fries",
        "chikken": "chicken",
        "chk": "chicken",
        "nagast": "nuggets",
        "naggets": "nuggets",
        "griller": "grillers",
    }
    for source, target in replacements.items():
        cleaned = re.sub(rf"\b{re.escape(source)}\b", target, cleaned)
    synonym_map = {
        "fries": "french fries",
        "french fries": "french fries",
        "small fries": "small french fries",
        "small french fries": "small french fries",
        "wing": "chicken wings",
        "wings": "chicken wings",
        "chicken wing": "chicken wings",
        "chicken wings": "chicken wings",
        "grillers": "grillers",
        "burger": "chicken burger",
        "chicken burger": "chicken burger",
        "nugget": "nuggets",
        "nuggets": "nuggets",
        "shawarma": "shawarma",
        "sausage": "sausage",
        "milk": "milk",
    }
    return synonym_map.get(cleaned, cleaned)


def _rule_based_order_parse(user_message: str) -> list[dict]:
    """
    Parse common WhatsApp order formats without relying on the LLM.
    Supports:
    - fries 2 ctn
    - wings 3 pack
    - product x2
    - product *2
    """
    items = []
    chunks = [c.strip() for c in re.split(r"[\n,]+", user_message) if c.strip()]

    pattern = re.compile(
        r"^(?P<name>.+?)\s+(?P<qty>\d+(?:\.\d+)?|1/2|half)\s*(?P<unit>ctn|ctns|carton|cartons|pack|packs|pcs|pieces|box|boxes)?$",
        re.IGNORECASE,
    )
    suffix_pattern = re.compile(
        r"^(?P<name>.+?)\s*[*xX]\s*(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>ctn|ctns|carton|cartons|pack|packs|pcs|pieces|box|boxes)?$",
        re.IGNORECASE,
    )

    for chunk in chunks:
        line = re.sub(r"\s+", " ", chunk.strip())
        match = pattern.match(line) or suffix_pattern.match(line)
        if not match:
            if len(chunks) > 1:
                normalized = _normalize_order_product_name(line)
                if normalized and re.search(r"[a-zA-Z]", normalized):
                    items.append({
                        "name": normalized,
                        "quantity": 1,
                        "unit": "ctn",
                    })
            continue

        qty_raw = match.group("qty").lower()
        qty = 0.5 if qty_raw in {"1/2", "half"} else float(qty_raw)
        unit_raw = (match.group("unit") or "ctn").lower()
        unit = "pack" if unit_raw in {"pack", "packs", "pcs", "pieces", "box", "boxes"} else "ctn"
        name = _normalize_order_product_name(match.group("name"))

        items.append({
            "name": name,
            "quantity": int(qty) if qty.is_integer() else qty,
            "unit": unit,
        })

    return items


def _pending_cart_add_parse(user_message: str) -> list[dict]:
    """Parse add/update messages without explicit quantity while a cart is pending."""
    text = re.sub(r"\s+", " ", user_message.strip().lower())
    patterns = [
        r"^(?:add|need|want)\s+(?P<name>.+)$",
        r"^i need\s+(?P<name>.+)$",
        r"^i want\s+(?P<name>.+)$",
        r"^please add\s+(?P<name>.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text)
        if not match:
            continue
        name = _normalize_order_product_name(match.group("name"))
        if not name:
            continue
        return [{"name": name, "quantity": 1, "unit": "ctn"}]
    return []


def _format_previous_cart(state: dict) -> str:
    items = state.get("previous_cart", []) or state.get("cart", [])
    total = state.get("previous_cart_total", 0.0) or state.get("cart_total", 0.0)

    if not items:
        return "Your previous cart is empty. Would you like to start shopping?"

    lines = ["Here are your previous cart details:"]
    for item in items:
        product_name = item.get("matched_product_name") or item.get("input_product") or "Item"
        qty = item.get("requested_qty", 1)
        pack_type = item.get("pack_type", "CTN")
        price = item.get("total_price", 0)
        lines.append(f"{product_name} — {qty} {pack_type} — AED {price}")
    lines.append("")
    lines.append(f"Total: AED {total}")
    return "\n".join(lines)


def _format_currency(value) -> str:
    amount = float(value or 0)
    text = f"{amount:.2f}"
    if text.endswith("00"):
        return text[:-3]
    if text.endswith("0"):
        return text[:-1]
    return text


def _format_cart_summary(cart_data: dict, customer_name: str) -> str:
    items = cart_data.get("data", [])
    cart_total = cart_data.get("cart_total", 0)

    if not items:
        return "Sorry, your cart looks empty. Please try again."

    lines = ["Here are the details:"]
    
    has_insufficient = False
    for item in items:
        if item.get("status") == "not_found":
            continue
        product_name = item.get("matched_product_name") or item.get("input_product") or "Item"
        qty = item.get("requested_qty") or item.get("requested_units") or 1
        pack_type = item.get("pack_type", "CTN")
        total_price = _format_currency(item.get("total_price", 0))
        
        is_error = item.get("status") == "error" or "insufficient" in str(item.get("message", "")).lower()
        if is_error:
            has_insufficient = True
            lines.append(f"{product_name} — {qty} {pack_type} — AED {total_price} ⚠️ (Insufficient Stock)")
        else:
            lines.append(f"{product_name} — {qty} {pack_type} — AED {total_price}")

    lines.append("")
    lines.append(f"Grand Total: AED {_format_currency(cart_total)}")
    lines.append("")

    if has_insufficient:
        lines.append("⚠️ *Note*: For items with insufficient stock, our team will call you.")
        lines.append("")

    closing = "Should I confirm this?"
    lines.append(closing)

    return "\n".join(lines)



def _format_confirm_order_reply(order_result: dict, customer_name: str, has_insufficient: bool = False) -> str:
    status = order_result.get("status")
    data = order_result.get("data", {})
    
    if status or data.get("id_order") or order_result.get("order_id"):
        ref = data.get("order_reference") or data.get("id_order") or order_result.get("reference") or order_result.get("order_id", "")
        invoice_link = data.get("invoice_link")
        
        ref_line = f"📦 Order Ref # *{ref}*\n" if ref else ""
        link_line = f"📄 Invoice: {invoice_link}\n" if invoice_link else ""
        
        note_line = ""
        if has_insufficient:
            note_line = "\n⚠️ *Note*: For items with insufficient stock, our team will call you.\n"
        
        return (
            f"Your order has been placed successfully. 🎉\n\n"
            f"{ref_line}"
            f"{link_line}"
            f"{note_line}\n"
            "Our team will process it shortly. 🚚\n"
            "Feel free to ask anything else! 😊"
        ).strip()

    message = order_result.get("message") or "Please contact our team or try again."
    return (
        f"Sorry, I couldn't confirm the order right now.\n"
        f"{message}"
    ).strip()


def _format_order_status_reply(api_response: dict) -> str:
    data = api_response.get("data", {})
    ref = data.get("reference", "")
    msg = data.get("message", "")
    return f"📦 Order {ref}\n{msg}"


def _extract_removed_product_name(user_message: str) -> str:
    """Detect simple line-item removal requests during cart confirmation."""
    text = re.sub(r"\s+", " ", user_message.strip().lower())
    patterns = [
        r"^i do not need (?P<name>.+)$",
        r"^i don't need (?P<name>.+)$",
        r"^dont need (?P<name>.+)$",
        r"^don't need (?P<name>.+)$",
        r"^remove (?P<name>.+)$",
        r"^delete (?P<name>.+)$",
        r"^no (?P<name>.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            return _normalize_order_product_name(match.group("name"))
    return ""


def _parsed_to_api_products(parsed_products: list[dict]) -> list[dict]:
    api_products = []
    for item in parsed_products:
        unit = item.get("unit", "ctn")
        pack_type = 1 if unit == "ctn" else 0
        
        # Check if quantity or name represents 1/2 or half
        qty_val = item.get("quantity", 1)
        name_val = item.get("name", "").lower()
        
        is_half = 0
        # Check quantity (float, integer, or string representations)
        if qty_val == 0.5 or str(qty_val).strip() in {"0.5", "1/2", "half"}:
            is_half = 1
        # Check name for "1/2" or "half"
        elif "1/2" in name_val or "half" in name_val:
            is_half = 1
            
        api_products.append({
            "product_name": name_val,
            "qty": qty_val,
            "pack_type": pack_type,
            "is_half_param": is_half,
        })
    return api_products


def _validate_and_format_cart(parsed_products: list[dict], sender: str, state: dict) -> tuple[str, dict]:
    api_products = _parsed_to_api_products(parsed_products)

    raw_cart = validate_cart.invoke({
        "sender_number": sender,
        "product_json": api_products,
    })
    print(f"\n[VALIDATE CART RAW RESPONSE] {raw_cart}\n")
    cart_data = _safe_json(raw_cart, {})

    items = cart_data.get("data", [])
    cart_total = cart_data.get("cart_total", 0)
    cart_session_id = cart_data.get("cart_session_id", "")

    state["cart"] = items
    state["cart_total"] = cart_total
    state["cart_session_id"] = cart_session_id
    state["stage"] = "awaiting_confirm"
    
    import time
    state["cart_updated_at"] = time.time()

    customer_name = cart_data.get("customer_name", state.get("customer_name", ""))
    state["customer_name"] = customer_name

    reply = _format_cart_summary(cart_data, customer_name)
    return reply, state


def _state_cart_to_parsed_products(state: dict) -> list[dict]:
    parsed_products = []
    for item in state.get("cart", []):
        name = item.get("input_product") or item.get("matched_product_name") or ""
        if not name:
            continue
        pack_type = str(item.get("pack_type", "CTN")).lower()
        unit = "pack" if pack_type == "pack" else "ctn"
        parsed_products.append({
            "name": _normalize_order_product_name(name),
            "quantity": item.get("requested_qty", 1),
            "unit": unit,
        })
    return parsed_products


def _match_product_name(target: str, existing_name: str) -> bool:
    target_norm = _normalize_order_product_name(target)
    existing_norm = _normalize_order_product_name(existing_name)
    if not target_norm or not existing_norm:
        return False

    # Extract any numeric values (sizes, quantities, weights) to prevent variant mismatches
    target_numbers = set(re.findall(r"\d+(?:\.\d+)?", target_norm))
    existing_numbers = set(re.findall(r"\d+(?:\.\d+)?", existing_norm))
    if target_numbers and not target_numbers.issubset(existing_numbers):
        return False

    if target_norm == existing_norm:
        return True
    if target_norm in existing_norm or existing_norm in target_norm:
        return True

    target_tokens = set(re.findall(r"[a-z]+", target_norm))
    existing_tokens = set(re.findall(r"[a-z]+", existing_norm))
    
    if not target_tokens or not existing_tokens:
        return False
        
    # Must be a subset in either direction (hierarchical matching) to prevent loose token sharing false positives
    return target_tokens.issubset(existing_tokens) or existing_tokens.issubset(target_tokens)



def _merge_cart_products(existing_products: list[dict], updated_products: list[dict]) -> list[dict]:
    merged = {}
    order = []

    for item in existing_products:
        key = _normalize_order_product_name(item.get("name", ""))
        if not key:
            continue
        merged[key] = {
            "name": key,
            "quantity": item.get("quantity", 1),
            "unit": item.get("unit", "ctn"),
            "original_name": item.get("name", ""),
        }
        order.append(key)

    for item in updated_products:
        target_name = _normalize_order_product_name(item.get("name", ""))
        if not target_name:
            continue
            
        matched_key = target_name
        for existing_key in order:
            if _match_product_name(target_name, merged[existing_key]["original_name"]):
                matched_key = existing_key
                break

        merged[matched_key] = {
            "name": merged.get(matched_key, {}).get("original_name", target_name),
            "quantity": item.get("quantity", 1),
            "unit": item.get("unit", "ctn"),
            "original_name": merged.get(matched_key, {}).get("original_name", target_name),
        }
        if matched_key not in order:
            order.append(matched_key)

    return [
        {"name": merged[key]["name"], "quantity": merged[key]["quantity"], "unit": merged[key]["unit"]}
        for key in order
    ]


def _parse_order_message(user_message: str) -> list[dict]:
    parsed_products = _rule_based_order_parse(user_message)
    if parsed_products:
        return parsed_products

    raw_parsed = parser_chain.invoke({"user_message": user_message})
    raw_parsed = raw_parsed if isinstance(raw_parsed, str) else raw_parsed.content
    print(f"\n[ORDER PARSER RAW RESPONSE] {raw_parsed}\n")
    parsed_products = _safe_json(raw_parsed, [])
    if isinstance(parsed_products, dict):
        parsed_products = [parsed_products]
    return parsed_products


def _parse_pending_cart_message(user_message: str) -> list[dict]:
    """Parse cart edits while awaiting confirmation."""
    parsed_products = _parse_order_message(user_message)
    if parsed_products:
        return parsed_products
    return _pending_cart_add_parse(user_message)


def _pending_products_from_intent(intent: dict) -> list[dict]:
    """Build a default cart item from intent context when user says 'order it'."""
    product_name = _normalize_order_product_name(intent.get("product_name", ""))
    if not product_name:
        return []
    unit = intent.get("unit", "ctn") or "ctn"
    quantity = intent.get("quantity", 1) or 1
    try:
        quantity = float(quantity)
    except Exception:
        quantity = 1
    if isinstance(quantity, float) and quantity.is_integer():
        quantity = int(quantity)
    return [{
        "name": product_name,
        "quantity": quantity,
        "unit": "pack" if str(unit).lower() == "pack" else "ctn",
    }]


def _reference_order_requested(user_message: str) -> bool:
    text = re.sub(r"\s+", " ", user_message.strip().lower())
    return text in {
        "order it",
        "order this",
        "order this also",
        "add this",
        "add this also",
        "send it",
        "send this",
    }


def _history_summary(state: dict, max_turns: int = 4) -> str:
    turns = state.get("history", [])[-max_turns:]
    if not turns:
        return "No prior conversation."
    return "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in turns)


def _multi_order_check(text: str) -> bool:
    """Heuristic: 2+ product lines with quantities → multi-order."""
    lines = text.strip().split("\n")
    count = 0
    for line in lines:
        line = line.lower().strip()
        if not line:
            continue
        if re.search(r"\*\d+", line):
            count += 1; continue
        if re.search(r"\b\d+(\.\d+)?\s*(ctn|ctns|box|pcs|kg|carton|cartons|pieces)\b", line):
            count += 1; continue
        if re.search(r"[a-zA-Z].*\d+|\d+.*[a-zA-Z]", line):
            count += 1
    return count >= 2


# ─────────────────────────────────────────────────────────────────────
# CHAIN BUILDERS
# ─────────────────────────────────────────────────────────────────────

def _build_chain(prompt, llm, parser=None):
    parser = parser or str_parser
    return prompt | llm | parser


intent_chain   = _build_chain(INTENT_PROMPT, llm)
parser_chain   = _build_chain(ORDER_PARSER_PROMPT, llm)
greeting_chain = _build_chain(GREETING_PROMPT, llm)
product_chain  = _build_chain(PRODUCT_REPLY_PROMPT, llm)
cart_chain     = _build_chain(CART_SUMMARY_PROMPT, llm)
confirm_chain  = _build_chain(ORDER_CONFIRM_PROMPT, llm)
general_chain  = _build_chain(GENERAL_REPLY_PROMPT, llm)


# ─────────────────────────────────────────────────────────────────────
# HANDLER:  GREETING
# ─────────────────────────────────────────────────────────────────────

def handle_greeting(state: dict, sender: str) -> str:
    reply = greeting_chain.invoke({"customer_name": ""})
    return reply if isinstance(reply, str) else reply.content


# ─────────────────────────────────────────────────────────────────────
# HANDLER:  PRICE / STOCK LOOKUP
# ─────────────────────────────────────────────────────────────────────

def handle_product_lookup(state: dict, sender: str, intent: dict) -> str:
    product_name = intent.get("product_name", "")
    if not product_name:
        return "Could you tell me which product you're looking for? 😊"

    raw = search_products.invoke({
        "product_name": product_name,
        "sender_number": sender,
    })
    data = _safe_json(raw, {})

    customer_name = data.get("customer_name", state.get("customer_name", ""))
    state["customer_name"] = customer_name
    products = data.get("products", [])
    top_product_name = products[0].get("name", "") if products else ""
    state["last_product"] = top_product_name or product_name
    state["last_lookup_product"] = top_product_name or product_name

    products_json = json.dumps(products)

    reply = product_chain.invoke({
        "customer_name": "",
        "search_query":  product_name,
        "intent":        intent.get("intent", "price_check"),
        "products_json": products_json,
    })
    return reply if isinstance(reply, str) else reply.content


# ─────────────────────────────────────────────────────────────────────
# HANDLER:  SINGLE / MULTI PRODUCT ORDER
# ─────────────────────────────────────────────────────────────────────

def _parse_and_validate(user_message: str, sender: str, state: dict) -> tuple[str, dict]:
    """
    Parse products from message, validate cart, format summary.
    Returns (reply_text, updated_state).
    """
    # Step 1 — parse products with rule-based fallback first, then LLM
    parsed_products = _parse_order_message(user_message)
    if not parsed_products:
        return "Sorry, I couldn't understand your order. Could you list products with quantities? 🙏", state

    return _validate_and_format_cart(parsed_products, sender, state)


def handle_direct_order(state: dict, sender: str, intent: dict, user_message: str) -> tuple[str, dict]:
    """Single product with quantity."""
    product  = intent.get("product_name", "")
    quantity = intent.get("quantity", "1")
    unit     = intent.get("unit", "ctn")

    # Rebuild as a synthetic order message for the parser
    synthetic = f"{product} {quantity} {unit}"
    return _parse_and_validate(synthetic, sender, state)


def handle_multi_order(state: dict, sender: str, user_message: str) -> tuple[str, dict]:
    """Multi-product order (full raw message to parser)."""
    return _parse_and_validate(user_message, sender, state)


def handle_cart_update(state: dict, sender: str, user_message: str) -> tuple[str, dict]:
    """Update an existing pending cart instead of replacing it."""
    updated_products = _parse_pending_cart_message(user_message)
    if not updated_products:
        return "Sorry, I couldn't understand the cart update. Please send product and quantity like 'fries 4 ctn'.", state
    return handle_cart_update_products(state, sender, updated_products)


def handle_cart_update_products(state: dict, sender: str, updated_products: list[dict]) -> tuple[str, dict]:
    """Merge parsed products into an existing pending cart."""
    existing_products = _state_cart_to_parsed_products(state)
    merged_products = _merge_cart_products(existing_products, updated_products)
    return _validate_and_format_cart(merged_products, sender, state)


def handle_cart_remove(state: dict, sender: str, user_message: str) -> tuple[str, dict]:
    """Remove one matching product from an existing pending cart."""
    target_name = _extract_removed_product_name(user_message)
    if not target_name:
        return "Sorry, I couldn't understand which product to remove.", state

    existing_products = _state_cart_to_parsed_products(state)
    remaining_products = [
        item for item in existing_products
        if not _match_product_name(target_name, item.get("name", ""))
    ]

    if len(remaining_products) == len(existing_products):
        return f"I couldn't find {target_name} in your current cart.", state

    if not remaining_products:
        return handle_cancel_order(state)

    return _validate_and_format_cart(remaining_products, sender, state)


# ─────────────────────────────────────────────────────────────────────
# HANDLER:  CONFIRM ORDER  (user says YES)
# ─────────────────────────────────────────────────────────────────────

def handle_confirm_order(state: dict, sender: str) -> tuple[str, dict]:
    cart_session_id = state.get("cart_session_id", "")

    if not cart_session_id and not state.get("cart"):
        reply = "I don't have an active cart for you. Please send your order first! 🛒"
        return reply, state

    raw_result = place_order_via_module.invoke({
        "sender_number":   sender,
        "cart_session_id": cart_session_id,
    })
    
    print(f"\n[PLACE ORDER RAW RESPONSE] {raw_result}\n")
    order_result = _safe_json(raw_result, {})

    cart_items = state.get("cart", [])
    has_insufficient = False
    for item in cart_items:
        if item.get("status") == "error" or "insufficient" in str(item.get("message", "")).lower():
            has_insufficient = True
            break

    customer_name = state.get("customer_name", "")
    reply = _format_confirm_order_reply(order_result, customer_name, has_insufficient=has_insufficient)

    state["previous_cart"] = state.get("cart", [])
    state["previous_cart_total"] = state.get("cart_total", 0.0)
    state["last_order"] = order_result

    # Clear cart after order
    state = clear_cart(state)
    state["stage"] = "idle"
    state.pop("cart_session_id", None)
    state.pop("cart_updated_at", None)

    return reply, state


# ─────────────────────────────────────────────────────────────────────
# HANDLER:  CANCEL ORDER
# ─────────────────────────────────────────────────────────────────────

def handle_cancel_order(state: dict) -> tuple[str, dict]:
    state = clear_cart(state)
    state.pop("cart_session_id", None)
    state.pop("cart_updated_at", None)
    reply = "No worries! Your cart has been cleared. 😊 Let me know when you're ready to order!"
    return reply, state


# ─────────────────────────────────────────────────────────────────────
# HANDLER:  ORDER STATUS
# ─────────────────────────────────────────────────────────────────────

def handle_order_status(state: dict, sender: str, intent: dict) -> tuple[str, dict]:
    order_id = intent.get("product_name", "")  # user might say "order 12345"
    
    if order_id:
        from tools.presta_tools import get_order_status_via_module
        raw = get_order_status_via_module.invoke({"order_id": order_id})
        data = _safe_json(raw, {})
        
        if data and data.get("status"):
            msg = _format_order_status_reply(data)
            return msg, state
        else:
            return f"I couldn't find order #{order_id}. Please double-check the number. 🙏", state

    from tools.presta_tools import get_undelivered_orders
    raw = get_undelivered_orders.invoke({"sender_number": sender})
    data = _safe_json(raw, {})
    
    orders = data.get("data", []) if isinstance(data.get("data"), list) else []
    
    if not orders:
        return "You currently have no undelivered orders.", state
        
    state["pending_orders"] = orders
    state["stage"] = "awaiting_order_selection"
    
    lines = ["Here are your recent undelivered orders. Please reply with the number (e.g. 1) to check its status:\n"]
    for i, o in enumerate(orders, start=1):
        ref = o.get("order_reference") or o.get("reference") or o.get("id_order", "Unknown")
        lines.append(f"{i}. Order Ref #{ref}")
        
    return "\n".join(lines), state


# ─────────────────────────────────────────────────────────────────────
# HANDLER:  COMPLAINT
# ─────────────────────────────────────────────────────────────────────

def handle_complaint(state: dict, sender: str, intent: dict) -> tuple[str, dict]:
    description = intent.get("complaint_description", "User has a general complaint.")
    priority = intent.get("priority", "medium")
    customer_name = state.get("customer_name", "Customer")
    
    from tools.presta_tools import create_ticket
    create_ticket.invoke({
        "sender_number": sender,
        "customer_name": customer_name,
        "description": description,
        "priority": priority
    })
    
    return "Your ticket is being generated and our agent will call you shortly.", state


# ─────────────────────────────────────────────────────────────────────
# HANDLER:  ORDER INTENT WITH NO QUANTITY
# ─────────────────────────────────────────────────────────────────────

def handle_order_no_qty(state: dict, intent: dict) -> str:
    product = intent.get("product_name") or state.get("last_product", "")
    if product:
        return f"Great! 😊 How many cartons of *{product}* would you like?"
    return "Of course! Which product and quantity would you like to order? 🛒"


# ─────────────────────────────────────────────────────────────────────
# MAIN DISPATCHER
# ─────────────────────────────────────────────────────────────────────

def process_message(user_input: str, sender_number: str) -> str:
    """
    Entry point. Classifies → dispatches → formats reply.
    Persists state to Redis.
    """
    # ── Check for empty input ──
    if not user_input or not user_input.strip():
        print(f"[EMPTY MSG] Dropping empty message from {sender_number}.")
        return ""

    number_key = clean_number(sender_number)
    state      = get_session(number_key)
    original_input = user_input


    # ── Call Request Shortcut ──
    user_trimmed = user_input.strip().lower()
    user_normalized = re.sub(r"\s+", " ", user_trimmed)
    
    call_patterns = [
        r"^plis\s+call$",
        r"^pls\s+call$",
        r"^please\s+call$",
        r"^call\s+me$",
        r"^call$",
        r"^call\s+back$",
        r"^call\s+kro$",
        r"^call\s+please$",
        r"^please\s+call\s+me$",
        r"^plis\s+call\s+me$",
        r"^pls\s+call\s+me$",
        r"^contact\s+me$",
        r"^call\s+please\s+me$",
        r"^plis\s+call\s+please$",
        r"^pls\s+call\s+please$",
    ]
    
    is_call_req = any(re.match(pat, user_normalized) for pat in call_patterns) or (
        "call" in user_normalized and ("please" in user_normalized or "plis" in user_normalized or "pls" in user_normalized or "me" in user_normalized or "back" in user_normalized)
        and not any(neg in user_normalized for neg in ("don't", "dont", "do not", "no call", "no need call"))
    )
    
    if is_call_req:
        import random
        call_phrases = [
            "give me 2 min, calling you. 😊",
            "Just 2 minutes, calling you. 📞",
            "Give me a moment, calling you in 2 mins. 😊",
            "Calling you in 2 minutes! 😊",
            "Hold on for 2 mins, calling you. 📞",
            "Okay, calling you in 2 minutes. 😊",
        ]
        reply = random.choice(call_phrases)
        
        customer_name_str = state.get("customer_name", "")
        if not customer_name_str:
            try:
                raw = search_products.invoke({
                    "product_name": "",
                    "sender_number": sender_number,
                })
                data = _safe_json(raw, {})
                customer_name_str = data.get("customer_name", "")
                if customer_name_str:
                    state["customer_name"] = customer_name_str
            except Exception:
                pass
            
        try:
            from tools.presta_tools import create_ticket
            create_ticket.invoke({
                "sender_number": sender_number,
                "customer_name": customer_name_str or "Customer",
                "description": f"Customer requested a call back: '{user_input}'",
                "priority": "medium"
            })
        except Exception as e:
            print(f"[CALL TICKET ERROR] {e}")
            
        state = append_history(state, "human", original_input)
        state = append_history(state, "assistant", reply)
        save_session(number_key, state)
        return reply

    # ── Cart Expiration Check ──
    import time
    cart_updated_at = state.get("cart_updated_at")
    if cart_updated_at and (time.time() - cart_updated_at > CART_EXPIRATION_MINUTES * 60):
        print(f"[CART EXPIRED] Cart was inactive for {time.time() - cart_updated_at:.1f} seconds (limit: {CART_EXPIRATION_MINUTES} mins). Clearing cart.")
        state = clear_cart(state)
        state.pop("cart_updated_at", None)
        state.pop("cart_session_id", None)

    # ── Check Customer Authentication ──
    customer_name = state.get("customer_name", "")
    if not customer_name:
        try:
            raw = search_products.invoke({
                "product_name": "",
                "sender_number": sender_number,
            })
            data = _safe_json(raw, {})
            customer_name = data.get("customer_name", "")
            if customer_name:
                state["customer_name"] = customer_name
                save_session(number_key, state)
        except Exception:
            pass

    # If customer is still not found, silently drop the message.
    if not customer_name:
        print(f"[AUTH FAILED] No customer found for {sender_number}. Dropping message.")
        return ""


    # ── Stage override: if awaiting_confirm, intercept YES/NO ──
    user_lower = user_input.lower().strip()
    stage      = state.get("stage", "idle")

    if "previous cart" in user_lower or "last cart" in user_lower or "cart details" in user_lower:
        reply = _format_previous_cart(state)
        state = append_history(state, "human", original_input)
        state = append_history(state, "assistant", reply)
        save_session(number_key, state)
        return reply

    if _reference_order_requested(user_input) and state.get("last_lookup_product"):
        pending_products = [{
            "name": _normalize_order_product_name(state["last_lookup_product"]),
            "quantity": 1,
            "unit": "ctn",
        }]
        if stage == "awaiting_confirm":
            reply, state = handle_cart_update_products(state, sender_number, pending_products)
        else:
            reply, state = _validate_and_format_cart(pending_products, sender_number, state)
        state = append_history(state, "human", original_input)
        state = append_history(state, "assistant", reply)
        save_session(number_key, state)
        return reply

    YES_WORDS = {"yes", "y", "ok", "okay", "confirm", "haan", "ha", "theek hai", "sure", "go ahead"}
    NO_WORDS  = {"no", "nahi", "cancel", "nevermind", "nope", "nope", "band kar"}

    if stage == "awaiting_order_selection":
        try:
            choice = int(user_input.strip())
            pending_orders = state.get("pending_orders", [])
            if 1 <= choice <= len(pending_orders):
                order_id = str(pending_orders[choice - 1].get("id_order", ""))
                from tools.presta_tools import get_order_status_via_module
                raw = get_order_status_via_module.invoke({"order_id": order_id})
                data = _safe_json(raw, {})
                print("[ORDER STATUS]", data)
                
                state["stage"] = "idle"
                state["pending_orders"] = []
                
                if data and data.get("status"):
                    reply = _format_order_status_reply(data)
                else:
                    reply = f"I couldn't find order #{order_id}."
                    
                state = append_history(state, "human", original_input)
                state = append_history(state, "assistant", reply)
                save_session(number_key, state)
                return reply
            else:
                reply = "Please reply with a valid number from the list."
                state = append_history(state, "human", original_input)
                state = append_history(state, "assistant", reply)
                save_session(number_key, state)
                return reply
        except ValueError:
            # Not a number -> drop out of selection stage and process as normal intent
            state["stage"] = "idle"
            state["pending_orders"] = []

    if stage == "awaiting_confirm":
        if _is_yes_confirmation(user_input):
            reply, state = handle_confirm_order(state, sender_number)
            state = append_history(state, "human", original_input)
            state = append_history(state, "assistant", reply)
            save_session(number_key, state)
            return reply

        target_remove = _extract_removed_product_name(user_input)
        if target_remove:
            existing_products = _state_cart_to_parsed_products(state)
            has_match = any(_match_product_name(target_remove, item.get("name", "")) for item in existing_products)
            if has_match:
                reply, state = handle_cart_remove(state, sender_number, user_input)
                state = append_history(state, "human", original_input)
                state = append_history(state, "assistant", reply)
                save_session(number_key, state)
                return reply

        rest_msg = _extract_negated_replacement(user_input)
        if rest_msg and not _is_no_discard(user_input):
            print(f"[NEGATED REPLACEMENT] Original: '{user_input}', Rest: '{rest_msg}'. Clearing cart.")
            state = clear_cart(state)
            state.pop("cart_session_id", None)
            state.pop("cart_updated_at", None)
            state["stage"] = "idle"
            stage = "idle"
            user_input = rest_msg
            user_trimmed = user_input.strip().lower()
            user_lower = user_input.lower().strip()

        if _is_no_discard(user_input):
            reply, state = handle_cancel_order(state)
            state = append_history(state, "human", original_input)
            state = append_history(state, "assistant", reply)
            save_session(number_key, state)
            return reply


    # ── Greeting shortcut (before LLM classify) ──
    GREET_WORDS = {"hi","hello","hey","hii","helo","salam","assalam","good morning","good evening","good afternoon"}
    ORDER_KEYWORDS = {
        "deliver","order","send","box","ctn","ctns","carton","pcs","kg",
        "price","rate","cost","available","stock","want","need"
    }
    is_greeting_only = (
        any(user_lower == g or user_lower.startswith(g + " ") for g in GREET_WORDS)
        and not any(kw in user_lower for kw in ORDER_KEYWORDS)
    )
    if is_greeting_only:
        reply = handle_greeting(state, sender_number)
        state = append_history(state, "human", original_input)
        state = append_history(state, "assistant", reply)
        save_session(number_key, state)
        return reply

    # ── Multi-order shortcut (before LLM classify, saves a round-trip) ──
    if stage != "awaiting_confirm" and _multi_order_check(user_input):
        reply, state = handle_multi_order(state, sender_number, user_input)
        state = append_history(state, "human", original_input)
        state = append_history(state, "assistant", reply)
        save_session(number_key, state)
        return reply

    # ── Classify intent via LLM ──
    history_summary = _history_summary(state)
    raw_intent      = intent_chain.invoke({
        "user_message":     user_input,
        "last_product":     state.get("last_product", ""),
        "history_summary":  history_summary,
    })
    raw_intent = raw_intent if isinstance(raw_intent, str) else raw_intent.content
    intent     = _safe_json(raw_intent, {"intent": "general", "general_reply": "How can I help? 😊"})

    print(f"\n[INTENT] {json.dumps(intent, indent=2)}")

    # ── Language Validation ──
    detected_language = str(intent.get("detected_language", "english")).lower().strip()
    SUPPORTED_LANGUAGES = {"english", "arabic", "hindi", "mixed"}
    if detected_language not in SUPPORTED_LANGUAGES:
        print(f"[LANGUAGE BLOCKED] Detected language '{detected_language}' is not supported. Dropping message.")
        return ""

    # Carry-over last_product if intent left it empty
    if not intent.get("product_name") and state.get("last_product"):
        if intent.get("needs_product_lookup") or intent.get("intent") in (
            "direct_order", "order_intent_no_qty", "confirm_order"
        ):
            intent["product_name"] = state["last_product"]

    # Update last_product if the LLM detected one, to maintain context
    # across general replies and no-quantity order intents.
    # Handlers like handle_product_lookup may refine this to a specific top_product_name.
    if intent.get("product_name"):
        state["last_product"] = intent["product_name"]

    reply = ""

    # ── Dispatch ──
    intent_key = intent.get("intent", "general")

    if intent_key == "greeting":
        reply = handle_greeting(state, sender_number)

    elif intent_key in ("price_check", "stock_check", "price_stock"):
        reply = handle_product_lookup(state, sender_number, intent)

    elif intent_key == "direct_order":
        if stage == "awaiting_confirm":
            pending_products = _parse_pending_cart_message(user_input)
            if not pending_products:
                pending_products = _pending_products_from_intent(intent)
            if pending_products:
                reply, state = handle_cart_update_products(state, sender_number, pending_products)
            else:
                reply = "Please send the product update with quantity, like 'fries 3 ctn'."
        else:
            reply, state = handle_direct_order(state, sender_number, intent, user_input)

    elif intent_key == "multi_order":
        if stage == "awaiting_confirm":
            pending_products = _parse_pending_cart_message(user_input)
            if not pending_products:
                pending_products = _pending_products_from_intent(intent)
            if pending_products:
                reply, state = handle_cart_update_products(state, sender_number, pending_products)
            else:
                reply = "Please send the cart update again with product names and quantities."
        else:
            reply, state = handle_multi_order(state, sender_number, user_input)

    elif intent_key == "cart_update":
        removed = intent.get("removed_product", "")
        added_product = intent.get("product_name", "")
        
        if stage == "awaiting_confirm":
            current_parsed = _state_cart_to_parsed_products(state)
            if removed:
                current_parsed = [
                    item for item in current_parsed
                    if not _match_product_name(removed, item.get("name", ""))
                ]
            if added_product:
                pending_products = _pending_products_from_intent(intent)
                current_parsed = _merge_cart_products(current_parsed, pending_products)
                
            if not current_parsed:
                reply, state = handle_cancel_order(state)
            else:
                reply, state = _validate_and_format_cart(current_parsed, sender_number, state)
        else:
            if added_product:
                reply, state = handle_direct_order(state, sender_number, intent, user_input)
            else:
                reply = "You don't have an active cart to update. Please add a product first!"

    elif intent_key == "order_intent_no_qty":
        if stage == "awaiting_confirm":
            pending_products = _parse_pending_cart_message(user_input)
            if not pending_products:
                pending_products = _pending_products_from_intent(intent)
            if pending_products:
                reply, state = handle_cart_update_products(state, sender_number, pending_products)
            else:
                reply = handle_order_no_qty(state, intent)
        else:
            reply = handle_order_no_qty(state, intent)

    elif intent_key == "confirm_order":
        reply, state = handle_confirm_order(state, sender_number)

    elif intent_key == "cancel_order":
        reply, state = handle_cancel_order(state)

    elif intent_key == "order_status":
        reply, state = handle_order_status(state, sender_number, intent)

    elif intent_key == "complaint":
        reply, state = handle_complaint(state, sender_number, intent)

    else:
        # general / fallback
        raw_reply = general_chain.invoke({
            "user_message":     user_input,
            "detected_language": intent.get("detected_language", "english"),
        })
        reply = raw_reply if isinstance(raw_reply, str) else raw_reply.content
        if not reply:
            reply = intent.get("general_reply", "How can I help? 😊")

    # ── Persist ──
    state = append_history(state, "human", original_input)
    state = append_history(state, "assistant", reply)
    save_session(number_key, state)

    return reply
