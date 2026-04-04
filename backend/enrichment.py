"""
Network/device enrichment for HumanGuard.

Extracts 7 numeric features per request that complement the 30 behavioural
signals from FeatureExtractor:

  is_headless_browser   — 0/1  headless Chrome, Puppeteer, Selenium, PhantomJS, etc.
  is_known_bot_ua       — 0/1  curl, wget, python-requests, scrapy, etc.
  is_datacenter_ip      — 0/1  AWS / GCP / Azure / DigitalOcean / Linode / Vultr / OVH
  ua_entropy            — float  length × char-diversity ratio (bots have low-entropy UAs)
  has_accept_language   — 0/1  browsers always send Accept-Language; bots often omit it
  accept_language_count — int  # of languages in the header (real browsers: 3-5)
  suspicious_header_count — int  # of expected browser headers that are absent
"""

import re
import time
import urllib.request
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── In-memory IP cache  ────────────────────────────────────────────────────
# ip -> (epoch_ts, result_dict)
_ip_cache: dict = {}
_IP_CACHE_TTL = 3600  # 1 hour

# ── Datacenter ASN / org keywords ─────────────────────────────────────────
_DATACENTER_KEYWORDS = frozenset([
    "amazon", "aws", "google", "azure", "microsoft", "digitalocean", "linode",
    "vultr", "ovh", "hetzner", "rackspace", "cloudflare", "fastly", "akamai",
    "leaseweb", "choopa", "psychz", "quadranet", "colocrossing", "packet",
    "equinix", "coresite", "serverius", "servercentral", "codero",
])

# ── Regex patterns  ────────────────────────────────────────────────────────
_BOT_UA_RE = re.compile(
    r"(curl|wget|python-requests|python-urllib|scrapy|httpx|aiohttp|"
    r"okhttp|go-http-client|java/|libwww|lwp-|node-fetch|axios|got/|"
    r"httpclient|bot|crawl|spider|monitor)",
    re.IGNORECASE,
)

_HEADLESS_UA_RE = re.compile(
    r"(HeadlessChrome|Headless|PhantomJS|Selenium|Playwright|Puppeteer|"
    r"WebDriver|ChromeDriver|geckodriver)",
    re.IGNORECASE,
)

_EXPECTED_HEADERS = frozenset(["accept", "accept-encoding", "accept-language", "user-agent"])

# ── Safe defaults returned when enrichment is unavailable ─────────────────
_SAFE_DEFAULTS: dict = {
    "is_headless_browser": False,
    "is_known_bot_ua": False,
    "browser_type": "unknown",
    "os_type": "unknown",
    "ua_entropy": 0.0,
    "is_datacenter_ip": False,
    "is_vpn": False,
    "country": "unknown",
    "org": "",
    "has_accept_language": True,
    "has_referer": False,
    "accept_language_count": 0,
    "suspicious_header_count": 0,
}


def parse_user_agent(ua: Optional[str]) -> dict:
    """
    Parse a User-Agent string into feature flags.

    Returns a dict with keys:
      is_headless_browser, is_known_bot_ua, browser_type, os_type, ua_entropy
    """
    if not ua:
        return {
            "is_headless_browser": True,   # missing UA is itself suspicious
            "is_known_bot_ua": False,
            "browser_type": "unknown",
            "os_type": "unknown",
            "ua_entropy": 0.0,
        }

    is_headless = bool(_HEADLESS_UA_RE.search(ua))
    is_bot = bool(_BOT_UA_RE.search(ua))

    ua_lower = ua.lower()

    # Browser type — order matters (Edge contains "Chrome")
    if "edg/" in ua_lower or "edge/" in ua_lower:
        browser = "edge"
    elif "chrome" in ua_lower and "safari" in ua_lower:
        browser = "chrome"
    elif "firefox" in ua_lower:
        browser = "firefox"
    elif "safari" in ua_lower:
        browser = "safari"
    else:
        browser = "other"

    # OS type
    if "windows" in ua_lower:
        os_type = "windows"
    elif "mac os" in ua_lower or "macintosh" in ua_lower:
        os_type = "mac"
    elif "android" in ua_lower or "iphone" in ua_lower or "ipad" in ua_lower:
        os_type = "mobile"
    elif "linux" in ua_lower:
        os_type = "linux"
    else:
        os_type = "unknown"

    # UA entropy: (length × unique-char ratio).  Bots often use short, templated UAs.
    unique_chars = len(set(ua))
    ua_entropy = round(len(ua) * (unique_chars / 128.0), 2)

    return {
        "is_headless_browser": is_headless,
        "is_known_bot_ua": is_bot,
        "browser_type": browser,
        "os_type": os_type,
        "ua_entropy": ua_entropy,
    }


def get_ip_info(ip: str) -> dict:
    """
    Fetch IP metadata from ipinfo.io free tier (≤50 k req/month).

    Results are cached in memory for _IP_CACHE_TTL seconds.
    Always returns a dict with safe defaults — never raises.

    Returns a dict with keys: is_datacenter, is_vpn, country, org
    """
    defaults = {"is_datacenter": False, "is_vpn": False, "country": "unknown", "org": ""}

    if not ip or ip in ("127.0.0.1", "::1", "localhost"):
        return defaults

    now = time.time()
    if ip in _ip_cache:
        cached_ts, cached_result = _ip_cache[ip]
        if now - cached_ts < _IP_CACHE_TTL:
            return cached_result

    try:
        req = urllib.request.Request(
            f"https://ipinfo.io/{ip}/json",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        org = data.get("org", "") or ""
        org_lower = org.lower()
        is_datacenter = any(kw in org_lower for kw in _DATACENTER_KEYWORDS)

        result = {
            "is_datacenter": is_datacenter,
            "is_vpn": False,   # ipinfo free tier does not expose VPN flag
            "country": data.get("country", "unknown") or "unknown",
            "org": org,
        }
        _ip_cache[ip] = (now, result)
        return result

    except Exception as exc:
        logger.debug("ipinfo.io lookup failed for %s: %s", ip, exc)
        return defaults


def parse_request_headers(headers: dict) -> dict:
    """
    Analyse HTTP headers for bot-indicator signals.

    Returns a dict with keys:
      has_accept_language, has_referer, accept_language_count,
      suspicious_header_count
    """
    present_lower = {k.lower() for k in headers}

    has_accept_language = "accept-language" in present_lower
    has_referer = "referer" in present_lower

    accept_lang = headers.get("Accept-Language") or headers.get("accept-language") or ""
    lang_count = (
        len([seg.strip() for seg in accept_lang.split(",") if seg.strip()])
        if accept_lang else 0
    )

    missing_count = sum(1 for h in _EXPECTED_HEADERS if h not in present_lower)

    return {
        "has_accept_language": has_accept_language,
        "has_referer": has_referer,
        "accept_language_count": lang_count,
        "suspicious_header_count": missing_count,
    }


def enrich_request(request) -> dict:
    """
    Extract all network/device features from a Flask request object.

    Returns a flat dict combining UA, IP, and header signals.
    Never raises — safe defaults returned on any failure.
    """
    try:
        ua = request.headers.get("User-Agent", "") or ""
        forwarded_for = request.headers.get("X-Forwarded-For", "") or ""
        ip = forwarded_for.split(",")[0].strip() if forwarded_for else (request.remote_addr or "")

        ua_features = parse_user_agent(ua or None)
        ip_features = get_ip_info(ip)
        header_features = parse_request_headers(dict(request.headers))

        return {
            **ua_features,
            "is_datacenter_ip": ip_features["is_datacenter"],
            "is_vpn": ip_features["is_vpn"],
            "country": ip_features["country"],
            "org": ip_features["org"],
            **header_features,
        }

    except Exception as exc:
        logger.warning("enrich_request failed, using safe defaults: %s", exc)
        return dict(_SAFE_DEFAULTS)
