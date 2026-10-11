"""Security guards: SSRF prevention, download safety, size/page limits, and input sanitization."""
import ipaddress
import socket
import urllib.parse

import requests

from .config import (
    HOST_ALLOWLIST,
    MAX_PDF_BYTES,
    MAX_REDIRECTS,
    PDF_DOWNLOAD_TIMEOUT_SECONDS,
)


class SecurityValidationError(Exception):
    """Raised when URL or content fails security checks."""


# Private & loopback IPv4/IPv6 networks to block for SSRF prevention
BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),          # Current network
    ipaddress.ip_network("10.0.0.0/8"),         # Private-use (RFC 1918)
    ipaddress.ip_network("100.64.0.0/10"),      # Shared Address Space / CGNAT
    ipaddress.ip_network("127.0.0.0/8"),        # Loopback
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local
    ipaddress.ip_network("172.16.0.0/12"),      # Private-use (RFC 1918)
    ipaddress.ip_network("192.0.0.0/24"),       # IETF Protocol Assignments
    ipaddress.ip_network("192.0.2.0/24"),       # TEST-NET-1
    ipaddress.ip_network("192.88.99.0/24"),     # 6to4 Relay Anycast
    ipaddress.ip_network("192.168.0.0/16"),     # Private-use (RFC 1918)
    ipaddress.ip_network("198.18.0.0/15"),      # Network benchmark tests
    ipaddress.ip_network("198.51.100.0/24"),    # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),     # TEST-NET-3
    ipaddress.ip_network("224.0.0.0/4"),        # Multicast
    ipaddress.ip_network("240.0.0.0/4"),        # Reserved
    ipaddress.ip_network("255.255.255.255/32"), # Broadcast
    # IPv6
    ipaddress.ip_network("::1/128"),            # Loopback
    ipaddress.ip_network("::/128"),             # Unspecified
    ipaddress.ip_network("::ffff:0:0/96"),      # IPv4-mapped
    ipaddress.ip_network("100::/64"),           # Discard-only
    ipaddress.ip_network("2001:db8::/32"),      # Documentation
    ipaddress.ip_network("fc00::/7"),           # Unique local
    ipaddress.ip_network("fe80::/10"),          # Link-local
    ipaddress.ip_network("ff00::/8"),           # Multicast
]


def is_ip_blocked(ip_str: str) -> bool:
    """Check if an IP string belongs to any blocked/private/loopback network."""
    try:
        ip = ipaddress.ip_address(ip_str)
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            return True
        for net in BLOCKED_NETWORKS:
            if ip in net:
                return True
        return False
    except ValueError:
        return True


def validate_url(url: str, allowlist: list[str] | None = None) -> tuple[bool, str]:
    """
    Validate that URL is HTTPS only, host does not resolve to private/loopback IP (SSRF),
    and matches optional host allowlist.
    """
    if not url or not isinstance(url, str):
        return False, "URL is empty or invalid"

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() != "https":
        return False, f"Non-HTTPS URL rejected ({parsed.scheme})"

    hostname = parsed.hostname
    if not hostname:
        return False, "URL missing hostname"

    # Host allowlist check (if configured)
    effective_allowlist = allowlist if allowlist is not None else HOST_ALLOWLIST
    if effective_allowlist:
        host_lower = hostname.lower()
        if not any(host_lower == allowed or host_lower.endswith(f".{allowed}") for allowed in effective_allowlist):
            return False, f"Host {hostname} not in allowlist"

    # SSRF: Resolve DNS and check all resulting IPs
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        if not addr_info:
            return False, f"Could not resolve host {hostname}"

        for family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            if is_ip_blocked(ip_str):
                return False, f"Host {hostname} resolved to blocked private/loopback IP ({ip_str})"
    except socket.gaierror as e:
        return False, f"DNS resolution failed for {hostname}: {e}"
    except Exception as e:
        return False, f"Error validating host {hostname}: {e}"

    return True, "OK"


def safe_download_pdf(
    url: str,
    max_bytes: int = MAX_PDF_BYTES,
    timeout: int = PDF_DOWNLOAD_TIMEOUT_SECONDS,
    max_redirects: int = MAX_REDIRECTS,
    allowlist: list[str] | None = None,
) -> bytes:
    """
    Safely download a PDF over HTTPS with SSRF protection, manual redirect following,
    stream size limit, and %PDF magic bytes check.
    """
    curr_url = url
    headers = {
        "User-Agent": "ResearchAssistant/1.0 (Open Access PDF Ingestion; mailto:polite@example.com)",
        "Accept": "application/pdf, */*",
    }

    session = requests.Session()
    session.trust_env = False  # Ignore system proxies for SSRF safety

    for redirect_count in range(max_redirects + 1):
        valid, reason = validate_url(curr_url, allowlist=allowlist)
        if not valid:
            raise SecurityValidationError(f"Download rejected: {reason}")

        try:
            resp = session.get(
                curr_url,
                headers=headers,
                timeout=timeout,
                stream=True,
                allow_redirects=False,
            )
        except requests.RequestException as e:
            raise SecurityValidationError(f"Download network error: {e}")

        # Check redirect
        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location")
            if not location:
                raise SecurityValidationError("Redirect missing Location header")
            # Resolve relative redirect
            curr_url = urllib.parse.urljoin(curr_url, location)
            resp.close()
            continue

        if resp.status_code != 200:
            resp.close()
            raise SecurityValidationError(f"HTTP error {resp.status_code} downloading PDF")

        # Verify content-type if present
        ctype = resp.headers.get("Content-Type", "").lower()
        # Some servers return application/octet-stream or application/x-download for PDFs
        if ctype and not any(t in ctype for t in ["pdf", "octet-stream", "binary"]):
            resp.close()
            raise SecurityValidationError(f"Invalid Content-Type for PDF: {ctype}")

        # Stream and enforce max_bytes
        chunks = []
        total_downloaded = 0
        try:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    total_downloaded += len(chunk)
                    if total_downloaded > max_bytes:
                        resp.close()
                        raise SecurityValidationError(
                            f"File size exceeded maximum limit of {max_bytes} bytes ({total_downloaded} bytes)"
                        )
                    chunks.append(chunk)
        finally:
            resp.close()

        pdf_bytes = b"".join(chunks)

        # Verify %PDF magic bytes
        if len(pdf_bytes) < 4 or not pdf_bytes.startswith(b"%PDF"):
            raise SecurityValidationError("File content does not start with %PDF magic bytes")

        return pdf_bytes

    raise SecurityValidationError(f"Exceeded maximum redirects ({max_redirects})")


def sanitize_extracted_text(text: str) -> str:
    """
    Sanitize extracted PDF text. Treat all PDF text as untrusted data.
    Remove null bytes and strip zero-width characters while preserving punctuation and whitespace.
    """
    if not text:
        return ""
    # Remove null bytes and non-printable control characters (except newline, tab, cr)
    cleaned = "".join(ch for ch in text if ch in ("\n", "\r", "\t") or (ch.isprintable() and ord(ch) >= 32))
    return cleaned.strip()
