"""
aup_scanner.py
Standalone tool: given a merchant's website URL, crawls the product
catalog and checks each product (title, description, AND image) against
four AUP categories - counterfeit goods, weapons, adult content, and
prescription-only medicine sold without a prescription.

Design choice, per your instructions: this does NOT build an exhaustive
report of every product. It checks products one at a time and STOPS at
the first violation it finds, printing that one result. If it gets
through every product with nothing flagged, it says so at the end.

This is intentionally separate from compliance_agent.py - a full catalog
crawl can take a while (network requests + a vision check per product),
which doesn't fit inside a live chat conversation. Run this on its own
whenever a case actually needs it.
"""

import sys
import re
import time
import base64
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

AUP_CATEGORIES = """
1. Counterfeit goods - judged on TWO things together, not either alone:
   (a) Design/branding match - does the image show a real, identifiable
       brand's product (logo, distinctive design, model name)?
   (b) Price point - is the listed price far below what that genuine
       branded item actually costs? A generic-looking item with no brand
       is NOT counterfeit just because it's cheap. A branded-looking item
       at a genuine market price is NOT counterfeit either. It's the
       COMBINATION - a real brand's product listed at a price no genuine
       unit of that item would sell for - that indicates counterfeiting.
       If you recognize a brand in the image/title, use web_search to look
       up that item's actual retail price and compare it to the listed
       price before concluding counterfeit.
   EXCEPTION - self-declared replicas: if the title, description, or page
   text itself uses words like "first copy", "replica", "dupe", "master
   quality", "AAA quality", or "mirror quality", that alone is a direct
   admission the product is not genuine - flag it as counterfeit
   immediately, no price/brand comparison needed to confirm this case.
2. Weapons - firearms, ammunition, knives or other items marketed and
   sold specifically AS weapons (not ordinary kitchen or utility tools).
3. Adult content - sexually explicit products or imagery.
4. Prescription-only medicine - medicines that normally require a
   doctor's prescription, listed for sale with no prescription
   requirement mentioned anywhere on the page.
"""

WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 3
}

# A generic browser User-Agent - some sites block requests that don't look
# like they're coming from an actual browser.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AUP-Scanner/1.0)"}


def get_soup(url):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def find_product_links(base_url):
    """
    Two ways to find product pages, tried in order:

    1. sitemap.xml - most storefront platforms (Shopify, WooCommerce, etc.)
       publish one automatically. It's just a clean XML list of every page
       on the site, so this is far more reliable than trying to click
       through category pages and pagination ourselves.

    2. Fallback: scan the homepage for any link containing "/product" in
       the URL, in case the site has no sitemap or an unusual one.
    """
    links = set()
    domain = urlparse(base_url).netloc

    try:
        sitemap_url = urljoin(base_url, "/sitemap.xml")
        resp = requests.get(sitemap_url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "xml")
            for loc in soup.find_all("loc"):
                url = loc.text.strip()
                if "/product" in url.lower():
                    links.add(url)
    except requests.RequestException:
        pass

    if not links:
        try:
            soup = get_soup(base_url)
            for a in soup.find_all("a", href=True):
                href = urljoin(base_url, a["href"])
                if urlparse(href).netloc == domain and "/product" in href.lower():
                    links.add(href)
        except requests.RequestException:
            pass

    return list(links)


def extract_price(soup):
    """
    Tries a few common places storefronts put the price:
    1. An element whose class name contains "price" (covers most themes -
       Shopify/WooCommerce commonly use classes like "product-price",
       "price__regular", etc.)
    2. Standard e-commerce meta tags (product:price:amount, og:price:amount)
       that some themes include for social-sharing/SEO purposes.
    Returns None if nothing matches - the site may use a JS-rendered price
    that a simple HTML fetch can't see, in which case this script can't
    extract it without a headless browser (a possible future upgrade).
    """
    price_tag = soup.find(attrs={"class": lambda c: c and "price" in c.lower()})
    if price_tag:
        return price_tag.get_text(strip=True)

    for prop in ("product:price:amount", "og:price:amount"):
        meta_tag = soup.find("meta", attrs={"property": prop})
        if meta_tag and meta_tag.get("content"):
            return meta_tag["content"]

    return None


def extract_image(soup, page_url):
    """
    The first <img> tag on a page is very often the site's own logo in the
    header - not the product photo (confirmed on this exact site: FOOTHUNK's
    logo was being sent to Claude for every single product). Two better
    strategies, tried in order:

    1. og:image meta tag - the standard way a page declares "this is MY
       representative image" (used for social-media link previews), which
       is much more reliably the actual product photo.
    2. Fallback: scan all <img> tags and skip anything whose src/alt text
       suggests it's a logo or icon rather than product content.
    """
    og_image = soup.find("meta", attrs={"property": "og:image"})
    if og_image and og_image.get("content"):
        return urljoin(page_url, og_image["content"])

    for img in soup.find_all("img", src=True):
        alt = (img.get("alt") or "").lower()
        src = img["src"].lower()
        if "logo" in alt or "logo" in src or "icon" in alt or "icon" in src:
            continue
        return urljoin(page_url, img["src"])

    return None


def extract_product_content(url):
    """
    Pulls the title, a bit of description text, the price, and the first
    product image from one product page. These selectors are generic
    best-effort (h1 for title, first image on the page, any element with
    "description"/"price" in its class name) - every storefront theme
    names things differently, so this may need small tweaks for a
    specific site if it comes back empty.
    """
    soup = get_soup(url)

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    description = ""
    desc_candidates = soup.find_all(["p", "div"], class_=lambda c: c and "description" in c.lower())
    if desc_candidates:
        description = " ".join(d.get_text(strip=True) for d in desc_candidates[:2])

    price = extract_price(soup)
    image_url = extract_image(soup, url)

    return title, description, price, image_url


def download_image_as_base64(image_url):
    resp = requests.get(image_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    media_type = resp.headers.get("Content-Type", "image/jpeg")
    encoded = base64.standard_b64encode(resp.content).decode("utf-8")
    return media_type, encoded


def check_product_for_violation(title, description, price, image_url):
    """
    Sends the product's text, price, AND image to Claude in one message
    and asks for a judgment against the four AUP categories. This is a
    vision request - the image is sent as base64-encoded data alongside
    the text, which is how you give Claude something to actually look at,
    not just read about.

    web_search is included as a tool here specifically for the counterfeit
    check: if Claude recognizes a brand in the image/title, it can look up
    that item's real retail price and compare it against the listed price,
    instead of guessing from memory whether the price "feels" too low.
    Since web_search is a built-in server tool, no manual tool-loop code
    is needed here - it resolves within this one call.
    """
    content = [{
        "type": "text",
        "text": f"""Product title: {title}
Product description: {description or "(none found)"}
Listed price: {price or "(price not found on page)"}

Check this product against these AUP categories:
{AUP_CATEGORIES}

If the image/title shows a recognizable brand, use web_search to look up
that item's genuine retail price before judging counterfeiting - state
what you found and how it compares to the listed price.

Does this product violate ANY of these categories? Respond in exactly
this format:
VIOLATION: yes or no
CATEGORY: which category it violates, or "none"
REASON: one or two sentences explaining your judgment, including the
price comparison if counterfeiting was the category considered
"""
    }]

    if image_url:
        try:
            media_type, encoded = download_image_as_base64(image_url)
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": encoded}
            })
        except requests.RequestException:
            pass  # fall back to a text-only judgment if the image can't be fetched

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": content}]
    )

    text = "".join(b.text for b in response.content if b.type == "text")

    # An exact substring match (e.g. "violation: yes") is fragile - Claude
    # may wrap the label in markdown (**VIOLATION:** Yes) even when asked
    # for a plain format, which would silently break a plain match and
    # report "no violation" even when the actual judgment was yes. This
    # strips formatting characters and allows flexible spacing before
    # checking, so the parsing doesn't depend on exact character-for-
    # character output.
    clean_text = re.sub(r"[*_`]", "", text)
    is_violation = bool(re.search(r"violation\s*:\s*yes", clean_text, re.IGNORECASE))
    return is_violation, text


def scan_website(base_url):
    print(f"Crawling {base_url} for product pages...")
    product_links = find_product_links(base_url)
    print(f"Found {len(product_links)} product page(s) to check.\n")

    if not product_links:
        print("No product pages found. This site's link/sitemap structure "
              "may not match what this script looks for - try pointing it "
              "at a specific category page URL instead of the homepage.")
        return

    for i, url in enumerate(product_links, 1):
        print(f"[{i}/{len(product_links)}] Checking {url}")
        try:
            title, description, price, image_url = extract_product_content(url)
            is_violation, full_response = check_product_for_violation(title, description, price, image_url)
        except Exception as e:
            print(f"   Skipped - error reading this page: {e}")
            continue

        if is_violation:
            print("\n" + "=" * 60)
            print("AUP VIOLATION FOUND")
            print("=" * 60)
            print(f"URL: {url}")
            print(full_response)
            print("=" * 60)
            return  # stop at the first violation, as requested

        time.sleep(0.5)  # small pause between requests, polite to the server

    print("\nNo AUP violations found across the pages checked.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aup_scanner.py <website_url>")
        sys.exit(1)

    scan_website(sys.argv[1])