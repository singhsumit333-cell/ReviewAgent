"""
aup_scanner_app.py
Streamlit UI for aup_scanner.py.

This imports the actual scanning functions from aup_scanner.py rather than
rewriting the logic here - one source of truth for the crawling/checking
code, this file just adds a UI around it. Because aup_scanner.py guards
its CLI part with `if __name__ == "__main__":`, importing it like a normal
module is safe - that block simply won't run.

Note this is still a SEPARATE app from compliance_app.py, not merged into
it. A full catalog scan can take a while (one request + one vision check
per product), which doesn't fit inside a live chat conversation - same
reasoning as keeping aup_scanner.py a standalone script in the first
place. This just swaps "run in a terminal" for "run in a browser tab."
"""

import streamlit as st
from AUP_Scanner import find_product_links, extract_product_content, check_product_for_violation

st.set_page_config(page_title="AUP Scanner", page_icon="🚨")
st.title("🚨 AUP Catalog Scanner")
st.caption(
    "Enter a merchant's website URL. Checks each product's title, "
    "description, price, and image against AUP categories - counterfeit "
    "goods, weapons, adult content, and prescription-only medicine - and "
    "stops at the first violation found."
)

url = st.text_input("Website URL", placeholder="https://example.com")
scan_clicked = st.button("Scan", type="primary")

if scan_clicked and url:
    with st.spinner("Crawling for product pages..."):
        product_links = find_product_links(url)

    if not product_links:
        st.warning(
            "No product pages found. This site's link/sitemap structure "
            "may not match what this scanner looks for - try pointing it "
            "at a specific category page URL instead of the homepage."
        )
    else:
        st.write(f"Found **{len(product_links)}** product page(s) to check.")

        # This placeholder gets overwritten on every loop iteration, which
        # is what gives the "live progress" feel - without it, Streamlit
        # would just print a new line for every single product checked.
        progress = st.empty()
        violation_found = False

        for i, link in enumerate(product_links, 1):
            progress.markdown(f"🔍 Checking `[{i}/{len(product_links)}]` {link}")

            try:
                title, description, price, image_url = extract_product_content(link)
                is_violation, full_response = check_product_for_violation(
                    title, description, price, image_url
                )
            except Exception as e:
                # Surface the real error instead of silently skipping - a
                # swallowed exception here would make every product "fail"
                # silently and the scan would wrongly end in "no violations
                # found", which looks identical to a genuinely clean result.
                st.warning(f"Skipped `{link}` - error: {e}")
                continue

            if is_violation:
                violation_found = True
                progress.empty()

                st.error("AUP VIOLATION FOUND")
                st.markdown(f"**URL:** {link}")
                if image_url:
                    st.image(image_url, width=250)
                st.markdown(full_response)
                break  # stop at the first violation, same as the terminal version

        if not violation_found:
            progress.empty()
            st.success("No AUP violations found across the pages checked.")