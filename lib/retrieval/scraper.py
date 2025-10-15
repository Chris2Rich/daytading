from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("https://seekingalpha.com/news/4494763-tesla-rallies-to-an-eight-month-high-after-elon-musk-buys-more-than-1b-worth-of-shares")

    frame_locator = page.frame_locator('iframe[title="Human verification challenge"]')

    # Now, find the element within the iframe to interact with.
    # Since the content of the iframe is not visible in the HTML, 
    # you might need to inspect it in a live browser.
    # Often, the element to hold is the `<body>` or a specific `<div>`.
    # Let's assume you need to press and hold the body of the iframe.
    captcha_hold_element = frame_locator.locator("body")

    # Perform the press and hold action
    if captcha_hold_element:
        print("CAPTCHA iframe found, attempting to press and hold...")
        
        # Get the bounding box of the element to click in the center
        box = captcha_hold_element.bounding_box()
        if box:
            # Move the mouse to the center of the element within the iframe
            page.mouse.move(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)
            
            # Press and hold the mouse button
            page.mouse.down()
            
            # Wait for a few seconds (you may need to adjust this duration)
            page.wait_for_timeout(5000) 
            
            # Release the mouse button
            page.mouse.up()
            
            print("Interaction complete.")

    print(page.content())
    
    browser.close()