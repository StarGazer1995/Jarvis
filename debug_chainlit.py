
from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        print("Navigating to http://localhost:8000")
        page.goto("http://localhost:8000")
        
        # Wait for the chat input to appear
        print("Waiting for chat input...")
        page.wait_for_selector("#chat-input", timeout=10000)
        
        # Wait for everything to settle
        time.sleep(5)

        # Type the message
        print("Sending message...")
        page.fill("#chat-input", "Which number is larger, 9.11 or 9.8?")
        # Trigger input event just in case
        page.evaluate("document.querySelector('#chat-input').dispatchEvent(new Event('input', { bubbles: true }));")
        time.sleep(1) # Wait for UI to update
        
        # Check if submit button is enabled
        is_disabled = page.is_disabled("#chat-submit")
        print(f"Submit button disabled: {is_disabled}")
        
        if not is_disabled:
            page.click("#chat-submit")
        else:
            page.press("#chat-input", "Enter")
        
        # Wait for the "Thinking" step to appear
        print("Waiting for response...")
        # Wait for an element with "Thinking" text
        try:
            page.wait_for_selector("text=Thinking", timeout=20000)
            print("Found 'Thinking' step!")
        except:
            print("Did not find 'Thinking' step within timeout.")
            
        # Wait a bit more for completion
        time.sleep(5) 
        
        # Take a screenshot
        page.screenshot(path="chainlit_debug.png")
        print("Screenshot saved to chainlit_debug.png")
        
        # Get HTML content
        content = page.content()
        with open("chainlit_debug.html", "w") as f:
            f.write(content)
        print("HTML content saved to chainlit_debug.html")
        
        browser.close()

if __name__ == "__main__":
    run()
