from playwright.sync_api import sync_playwright

def scrape_bayut_url(url):
    """Scrape Bayut properties from a given URL"""
    print(f"🔍 DEBUG: Scraper received URL: {url}")
    print(f"🔍 DEBUG: Starting Playwright...")
    
    pw = sync_playwright().start()
    
    browser = pw.firefox.launch(
        headless=True,
        #slow_mo=2000,
    )
    
    page = browser.new_page()
    print(f"🔍 DEBUG: Navigating to: {url}")
    page.goto(url)
    print(f"🔍 DEBUG: Page loaded successfully")
    
    # Get all the links and extract unique URLs in one go
    print(f"🔍 DEBUG: Looking for property links...")
    links = page.locator("xpath=//a[contains(@href, 'property/details') and not(ancestor::div[@aria-label='Recommended search hits'])]").all()
    
    urls = list(set(link.get_attribute("href") for link in links if link.get_attribute("href")))
    print(f"🔍 DEBUG: Found {len(urls)} property links")
    
    results = []
    
    # Now visit each property detail page (limit to 5)
    for i, url in enumerate(urls[:5]):
        property_data = {
            "url": f"https://www.bayut.com{url}",
            "pricing": "Not found",
            "location": "Not found", 
            "description": "Not found"
        }
        
        # Navigate to the property detail page
        page.goto(f"https://www.bayut.com{url}")
        
        # Extract the 3 specific containers
        try:
            # 1. Pricing
            pricing = page.locator("xpath=//div[contains(@class, 'fc84e39c') and contains(@class, 'cd769dae')]").text_content()
            property_data["pricing"] = pricing
        except:
            pass
        
        try:
            # 2. Property Header (Location)
            location = page.locator("xpath=//div[@aria-label='Property header']").text_content()
            property_data["location"] = location
        except:
            pass
        
        try:
            # 3. Property Description
            description = page.locator("xpath=//div[@aria-label='Property description']").text_content()
            property_data["description"] = description
        except:
            pass
        
        results.append(property_data)
    
    browser.close()
    pw.stop()
    
    print(f"🔍 DEBUG: Scraper completed, returning {len(results)} results")
    return results

# Keep the original script functionality for direct execution
if __name__ == "__main__":
    pw = sync_playwright().start()
    
    browser = pw.firefox.launch(
        headless=True,
        #slow_mo=2000,
    )
    
    page=browser.new_page()
    page.goto("https://www.bayut.com/to-rent/villas/uae/?rent_frequency=monthly&price_min=5000&price_max=6000")
    
    # Get all the links and extract unique URLs in one go
    links = page.locator("xpath=//a[contains(@href, 'property/details') and not(ancestor::div[@aria-label='Recommended search hits'])]").all()
    
    urls = list(set(link.get_attribute("href") for link in links if link.get_attribute("href")))
    
    print(f"Found {len(urls)} unique property links:")
    for i, url in enumerate(urls):
        print(f"{i+1}. {url}")
    
    # Now visit each property detail page
    for i, url in enumerate(urls):
        print(f"\n{'='*50}")
        print(f"PROPERTY {i+1}: {url}")
        print(f"{'='*50}")
        
        # Navigate to the property detail page
        page.goto(f"https://www.bayut.com{url}")
        
        # Extract the 3 specific containers
        try:
            # 1. Pricing
            pricing = page.locator("xpath=//div[contains(@class, 'fc84e39c') and contains(@class, 'cd769dae')]").text_content()
            print(f"Pricing: {pricing}")
        except:
            print("Pricing: Not found")
        
        try:
            # 2. Property Header (Location)
            location = page.locator("xpath=//div[@aria-label='Property header']").text_content()
            print(f"Location: {location}")
        except:
            print("Location: Not found")
        
        try:
            # 3. Property Description
            description = page.locator("xpath=//div[@aria-label='Property description']").text_content()
            print(f"Description: {description}")
        except:
            print("Description: Not found")
        
        print(f"\n{'-'*30}")
    
    page.screenshot(path="screenshot.png")
    browser.close()


