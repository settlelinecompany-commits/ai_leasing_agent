"""
Scrape Rocky Real Estate properties from Bayut company page
Extracts detailed property information and saves to CSV
"""
import csv
import re
from playwright.sync_api import sync_playwright
from typing import List, Dict, Optional

def extract_price_value(price_text: str) -> Dict[str, Optional[float]]:
    """Extract numeric price values from text like 'AED 85,000/year' or 'AED 7,083/month'"""
    price_data = {
        "monthly_rent": None,
        "yearly_rent": None,
        "currency": "AED",
        "rent_frequency": None
    }
    
    if not price_text or price_text == "Not found":
        return price_data
    
    # Extract number (handles commas)
    numbers = re.findall(r'[\d,]+', price_text.replace(',', ''))
    if numbers:
        try:
            price_value = float(numbers[0].replace(',', ''))
            
            # Determine frequency
            price_lower = price_text.lower()
            if 'month' in price_lower or '/month' in price_lower:
                price_data["monthly_rent"] = price_value
                price_data["rent_frequency"] = "monthly"
                price_data["yearly_rent"] = price_value * 12
            elif 'year' in price_lower or '/year' in price_lower or '/yr' in price_lower:
                price_data["yearly_rent"] = price_value
                price_data["rent_frequency"] = "yearly"
                price_data["monthly_rent"] = price_value / 12
            elif 'quarter' in price_lower or '/quarter' in price_lower:
                price_data["rent_frequency"] = "quarterly"
                price_data["monthly_rent"] = price_value / 3
                price_data["yearly_rent"] = price_value * 4
        except ValueError:
            pass
    
    return price_data

def extract_property_details(page) -> Dict:
    """Extract detailed property information from a property detail page"""
    property_data = {
        # Basic info
        "property_id": None,
        "url": None,
        "property_manager": "Rocky Real Estate",
        
        # Pricing (raw and parsed)
        "pricing_raw": "Not found",
        "monthly_rent": None,
        "yearly_rent": None,
        "currency": "AED",
        "rent_frequency": None,
        
        # Location
        "location_raw": "Not found",
        "location": None,
        "area": None,
        "city": None,
        
        # Property details
        "bedrooms": None,
        "bathrooms": None,
        "sqft": None,
        "property_type": None,
        "furnished": None,
        "parking": None,
        "parking_spots": None,
        
        # Amenities
        "amenities": [],
        "pet_friendly": None,
        "security_24_7": None,
        "nearby_metro": None,
        "metro_distance_km": None,
        "nearby_shops": None,
        "shops_distance_km": None,
        
        # Description
        "description": "Not found",
        
        # For embedding (will be constructed later)
        "embedding_text": None
    }
    
    # Extract pricing
    try:
        pricing = page.locator("xpath=//div[contains(@class, 'fc84e39c') and contains(@class, 'cd769dae')]").text_content()
        if pricing:
            property_data["pricing_raw"] = pricing.strip()
            price_info = extract_price_value(pricing)
            property_data.update(price_info)
    except:
        pass
    
    # Extract location/header
    try:
        location = page.locator("xpath=//div[@aria-label='Property header']").text_content()
        if location:
            property_data["location_raw"] = location.strip()
            # Try to parse location components
            location_parts = location.split(',')
            if len(location_parts) >= 2:
                property_data["area"] = location_parts[0].strip()
                property_data["city"] = location_parts[-1].strip()
            property_data["location"] = location.strip()
    except:
        pass
    
    # Extract property type and size from location or description
    try:
        # Look for property type in location text
        location_text = property_data.get("location_raw", "").lower()
        if "apartment" in location_text or "flat" in location_text:
            property_data["property_type"] = "Apartment"
        elif "villa" in location_text:
            property_data["property_type"] = "Villa"
        elif "townhouse" in location_text:
            property_data["property_type"] = "Townhouse"
        elif "penthouse" in location_text:
            property_data["property_type"] = "Penthouse"
        elif "studio" in location_text:
            property_data["property_type"] = "Studio"
    except:
        pass
    
    # Extract description FIRST (needed for other extractions)
    description_text = ""
    try:
        description = page.locator("xpath=//div[@aria-label='Property description']").text_content()
        if description:
            description_text = description.strip()
            property_data["description"] = description_text
    except:
        pass
    
    # Extract bedrooms and bathrooms using aria-label attributes (most reliable)
    try:
        # Use aria-label="Beds" to find beds count directly from page
        try:
            beds_element = page.locator("span[aria-label='Beds'] span._3458a9d4").first
            if beds_element.is_visible(timeout=1000):
                beds_text = beds_element.text_content()
                if beds_text:
                    # Extract number from "3 Beds" or "Studio"
                    beds_text = beds_text.strip()
                    if beds_text.lower() == "studio":
                        property_data["bedrooms"] = 0
                    else:
                        bed_num = re.search(r'(\d+)', beds_text)
                        if bed_num:
                            property_data["bedrooms"] = int(bed_num.group(1))
        except:
            pass
        
        # Use aria-label="Baths" to find baths count directly from page
        try:
            baths_element = page.locator("span[aria-label='Baths'] span._3458a9d4").first
            if baths_element.is_visible(timeout=1000):
                baths_text = baths_element.text_content()
                if baths_text:
                    # Extract number from "4 Baths"
                    bath_num = re.search(r'(\d+)', baths_text)
                    if bath_num:
                        property_data["bathrooms"] = int(bath_num.group(1))
        except:
            pass
        
        # Fallback: Try location if aria-label didn't work
        if not property_data.get("bedrooms"):
            location_text = property_data.get("location_raw", "")
            bed_match = re.search(r'(\d+)\s*(?:BR|Bed|Bedroom)', location_text, re.IGNORECASE)
            if bed_match:
                property_data["bedrooms"] = int(bed_match.group(1))
        
        if not property_data.get("bathrooms"):
            location_text = property_data.get("location_raw", "")
            bath_match = re.search(r'(\d+)\s*(?:Bath|Bathroom)', location_text, re.IGNORECASE)
            if bath_match:
                property_data["bathrooms"] = int(bath_match.group(1))
        
        # Last resort: Try description if still not found
        if not property_data.get("bedrooms") and description_text:
            # Look for "Three Bedroom", "2 Bedroom", "Studio" patterns
            bed_patterns = [
                r'(\d+)\s*(?:bedroom|bed|br)',
                r'(one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:bedroom|bed)',
                r'studio',
            ]
            for pattern in bed_patterns:
                bed_match = re.search(pattern, description_text, re.IGNORECASE)
                if bed_match:
                    if bed_match.group(1).isdigit():
                        property_data["bedrooms"] = int(bed_match.group(1))
                    elif bed_match.group(1).lower() == "studio":
                        property_data["bedrooms"] = 0
                    else:
                        # Convert word to number
                        word_to_num = {
                            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
                        }
                        word = bed_match.group(1).lower()
                        if word in word_to_num:
                            property_data["bedrooms"] = word_to_num[word]
                    break
        
        if not property_data.get("bathrooms") and description_text:
            bath_match = re.search(r'(\d+)\s*(?:bathroom|bath)', description_text, re.IGNORECASE)
            if bath_match:
                property_data["bathrooms"] = int(bath_match.group(1))
    except:
        pass
    
    # Extract additional details from description
    if description_text:
        try:
            desc_lower = description_text.lower()
            
            # Furnished status
            if "furnished" in desc_lower:
                if "unfurnished" in desc_lower or "not furnished" in desc_lower:
                    property_data["furnished"] = False
                elif "semi-furnished" in desc_lower or "semi furnished" in desc_lower:
                    property_data["furnished"] = "Semi-furnished"
                else:
                    property_data["furnished"] = True
            
            # Parking
            if "parking" in desc_lower:
                property_data["parking"] = True
                # Try to extract number of parking spots - look for "2 Allotted Parking", "1 parking", etc.
                parking_patterns = [
                    r'(\d+)\s*(?:allotted\s*)?parking',
                    r'(\d+)\s*(?:parking\s*)?spot',
                    r'parking[:\s]+(\d+)',
                ]
                for pattern in parking_patterns:
                    parking_match = re.search(pattern, desc_lower)
                    if parking_match:
                        property_data["parking_spots"] = int(parking_match.group(1))
                        break
                if not property_data.get("parking_spots"):
                    property_data["parking_spots"] = 1
            
            # Amenities
            amenities_list = []
            if "gym" in desc_lower or "fitness" in desc_lower:
                amenities_list.append("gym")
            if "pool" in desc_lower or "swimming" in desc_lower:
                amenities_list.append("pool")
            if "security" in desc_lower or "24/7" in desc_lower or "24 hour" in desc_lower:
                amenities_list.append("security")
                property_data["security_24_7"] = True
            if "elevator" in desc_lower or "lift" in desc_lower:
                amenities_list.append("elevator")
            if "balcony" in desc_lower:
                amenities_list.append("balcony")
            if "metro" in desc_lower or "station" in desc_lower:
                property_data["nearby_metro"] = True
            if "shop" in desc_lower or "mall" in desc_lower or "supermarket" in desc_lower:
                property_data["nearby_shops"] = True
            if "pet" in desc_lower and ("friendly" in desc_lower or "allowed" in desc_lower):
                property_data["pet_friendly"] = True
            
            property_data["amenities"] = amenities_list
            
            # Square footage - handle commas and decimals
            sqft_patterns = [
                r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:sqft|sq\.?\s*ft|square feet|sq\.?\s*ft\.?)',
                r'built[-\s]?up[-\s]?size[:\s]+(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
                r'size[:\s]+(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:sqft|sq\.?\s*ft|square feet)',
            ]
            for pattern in sqft_patterns:
                sqft_match = re.search(pattern, desc_lower)
                if sqft_match:
                    sqft_str = sqft_match.group(1).replace(',', '')
                    try:
                        property_data["sqft"] = int(float(sqft_str))
                        break
                    except ValueError:
                        pass
        except:
            pass
    
    return property_data

def scrape_rocky_real_estate(url: str, limit: Optional[int] = None) -> List[Dict]:
    """Scrape Rocky Real Estate properties from Bayut company page"""
    print(f"🔍 Starting scraper for Rocky Real Estate...")
    print(f"📄 URL: {url}")
    
    pw = sync_playwright().start()
    browser = pw.firefox.launch(headless=True)
    page = browser.new_page()
    
    print(f"🌐 Navigating to company page...")
    page.goto(url, wait_until="networkidle")
    print(f"✓ Page loaded")
    
    # Wait a bit for dynamic content to load
    page.wait_for_timeout(3000)
    
    # Get all property links from the listing page
    print(f"🔍 Looking for property links...")
    links = page.locator("xpath=//a[contains(@href, 'property/details')]").all()
    
    urls = []
    for link in links:
        href = link.get_attribute("href")
        if href and href not in urls:
            urls.append(href)
    
    print(f"✓ Found {len(urls)} unique property links")
    
    if limit:
        urls = urls[:limit]
        print(f"📊 Limiting to first {limit} properties")
    
    results = []
    
    # Visit each property detail page
    for i, property_url in enumerate(urls, 1):
        print(f"\n{'='*60}")
        print(f"🏠 Property {i}/{len(urls)}")
        print(f"📄 URL: {property_url}")
        print(f"{'='*60}")
        
        try:
            # Navigate to property detail page with timeout handling
            try:
                page.goto(f"https://www.bayut.com{property_url}", wait_until="networkidle", timeout=20000)
            except Exception as nav_error:
                print(f"  ⚠️  Navigation timeout, trying with domcontentloaded...")
                try:
                    page.goto(f"https://www.bayut.com{property_url}", wait_until="domcontentloaded", timeout=15000)
                except:
                    print(f"  ✗ Failed to load property page, skipping...")
                    continue
            page.wait_for_timeout(2000)  # Wait for content to load
            
            # Extract property details
            property_data = extract_property_details(page)
            property_data["url"] = f"https://www.bayut.com{property_url}"
            property_data["property_id"] = f"rocky_{i:03d}"
            
            # Construct embedding text for better semantic search
            embedding_parts = []
            if property_data.get("bedrooms"):
                embedding_parts.append(f"{property_data['bedrooms']} bedroom")
            if property_data.get("property_type"):
                embedding_parts.append(property_data["property_type"].lower())
            if property_data.get("location"):
                embedding_parts.append(f"in {property_data['location']}")
            if property_data.get("sqft"):
                embedding_parts.append(f"{property_data['sqft']} sqft")
            if property_data.get("furnished"):
                if property_data["furnished"] is True:
                    embedding_parts.append("furnished")
                elif property_data["furnished"] is False:
                    embedding_parts.append("unfurnished")
            if property_data.get("monthly_rent"):
                embedding_parts.append(f"AED {property_data['monthly_rent']:.0f} monthly")
            if property_data.get("amenities"):
                embedding_parts.append("with " + ", ".join(property_data["amenities"]))
            
            property_data["embedding_text"] = ", ".join(embedding_parts) if embedding_parts else property_data.get("description", "")
            
            results.append(property_data)
            
            print(f"✓ Extracted: {property_data.get('location', 'N/A')}")
            print(f"  Price: {property_data.get('pricing_raw', 'N/A')}")
            print(f"  Beds/Baths: {property_data.get('bedrooms', 'N/A')}BR / {property_data.get('bathrooms', 'N/A')} Bath")
            
        except Exception as e:
            print(f"✗ Error scraping property {i}: {str(e)}")
            continue
    
    browser.close()
    pw.stop()
    
    print(f"\n{'='*60}")
    print(f"✅ Scraping completed!")
    print(f"📊 Total properties scraped: {len(results)}")
    print(f"{'='*60}")
    
    return results

def save_to_csv(properties: List[Dict], filename: str = "rocky_real_estate_properties.csv"):
    """Save scraped properties to CSV file"""
    if not properties:
        print("⚠️  No properties to save")
        return
    
    # Get all unique keys from all properties
    all_keys = set()
    for prop in properties:
        all_keys.update(prop.keys())
    
    # Define column order (important fields first)
    column_order = [
        "property_id",
        "url",
        "property_manager",
        "pricing_raw",
        "monthly_rent",
        "yearly_rent",
        "currency",
        "rent_frequency",
        "location_raw",
        "location",
        "area",
        "city",
        "bedrooms",
        "bathrooms",
        "sqft",
        "property_type",
        "furnished",
        "parking",
        "parking_spots",
        "amenities",
        "pet_friendly",
        "security_24_7",
        "nearby_metro",
        "metro_distance_km",
        "nearby_shops",
        "shops_distance_km",
        "description",
        "embedding_text"
    ]
    
    # Add any remaining keys
    for key in sorted(all_keys):
        if key not in column_order:
            column_order.append(key)
    
    # Write to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_order, extrasaction='ignore')
        writer.writeheader()
        
        for prop in properties:
            # Convert lists to strings for CSV
            row = prop.copy()
            if isinstance(row.get("amenities"), list):
                row["amenities"] = ", ".join(row["amenities"])
            writer.writerow(row)
    
    print(f"💾 Saved {len(properties)} properties to {filename}")

if __name__ == "__main__":
    # Rocky Real Estate company page URLs (pages 2-5)
    base_url = "https://www.bayut.com/companies/rocky-real-estate-head-office-250/"
    pages = [2, 3, 4, 5]
    
    all_properties = []
    property_counter = 1
    
    print("="*70)
    print("SCRAPING ROCKY REAL ESTATE - PAGES 2-5")
    print("="*70)
    
    # Loop through pages 2-5
    for page_num in pages:
        print(f"\n{'='*70}")
        print(f"📄 PAGE {page_num}")
        print(f"{'='*70}")
        
        page_url = f"{base_url}?page={page_num}"
        
        # Scrape current page
        page_properties = scrape_rocky_real_estate(page_url, limit=None)
        
        if page_properties:
            # Update property IDs to be sequential across all pages
            for prop in page_properties:
                prop["property_id"] = f"rocky_{property_counter:03d}"
                property_counter += 1
            
            all_properties.extend(page_properties)
            print(f"\n✓ Page {page_num}: Scraped {len(page_properties)} properties")
            print(f"  Total so far: {len(all_properties)} properties")
        else:
            print(f"\n⚠️  Page {page_num}: No properties found")
    
    # Save all properties to CSV
    if all_properties:
        save_to_csv(all_properties, "rocky_real_estate_properties.csv")
        print(f"\n{'='*70}")
        print(f"✅ COMPLETE!")
        print(f"{'='*70}")
        print(f"📊 Total properties scraped: {len(all_properties)}")
        print(f"💾 Saved to 'rocky_real_estate_properties.csv'")
        print(f"{'='*70}")
    else:
        print("\n⚠️  No properties were scraped. Check the URLs and try again.")

