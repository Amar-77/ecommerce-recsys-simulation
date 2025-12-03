import pandas as pd
import json
import os

# --- CONFIGURATION ---
csv_path = "ecommerce_medium_clean.csv"

def generate_diverse_catalog():
    if not os.path.exists(csv_path):
        print(f"❌ Error: Could not find '{csv_path}'.")
        return

    print(f"⏳ Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. GROUP BY PRODUCT (Create the Inventory)
    inventory = df.groupby('product_id').agg({
        'brand': 'first',
        'category_code': 'first',
        'price': 'mean',
        'event_type': 'count'
    }).reset_index()
    
    inventory.rename(columns={'event_type': 'popularity'}, inplace=True)

    # 2. EXTRACT MAIN CATEGORY
    # Turn "electronics.smartphone" -> "electronics"
    # Turn "appliances.kitchen.washer" -> "appliances"
    def get_main_cat(text):
        if str(text) == 'nan' or 'unknown' in str(text): return 'Other'
        return str(text).split('.')[0].title()

    inventory['main_category'] = inventory['category_code'].apply(get_main_cat)

    # 3. SELECT TOP ITEMS PER CATEGORY (The Diversity Logic)
    website_data = []
    
    # Get list of unique main categories (e.g., Electronics, Appliances, Computers, Furniture)
    unique_cats = inventory['main_category'].unique()
    
    print("\n📦 BUILDING DIVERSE CATALOG:")
    print(f"{'CATEGORY':<15} | {'PRODUCT NAME':<30} | {'PRICE'}")
    print("-" * 65)

    for cat in unique_cats:
        if cat == 'Other': continue # Skip junk for now
        
        # Get items in this category
        cat_items = inventory[inventory['main_category'] == cat]
        
        # Sort by popularity and take Top 3
        top_5 = cat_items.sort_values('popularity', ascending=False).head(5)
        
        for _, row in top_5.iterrows():
            # Format Name
            brand = str(row['brand']).title()
            if brand in ['Nan', 'Generic']: brand = ""
            sub_cat = str(row['category_code']).split('.')[-1].replace('_', ' ').title()
            
            display_name = f"{brand} {sub_cat}".strip()
            
            # Add to list
            item_data = {
                "id": int(row['product_id']),
                "name": display_name,
                "price": round(row['price'], 2),
                "category": cat, # Main category (for filtering UI)
                "popularity": int(row['popularity'])
            }
            website_data.append(item_data)
            
            print(f"{cat:<15} | {display_name:<30} | ${row['price']:.0f}")

    # 4. SAVE TO JSON
    with open('website_inventory.json', 'w') as f:
        json.dump(website_data, f, indent=4)

    print("\n✅ SUCCESS! Saved diverse items to 'website_inventory.json'.")
    print("   Open this file to pick 3 totally different items for your demo!")

if __name__ == "__main__":
    generate_diverse_catalog()