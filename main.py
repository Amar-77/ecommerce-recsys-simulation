import pandas as pd
import numpy as np
import implicit
from scipy.sparse import coo_matrix, csr_matrix # <--- Added csr_matrix import
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import time
from datetime import datetime

# --- CONFIGURATION ---
DATA_FILE = "ecommerce_medium_clean.csv" 
LOG_FILE = "session_logs.json"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL STATE ---
model = None
user_to_idx = {}
item_to_idx = {}
idx_to_item = {}
product_lookup = {} 
train_matrix = None # Global matrix storage

# --- CORE LOGIC ---
def train_model():
    """
    Rebuilds the Model and the Matrix from CSV + Logs
    """
    global model, user_to_idx, item_to_idx, idx_to_item, product_lookup, train_matrix
    
    print("⏳ Loading Data Pipeline...")
    
    # 1. Load History
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: Could not find {DATA_FILE}")
        return

    # 2. Merge Logs
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
                if logs:
                    print(f"   -> Merging {len(logs)} new interactions...")
                    log_df = pd.DataFrame(logs)
                    weight_map = {'view': 1.0, 'cart': 10.0, 'purchase': 50.0}
                    log_df['weight'] = log_df['event_type'].map(weight_map)
                    df = pd.concat([df, log_df], ignore_index=True)
        except Exception as e:
            print(f"⚠️ Error reading logs: {e}")

    # 3. Update Product Lookup
    print("   -> Indexing Products....")
    product_lookup = {}
    for _, row in df.iterrows():
        pid = int(row['product_id'])
        if pid not in product_lookup and pd.notna(row.get('brand')):
            brand = str(row['brand']).title()
            if brand in ['Nan', 'Generic']: brand = ""
            cat = str(row['category_code']).split('.')[-1].title().replace('_', ' ')
            price = row['price'] if pd.notna(row['price']) else 0.0
            
            product_lookup[pid] = {"name": f"{brand} {cat}".strip(), "price": price, "category": cat}

    # 4. Build Matrix
    print("   -> Building Matrix.......")
    df['user_idx'] = df['user_id'].astype("category").cat.codes
    df['product_idx'] = df['product_id'].astype("category").cat.codes
    
    user_to_idx = dict(zip(df['user_id'], df['user_idx']))
    item_to_idx = dict(zip(df['product_id'], df['product_idx']))
    idx_to_item = dict(zip(df['product_idx'], df['product_id']))
    
    # Store matrix GLOBALLY
    train_matrix = coo_matrix(
        (df['weight'].astype(float), (df['user_idx'], df['product_idx'])),
        shape=(len(user_to_idx), len(item_to_idx))
    ).tocsr()
    
    # 5. Train
    print("   -> Training Model......")
    model = implicit.als.AlternatingLeastSquares(factors=32, iterations=10, random_state=42)
    model.fit(train_matrix * 10)
    print("✅ Model Updated & Ready!")

# --- API ENDPOINTS ---

@app.on_event("startup")
def startup_event():
    train_model()

@app.get("/")
def home():
    return {"status": "Active"}

class UserAction(BaseModel):
    user_id: int
    product_id: int
    event_type: str

@app.post("/log_action")
def log_action(action: UserAction):
    new_entry = action.dict()
    new_entry['event_time'] = str(datetime.now()) 
    
    current_logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            try: current_logs = json.load(f)
            except: pass
            
    current_logs.append(new_entry)
    with open(LOG_FILE, 'w') as f:
        json.dump(current_logs, f, indent=4)
        
    return {"status": "Logged", "total_logs": len(current_logs)}

@app.post("/trigger_retrain")
def trigger_retrain():
    start = time.time()
    train_model()
    duration = time.time() - start
    return {"status": "Retrained", "duration": f"{duration:.2f}s"}

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int):
    # 1. Cold Start
    if user_id not in user_to_idx:
        return {"type": "Popular (New User)", "items": []}
    
    # 2. Get Recommendations
    user_idx = user_to_idx[user_id]
    
    # --- CRITICAL FIX HERE ---
    # We explicitly convert the row to CSR format to satisfy the library requirement
    user_items = train_matrix[user_idx]
    if not isinstance(user_items, csr_matrix):
        user_items = user_items.tocsr()
    
    # Generate Recs
    ids, scores = model.recommend(user_idx, user_items=user_items, N=5)
    
    # 3. Format Results
    results = []
    for idx in ids:
        pid = idx_to_item[idx]
        details = product_lookup.get(pid)
        if details:
            results.append({"id": pid, "name": details['name'], "price": f"${details['price']:.2f}", "category": details['category']})
        else:
             results.append({"id": pid, "name": f"Item #{pid}", "price": "Unknown", "category": "General"})
        
    return {"type": "Personalized (AI)", "items": results}