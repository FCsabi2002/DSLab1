import os
from datetime import datetime
from typing import Annotated, List

import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="Supply Chain Decision Support API")

# ============================================================
# CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later you can replace with your exact SAC domain
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LOAD MODELS
# ============================================================
best_demand_model = joblib.load("demand_model.pkl")
best_delay_model = joblib.load("delay_model.pkl")

# ============================================================
# FILE PATH FOR SCORED RESULTS
# ============================================================
SCORED_RESULTS_FILE = "scored_results.csv"


DB_FILE = Path("predictions.db")


def get_db_connection():
    print("USING DB FILE:", DB_FILE.resolve())
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ScoredResults (
            record_id INTEGER PRIMARY KEY,
            order_year INTEGER,
            order_month INTEGER,
            category_name TEXT,
            shipping_mode TEXT,
            market TEXT,
            order_region TEXT,
            department TEXT,
            customer_segment TEXT,
            payment_type TEXT,
            predicted_demand REAL,
            category_avg REAL,
            inventory_decision TEXT,
            delay_risk REAL,
            delay_decision TEXT,
            scored_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_results_to_db(results: list[dict]):
    conn = get_db_connection()
    cur = conn.cursor()

    for row in results:
        cur.execute("""
            INSERT OR REPLACE INTO ScoredResults (
                record_id,
                order_year,
                order_month,
                category_name,
                shipping_mode,
                market,
                order_region,
                department,
                customer_segment,
                payment_type,
                predicted_demand,
                category_avg,
                inventory_decision,
                delay_risk,
                delay_decision,
                scored_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["record_id"],
            row["order_year"],
            row["order_month"],
            row["category_name"],
            row["shipping_mode"],
            row["market"],
            row["order_region"],
            row["department"],
            row["customer_segment"],
            row["payment_type"],
            row["predicted_demand"],
            row["category_avg"],
            row["inventory_decision"],
            row["delay_risk"],
            row["delay_decision"],
            row["scored_at"],
        ))

    conn.commit()
    conn.close()
    
# ============================================================
# TYPE DEFINITIONS
# ============================================================
Uint8 = Annotated[int, Field(0, ge=0, le=255)]
zipcode_type = Annotated[int, Field(0, ge=0, le=100000)]

# ============================================================
# INPUT SCHEMAS
# ============================================================

class DataInput(BaseModel):
    name: str


class DemandInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    order_year: int
    order_month: int
    avg_price: float
    order_count: int

    as_seen_on_tv: Uint8 = Field(0, alias="Category Name_As Seen on  TV!")
    baby: Uint8 = Field(0, alias="Category Name_Baby ")
    baseball_softball: Uint8 = Field(0, alias="Category Name_Baseball & Softball")
    basketball: Uint8 = Field(0, alias="Category Name_Basketball")
    books: Uint8 = Field(0, alias="Category Name_Books ")
    boxing_mma: Uint8 = Field(0, alias="Category Name_Boxing & MMA")
    cds: Uint8 = Field(0, alias="Category Name_CDs ")
    cameras: Uint8 = Field(0, alias="Category Name_Cameras ")
    camping_hiking: Uint8 = Field(0, alias="Category Name_Camping & Hiking")
    cardio_equipment: Uint8 = Field(0, alias="Category Name_Cardio Equipment")
    childrens_clothing: Uint8 = Field(0, alias="Category Name_Children's Clothing")
    cleats: Uint8 = Field(0, alias="Category Name_Cleats")
    computers: Uint8 = Field(0, alias="Category Name_Computers")
    consumer_electronics: Uint8 = Field(0, alias="Category Name_Consumer Electronics")
    crafts: Uint8 = Field(0, alias="Category Name_Crafts")
    dvds: Uint8 = Field(0, alias="Category Name_DVDs")
    electronics: Uint8 = Field(0, alias="Category Name_Electronics")
    fishing: Uint8 = Field(0, alias="Category Name_Fishing")
    fitness_accessories: Uint8 = Field(0, alias="Category Name_Fitness Accessories")
    garden: Uint8 = Field(0, alias="Category Name_Garden")
    girls_apparel: Uint8 = Field(0, alias="Category Name_Girls' Apparel")
    golf_apparel: Uint8 = Field(0, alias="Category Name_Golf Apparel")
    golf_bags_carts: Uint8 = Field(0, alias="Category Name_Golf Bags & Carts")
    golf_balls: Uint8 = Field(0, alias="Category Name_Golf Balls")
    golf_gloves: Uint8 = Field(0, alias="Category Name_Golf Gloves")
    golf_shoes: Uint8 = Field(0, alias="Category Name_Golf Shoes")
    health_beauty: Uint8 = Field(0, alias="Category Name_Health and Beauty")
    hockey: Uint8 = Field(0, alias="Category Name_Hockey")
    hunting_shooting: Uint8 = Field(0, alias="Category Name_Hunting & Shooting")
    indoor_outdoor_games: Uint8 = Field(0, alias="Category Name_Indoor/Outdoor Games")
    kids_golf_clubs: Uint8 = Field(0, alias="Category Name_Kids' Golf Clubs")
    lacrosse: Uint8 = Field(0, alias="Category Name_Lacrosse")
    mens_clothing: Uint8 = Field(0, alias="Category Name_Men's Clothing")
    mens_footwear: Uint8 = Field(0, alias="Category Name_Men's Footwear")
    mens_golf_clubs: Uint8 = Field(0, alias="Category Name_Men's Golf Clubs")
    music: Uint8 = Field(0, alias="Category Name_Music")
    pet_supplies: Uint8 = Field(0, alias="Category Name_Pet Supplies")
    shop_by_sport: Uint8 = Field(0, alias="Category Name_Shop By Sport")
    soccer: Uint8 = Field(0, alias="Category Name_Soccer")
    sporting_goods: Uint8 = Field(0, alias="Category Name_Sporting Goods")
    strength_training: Uint8 = Field(0, alias="Category Name_Strength Training")
    tennis_racquet: Uint8 = Field(0, alias="Category Name_Tennis & Racquet")
    toys: Uint8 = Field(0, alias="Category Name_Toys")
    trade_in: Uint8 = Field(0, alias="Category Name_Trade-In")
    video_games: Uint8 = Field(0, alias="Category Name_Video Games")
    water_sports: Uint8 = Field(0, alias="Category Name_Water Sports")
    womens_apparel: Uint8 = Field(0, alias="Category Name_Women's Apparel")
    womens_clothing: Uint8 = Field(0, alias="Category Name_Women's Clothing")
    womens_golf_clubs: Uint8 = Field(0, alias="Category Name_Women's Golf Clubs")


class DelayInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cust_zip: zipcode_type = Field(0, alias="Customer Zipcode")
    order_year: int
    order_month: int
    order_week_day: int
    order_hour: int

    ship_mode_same_day: Uint8 = Field(0, alias="Shipping Mode_Same Day")
    ship_mode_second_class: Uint8 = Field(0, alias="Shipping Mode_Second Class")
    ship_mode_standard_class: Uint8 = Field(0, alias="Shipping Mode_Standard Class")

    market_europe: Uint8 = Field(0, alias="Market_Europe")
    market_latam: Uint8 = Field(0, alias="Market_LATAM")
    market_pacific_asia: Uint8 = Field(0, alias="Market_Pacific Asia")
    market_usca: Uint8 = Field(0, alias="Market_USCA")

    region_caribbean: Uint8 = Field(0, alias="Order Region_Caribbean")
    region_central_africa: Uint8 = Field(0, alias="Order Region_Central Africa")
    region_central_america: Uint8 = Field(0, alias="Order Region_Central America")
    region_central_asia: Uint8 = Field(0, alias="Order Region_Central Asia")
    region_east_africa: Uint8 = Field(0, alias="Order Region_East Africa")
    region_east_usa: Uint8 = Field(0, alias="Order Region_East of USA")
    region_eastern_asia: Uint8 = Field(0, alias="Order Region_Eastern Asia")
    region_eastern_europe: Uint8 = Field(0, alias="Order Region_Eastern Europe")
    region_north_africa: Uint8 = Field(0, alias="Order Region_North Africa")
    region_northern_europe: Uint8 = Field(0, alias="Order Region_Northern Europe")
    region_oceania: Uint8 = Field(0, alias="Order Region_Oceania")
    region_south_america: Uint8 = Field(0, alias="Order Region_South America")
    region_south_asia: Uint8 = Field(0, alias="Order Region_South Asia")
    region_south_usa: Uint8 = Field(0, alias="Order Region_South of  USA ")
    region_southeast_asia: Uint8 = Field(0, alias="Order Region_Southeast Asia")
    region_southern_africa: Uint8 = Field(0, alias="Order Region_Southern Africa")
    region_southern_europe: Uint8 = Field(0, alias="Order Region_Southern Europe")
    region_us_center: Uint8 = Field(0, alias="Order Region_US Center ")
    region_west_africa: Uint8 = Field(0, alias="Order Region_West Africa")
    region_west_asia: Uint8 = Field(0, alias="Order Region_West Asia")
    region_west_usa: Uint8 = Field(0, alias="Order Region_West of USA ")
    region_western_europe: Uint8 = Field(0, alias="Order Region_Western Europe")

    cat_as_seen_on_tv: Uint8 = Field(0, alias="Category Name_As Seen on  TV!")
    cat_baby: Uint8 = Field(0, alias="Category Name_Baby ")
    cat_baseball_softball: Uint8 = Field(0, alias="Category Name_Baseball & Softball")
    cat_basketball: Uint8 = Field(0, alias="Category Name_Basketball")
    cat_books: Uint8 = Field(0, alias="Category Name_Books ")
    cat_boxing_mma: Uint8 = Field(0, alias="Category Name_Boxing & MMA")
    cat_cds: Uint8 = Field(0, alias="Category Name_CDs ")
    cat_cameras: Uint8 = Field(0, alias="Category Name_Cameras ")
    cat_camping_hiking: Uint8 = Field(0, alias="Category Name_Camping & Hiking")
    cat_cardio_equipment: Uint8 = Field(0, alias="Category Name_Cardio Equipment")
    cat_childrens_clothing: Uint8 = Field(0, alias="Category Name_Children's Clothing")
    cat_cleats: Uint8 = Field(0, alias="Category Name_Cleats")
    cat_computers: Uint8 = Field(0, alias="Category Name_Computers")
    cat_consumer_electronics: Uint8 = Field(0, alias="Category Name_Consumer Electronics")
    cat_crafts: Uint8 = Field(0, alias="Category Name_Crafts")
    cat_dvds: Uint8 = Field(0, alias="Category Name_DVDs")
    cat_electronics: Uint8 = Field(0, alias="Category Name_Electronics")
    cat_fishing: Uint8 = Field(0, alias="Category Name_Fishing")
    cat_fitness_accessories: Uint8 = Field(0, alias="Category Name_Fitness Accessories")
    cat_garden: Uint8 = Field(0, alias="Category Name_Garden")
    cat_girls_apparel: Uint8 = Field(0, alias="Category Name_Girls' Apparel")
    cat_golf_apparel: Uint8 = Field(0, alias="Category Name_Golf Apparel")
    cat_golf_bags_carts: Uint8 = Field(0, alias="Category Name_Golf Bags & Carts")
    cat_golf_balls: Uint8 = Field(0, alias="Category Name_Golf Balls")
    cat_golf_gloves: Uint8 = Field(0, alias="Category Name_Golf Gloves")
    cat_golf_shoes: Uint8 = Field(0, alias="Category Name_Golf Shoes")
    cat_health_beauty: Uint8 = Field(0, alias="Category Name_Health and Beauty")
    cat_hockey: Uint8 = Field(0, alias="Category Name_Hockey")
    cat_hunting_shooting: Uint8 = Field(0, alias="Category Name_Hunting & Shooting")
    cat_indoor_outdoor_games: Uint8 = Field(0, alias="Category Name_Indoor/Outdoor Games")
    cat_kids_golf_clubs: Uint8 = Field(0, alias="Category Name_Kids' Golf Clubs")
    cat_lacrosse: Uint8 = Field(0, alias="Category Name_Lacrosse")
    cat_mens_clothing: Uint8 = Field(0, alias="Category Name_Men's Clothing")
    cat_mens_footwear: Uint8 = Field(0, alias="Category Name_Men's Footwear")
    cat_mens_golf_clubs: Uint8 = Field(0, alias="Category Name_Men's Golf Clubs")
    cat_music: Uint8 = Field(0, alias="Category Name_Music")
    cat_pet_supplies: Uint8 = Field(0, alias="Category Name_Pet Supplies")
    cat_shop_by_sport: Uint8 = Field(0, alias="Category Name_Shop By Sport")
    cat_soccer: Uint8 = Field(0, alias="Category Name_Soccer")
    cat_sporting_goods: Uint8 = Field(0, alias="Category Name_Sporting Goods")
    cat_strength_training: Uint8 = Field(0, alias="Category Name_Strength Training")
    cat_tennis_racquet: Uint8 = Field(0, alias="Category Name_Tennis & Racquet")
    cat_toys: Uint8 = Field(0, alias="Category Name_Toys")
    cat_trade_in: Uint8 = Field(0, alias="Category Name_Trade-In")
    cat_video_games: Uint8 = Field(0, alias="Category Name_Video Games")
    cat_water_sports: Uint8 = Field(0, alias="Category Name_Water Sports")
    cat_womens_apparel: Uint8 = Field(0, alias="Category Name_Women's Apparel")
    cat_womens_clothing: Uint8 = Field(0, alias="Category Name_Women's Clothing")
    cat_womens_golf_clubs: Uint8 = Field(0, alias="Category Name_Women's Golf Clubs")

    dept_book_shop: Uint8 = Field(0, alias="Department Name_Book Shop")
    dept_discs_shop: Uint8 = Field(0, alias="Department Name_Discs Shop")
    dept_fan_shop: Uint8 = Field(0, alias="Department Name_Fan Shop")
    dept_fitness: Uint8 = Field(0, alias="Department Name_Fitness")
    dept_footwear: Uint8 = Field(0, alias="Department Name_Footwear")
    dept_golf: Uint8 = Field(0, alias="Department Name_Golf")
    dept_health_beauty: Uint8 = Field(0, alias="Department Name_Health and Beauty ")
    dept_outdoors: Uint8 = Field(0, alias="Department Name_Outdoors")
    dept_pet_shop: Uint8 = Field(0, alias="Department Name_Pet Shop")
    dept_technology: Uint8 = Field(0, alias="Department Name_Technology")

    segment_corporate: Uint8 = Field(0, alias="Customer Segment_Corporate")
    segment_home_office: Uint8 = Field(0, alias="Customer Segment_Home Office")

    type_debit: Uint8 = Field(0, alias="Type_DEBIT")
    type_payment: Uint8 = Field(0, alias="Type_PAYMENT")
    type_transfer: Uint8 = Field(0, alias="Type_TRANSFER")


class DemandSimpleInput(BaseModel):
    order_year: int
    order_month: int
    avg_price: float
    order_count: int
    category_name: str


class DelaySimpleInput(BaseModel):
    customer_zipcode: int = 0
    order_year: int
    order_month: int
    order_week_day: int
    order_hour: int
    shipping_mode: str
    market: str
    order_region: str
    category_name: str
    department: str
    customer_segment: str = "Consumer"
    payment_type: str = "TRANSFER"


class CombinedSimpleInput(BaseModel):
    record_id: int
    order_year: int
    order_month: int
    avg_price: float
    order_count: int
    order_week_day: int
    order_hour: int
    customer_zipcode: int = 0
    shipping_mode: str
    market: str
    order_region: str
    category_name: str
    department: str
    customer_segment: str = "Consumer"
    payment_type: str = "TRANSFER"


# ============================================================
# CONSTANTS FOR ENCODING
# ============================================================

DEMAND_CATEGORY_COLUMNS = [
    "Category Name_As Seen on  TV!",
    "Category Name_Baby ",
    "Category Name_Baseball & Softball",
    "Category Name_Basketball",
    "Category Name_Books ",
    "Category Name_Boxing & MMA",
    "Category Name_CDs ",
    "Category Name_Cameras ",
    "Category Name_Camping & Hiking",
    "Category Name_Cardio Equipment",
    "Category Name_Children's Clothing",
    "Category Name_Cleats",
    "Category Name_Computers",
    "Category Name_Consumer Electronics",
    "Category Name_Crafts",
    "Category Name_DVDs",
    "Category Name_Electronics",
    "Category Name_Fishing",
    "Category Name_Fitness Accessories",
    "Category Name_Garden",
    "Category Name_Girls' Apparel",
    "Category Name_Golf Apparel",
    "Category Name_Golf Bags & Carts",
    "Category Name_Golf Balls",
    "Category Name_Golf Gloves",
    "Category Name_Golf Shoes",
    "Category Name_Health and Beauty",
    "Category Name_Hockey",
    "Category Name_Hunting & Shooting",
    "Category Name_Indoor/Outdoor Games",
    "Category Name_Kids' Golf Clubs",
    "Category Name_Lacrosse",
    "Category Name_Men's Clothing",
    "Category Name_Men's Footwear",
    "Category Name_Men's Golf Clubs",
    "Category Name_Music",
    "Category Name_Pet Supplies",
    "Category Name_Shop By Sport",
    "Category Name_Soccer",
    "Category Name_Sporting Goods",
    "Category Name_Strength Training",
    "Category Name_Tennis & Racquet",
    "Category Name_Toys",
    "Category Name_Trade-In",
    "Category Name_Video Games",
    "Category Name_Water Sports",
    "Category Name_Women's Apparel",
    "Category Name_Women's Clothing",
    "Category Name_Women's Golf Clubs",
]

DELAY_CATEGORY_COLUMNS = DEMAND_CATEGORY_COLUMNS.copy()

DELAY_REGION_COLUMNS = [
    "Order Region_Caribbean",
    "Order Region_Central Africa",
    "Order Region_Central America",
    "Order Region_Central Asia",
    "Order Region_East Africa",
    "Order Region_East of USA",
    "Order Region_Eastern Asia",
    "Order Region_Eastern Europe",
    "Order Region_North Africa",
    "Order Region_Northern Europe",
    "Order Region_Oceania",
    "Order Region_South America",
    "Order Region_South Asia",
    "Order Region_South of  USA ",
    "Order Region_Southeast Asia",
    "Order Region_Southern Africa",
    "Order Region_Southern Europe",
    "Order Region_US Center ",
    "Order Region_West Africa",
    "Order Region_West Asia",
    "Order Region_West of USA ",
    "Order Region_Western Europe",
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_demand_features_simple(input_data: DemandSimpleInput) -> pd.DataFrame:
    base = {
        "order_year": input_data.order_year,
        "order_month": input_data.order_month,
        "avg_price": input_data.avg_price,
        "order_count": input_data.order_count,
    }

    for col in DEMAND_CATEGORY_COLUMNS:
        base[col] = 0

    cat_key = "Category Name_" + input_data.category_name
    if cat_key in base:
        base[cat_key] = 1

    return pd.DataFrame([base])
    
def align_to_model_features(df: pd.DataFrame, model) -> pd.DataFrame:
    if not hasattr(model, "feature_names_in_"):
        return df

    expected = list(model.feature_names_in_)

    for col in expected:
        if col not in df.columns:
            df[col] = 0

    # keep only model columns and in exact order
    df = df[expected]
    return df


def build_delay_features_simple(input_data: DelaySimpleInput) -> pd.DataFrame:
    base = {
        "Customer Zipcode": input_data.customer_zipcode,
        "order_year": input_data.order_year,
        "order_month": input_data.order_month,
        "order_week_day": input_data.order_week_day,
        "order_hour": input_data.order_hour,
        "Shipping Mode_Same Day": 1 if input_data.shipping_mode == "Same Day" else 0,
        "Shipping Mode_Second Class": 1 if input_data.shipping_mode == "Second Class" else 0,
        "Shipping Mode_Standard Class": 1 if input_data.shipping_mode == "Standard Class" else 0,
        "Market_Europe": 1 if input_data.market == "Europe" else 0,
        "Market_LATAM": 1 if input_data.market == "LATAM" else 0,
        "Market_Pacific Asia": 1 if input_data.market == "Pacific Asia" else 0,
        "Market_USCA": 1 if input_data.market == "USCA" else 0,
        "Department Name_Book Shop": 1 if input_data.department == "Book Shop" else 0,
        "Department Name_Discs Shop": 1 if input_data.department == "Discs Shop" else 0,
        "Department Name_Fan Shop": 1 if input_data.department == "Fan Shop" else 0,
        "Department Name_Fitness": 1 if input_data.department == "Fitness" else 0,
        "Department Name_Footwear": 1 if input_data.department == "Footwear" else 0,
        "Department Name_Golf": 1 if input_data.department == "Golf" else 0,
        "Department Name_Health and Beauty ": 1 if input_data.department == "Health and Beauty" else 0,
        "Department Name_Outdoors": 1 if input_data.department == "Outdoors" else 0,
        "Department Name_Pet Shop": 1 if input_data.department == "Pet Shop" else 0,
        "Department Name_Technology": 1 if input_data.department == "Technology" else 0,
        "Customer Segment_Corporate": 1 if input_data.customer_segment == "Corporate" else 0,
        "Customer Segment_Home Office": 1 if input_data.customer_segment == "Home Office" else 0,
        "Type_DEBIT": 1 if input_data.payment_type == "DEBIT" else 0,
        "Type_PAYMENT": 1 if input_data.payment_type == "PAYMENT" else 0,
        "Type_TRANSFER": 1 if input_data.payment_type == "TRANSFER" else 0,
    }

    for col in DELAY_REGION_COLUMNS:
        base[col] = 0
    for col in DELAY_CATEGORY_COLUMNS:
        base[col] = 0

    region_key = "Order Region_" + input_data.order_region
    if region_key in base:
        base[region_key] = 1

    cat_key = "Category Name_" + input_data.category_name
    if cat_key in base:
        base[cat_key] = 1

    return pd.DataFrame([base])


def score_demand_simple(input_data: DemandSimpleInput) -> dict:
    features = build_demand_features_simple(input_data)
    features = align_to_model_features(features, best_demand_model)
    predicted_demand = best_demand_model.predict(features)[0]
    category_avg = input_data.avg_price

    if predicted_demand > category_avg * 1.2:
        decision = "Increase Stock"
    elif predicted_demand < category_avg * 0.8:
        decision = "Reduce Stock"
    else:
        decision = "Maintain"

    return {
        "predicted_demand": round(float(predicted_demand), 2),
        "category_avg": round(float(category_avg), 2),
        "inventory_decision": decision,
    }


def score_delay_simple(input_data: DelaySimpleInput) -> dict:
    features = build_delay_features_simple(input_data)
    features = align_to_model_features(features, best_delay_model)
    delay_risk = best_delay_model.predict_proba(features)[0][1]

    if delay_risk > 0.7:
        decision = "Expedite Shipment"
    elif delay_risk > 0.4:
        decision = "Monitor Closely"
    else:
        decision = "No Action"

    return {
        "delay_risk": round(float(delay_risk), 3),
        "delay_decision": decision,
    }

init_db()
# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Supply Chain Decision Support API",
        "endpoints": [
            "/predict/demand",
            "/predict/delay",
            "/predict/demand/simple",
            "/predict/delay/simple",
            "/score-batch",
            "/scored-results",
            "/health",
        ],
    }


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/predict/demand")
def predict_demand(input_data: DemandInput):
    features = pd.DataFrame([input_data.model_dump(by_alias=True)])
    predicted_demand = best_demand_model.predict(features)[0]
    category_avg = input_data.avg_price

    if predicted_demand > category_avg * 1.2:
        decision = "Increase Stock"
    elif predicted_demand < category_avg * 0.8:
        decision = "Reduce Stock"
    else:
        decision = "Maintain"

    return {
        "predicted_demand": round(float(predicted_demand), 2),
        "category_avg": round(float(category_avg), 2),
        "inventory_decision": decision,
    }


@app.post("/predict/delay")
def predict_delay(input_data: DelayInput):
    features = pd.DataFrame([input_data.model_dump(by_alias=True)])
    delay_risk = best_delay_model.predict_proba(features)[0][1]

    if delay_risk > 0.7:
        decision = "Expedite Shipment"
    elif delay_risk > 0.4:
        decision = "Monitor Closely"
    else:
        decision = "No Action"

    return {
        "delay_risk": round(float(delay_risk), 3),
        "delay_decision": decision,
    }


@app.post("/predict/demand/simple")
def predict_demand_simple(input_data: DemandSimpleInput):
    result = score_demand_simple(input_data)
    return {
        **result,
        "category_name": input_data.category_name,
        "order_year": input_data.order_year,
        "order_month": input_data.order_month,
    }


@app.post("/predict/delay/simple")
def predict_delay_simple(input_data: DelaySimpleInput):
    result = score_delay_simple(input_data)
    return {
        **result,
        "shipping_mode": input_data.shipping_mode,
        "order_region": input_data.order_region,
        "category_name": input_data.category_name,
    }


@app.post("/score-batch")
def score_batch(rows: List[CombinedSimpleInput]):
    if not rows:
        raise HTTPException(status_code=400, detail="No rows provided")

    results = []

    for row in rows:
        demand_input = DemandSimpleInput(
            order_year=row.order_year,
            order_month=row.order_month,
            avg_price=row.avg_price,
            order_count=row.order_count,
            category_name=row.category_name,
        )

        delay_input = DelaySimpleInput(
            order_year=row.order_year,
            order_month=row.order_month,
            order_week_day=row.order_week_day,
            order_hour=row.order_hour,
            customer_zipcode=row.customer_zipcode,
            shipping_mode=row.shipping_mode,
            market=row.market,
            order_region=row.order_region,
            category_name=row.category_name,
            department=row.department,
            customer_segment=row.customer_segment,
            payment_type=row.payment_type,
        )

        demand_result = score_demand_simple(demand_input)
        delay_result = score_delay_simple(delay_input)

        results.append({
            "record_id": row.record_id,
            "order_year": row.order_year,
            "order_month": row.order_month,
            "category_name": row.category_name,
            "shipping_mode": row.shipping_mode,
            "market": row.market,
            "order_region": row.order_region,
            "department": row.department,
            "customer_segment": row.customer_segment,
            "payment_type": row.payment_type,
            "predicted_demand": demand_result["predicted_demand"],
            "category_avg": demand_result["category_avg"],
            "inventory_decision": demand_result["inventory_decision"],
            "delay_risk": delay_result["delay_risk"],
            "delay_decision": delay_result["delay_decision"],
            "scored_at": datetime.utcnow().isoformat(),
        })

    df = pd.DataFrame(results)
    df.to_csv(SCORED_RESULTS_FILE, index=False)
    save_results_to_db(results)

    return {
        "status": "success",
        "rows_scored": len(results),
        "output_file": SCORED_RESULTS_FILE,
    }


@app.get("/scored-results")
def get_scored_results():
    if not os.path.exists(SCORED_RESULTS_FILE):
        raise HTTPException(status_code=404, detail="No scored results found")

    df = pd.read_csv(SCORED_RESULTS_FILE)
    return {
        "status": "success",
        "row_count": len(df),
        "data": df.to_dict(orient="records"),
    }

@app.get("/scored-results-flat")
def get_scored_results_flat():
    if not os.path.exists(SCORED_RESULTS_FILE):
        raise HTTPException(status_code=404, detail="No scored results found")
    df = pd.read_csv(SCORED_RESULTS_FILE)
    return df.to_dict(orient="records")
    
@app.get("/scored-results-db")
def get_scored_results_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM ScoredResults ORDER BY record_id")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    return {
        "status": "success",
        "row_count": len(rows),
        "data": rows,
    }

@app.post("/ping")
def ping(item: DataInput):
    return {"Hello": item.name, "Status": "Live on BTP"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

# import os
# from fastapi import FastAPI
# import uvicorn
# from pydantic import BaseModel
# import pandas as pd
# from pydantic import Field, ConfigDict
# from typing import Annotated

# app = FastAPI()

# class DataInput(BaseModel):
    # name: str

# import joblib
# best_demand_model = joblib.load("demand_model.pkl")
# best_delay_model = joblib.load("delay_model.pkl")

# Uint8 = Annotated[int, Field(0, ge=0, le=255)]
# zipcode_type = Annotated[int, Field(0, ge=0, le=100000)]

# class DemandInput(BaseModel):
    # model_config = ConfigDict(populate_by_name=True)
    # order_year:   int
    # order_month:  int
    # avg_price:    float
    # order_count:  int

    # #accessories: Uint8 = Field(alias="Category Name_Accessories")
    # as_seen_on_tv: Uint8 = Field(alias="Category Name_As Seen on  TV!")
    # baby: Uint8 = Field(alias="Category Name_Baby ")
    # baseball_softball: Uint8 = Field(alias="Category Name_Baseball & Softball")
    # basketball: Uint8 = Field(alias="Category Name_Basketball")
    # books: Uint8 = Field(alias="Category Name_Books ")
    # boxing_mma: Uint8 = Field(alias="Category Name_Boxing & MMA")
    # cds: Uint8 = Field(alias="Category Name_CDs ")
    # cameras: Uint8 = Field(alias="Category Name_Cameras ")
    # camping_hiking: Uint8 = Field(alias="Category Name_Camping & Hiking")
    # cardio_equipment: Uint8 = Field(alias="Category Name_Cardio Equipment")
    # childrens_clothing: Uint8 = Field(alias="Category Name_Children's Clothing")
    # cleats: Uint8 = Field(alias="Category Name_Cleats")
    # computers: Uint8 = Field(alias="Category Name_Computers")
    # consumer_electronics: Uint8 = Field(alias="Category Name_Consumer Electronics")
    # crafts: Uint8 = Field(alias="Category Name_Crafts")
    # dvds: Uint8 = Field(alias="Category Name_DVDs")
    # electronics: Uint8 = Field(alias="Category Name_Electronics")
    # fishing: Uint8 = Field(alias="Category Name_Fishing")
    # fitness_accessories: Uint8 = Field(alias="Category Name_Fitness Accessories")
    # garden: Uint8 = Field(alias="Category Name_Garden")
    # girls_apparel: Uint8 = Field(alias="Category Name_Girls' Apparel")
    # golf_apparel: Uint8 = Field(alias="Category Name_Golf Apparel")
    # golf_bags_carts: Uint8 = Field(alias="Category Name_Golf Bags & Carts")
    # golf_balls: Uint8 = Field(alias="Category Name_Golf Balls")
    # golf_gloves: Uint8 = Field(alias="Category Name_Golf Gloves")
    # golf_shoes: Uint8 = Field(alias="Category Name_Golf Shoes")
    # health_beauty: Uint8 = Field(alias="Category Name_Health and Beauty")
    # hockey: Uint8 = Field(alias="Category Name_Hockey")
    # hunting_shooting: Uint8 = Field(alias="Category Name_Hunting & Shooting")
    # indoor_outdoor_games: Uint8 = Field(alias="Category Name_Indoor/Outdoor Games")
    # kids_golf_clubs: Uint8 = Field(alias="Category Name_Kids' Golf Clubs")
    # lacrosse: Uint8 = Field(alias="Category Name_Lacrosse")
    # mens_clothing: Uint8 = Field(alias="Category Name_Men's Clothing")
    # mens_footwear: Uint8 = Field(alias="Category Name_Men's Footwear")
    # mens_golf_clubs: Uint8 = Field(alias="Category Name_Men's Golf Clubs")
    # music: Uint8 = Field(alias="Category Name_Music")
    # pet_supplies: Uint8 = Field(alias="Category Name_Pet Supplies")
    # shop_by_sport: Uint8 = Field(alias="Category Name_Shop By Sport")
    # soccer: Uint8 = Field(alias="Category Name_Soccer")
    # sporting_goods: Uint8 = Field(alias="Category Name_Sporting Goods")
    # strength_training: Uint8 = Field(alias="Category Name_Strength Training")
    # tennis_racquet: Uint8 = Field(alias="Category Name_Tennis & Racquet")
    # toys: Uint8 = Field(alias="Category Name_Toys")
    # trade_in: Uint8 = Field(alias="Category Name_Trade-In")
    # video_games: Uint8 = Field(alias="Category Name_Video Games")
    # water_sports: Uint8 = Field(alias="Category Name_Water Sports")
    # womens_apparel: Uint8 = Field(alias="Category Name_Women's Apparel")
    # womens_clothing: Uint8 = Field(alias="Category Name_Women's Clothing")
    # womens_golf_clubs: Uint8 = Field(alias="Category Name_Women's Golf Clubs")

# class DelayInput(BaseModel):
    # model_config = ConfigDict(populate_by_name=True)
    # cust_zip: zipcode_type = Field(alias="Customer Zipcode")
    # order_year:     int
    # order_month:    int
    # order_week_day: int
    # order_hour:     int


    # # --- Shipping Mode ---
    # ship_mode_same_day: Uint8 = Field(alias="Shipping Mode_Same Day")
    # ship_mode_second_class: Uint8 = Field(alias="Shipping Mode_Second Class")
    # ship_mode_standard_class: Uint8 = Field(alias="Shipping Mode_Standard Class")

    # # --- Market ---
    # market_europe: Uint8 = Field(alias="Market_Europe")
    # market_latam: Uint8 = Field(alias="Market_LATAM")
    # market_pacific_asia: Uint8 = Field(alias="Market_Pacific Asia")
    # market_usca: Uint8 = Field(alias="Market_USCA")

    # # --- Order Region ---
    # region_caribbean: Uint8 = Field(alias="Order Region_Caribbean")
    # region_central_africa: Uint8 = Field(alias="Order Region_Central Africa")
    # region_central_america: Uint8 = Field(alias="Order Region_Central America")
    # region_central_asia: Uint8 = Field(alias="Order Region_Central Asia")
    # region_east_africa: Uint8 = Field(alias="Order Region_East Africa")
    # region_east_usa: Uint8 = Field(alias="Order Region_East of USA")
    # region_eastern_asia: Uint8 = Field(alias="Order Region_Eastern Asia")
    # region_eastern_europe: Uint8 = Field(alias="Order Region_Eastern Europe")
    # region_north_africa: Uint8 = Field(alias="Order Region_North Africa")
    # region_northern_europe: Uint8 = Field(alias="Order Region_Northern Europe")
    # region_oceania: Uint8 = Field(alias="Order Region_Oceania")
    # region_south_america: Uint8 = Field(alias="Order Region_South America")
    # region_south_asia: Uint8 = Field(alias="Order Region_South Asia")
    # region_south_usa: Uint8 = Field(alias="Order Region_South of  USA ")
    # region_southeast_asia: Uint8 = Field(alias="Order Region_Southeast Asia")
    # region_southern_africa: Uint8 = Field(alias="Order Region_Southern Africa")
    # region_southern_europe: Uint8 = Field(alias="Order Region_Southern Europe")
    # region_us_center: Uint8 = Field(alias="Order Region_US Center ")
    # region_west_africa: Uint8 = Field(alias="Order Region_West Africa")
    # region_west_asia: Uint8 = Field(alias="Order Region_West Asia")
    # region_west_usa: Uint8 = Field(alias="Order Region_West of USA ")
    # region_western_europe: Uint8 = Field(alias="Order Region_Western Europe")

    # # --- Category Name ---
    # cat_as_seen_on_tv: Uint8 = Field(alias="Category Name_As Seen on  TV!")
    # cat_baby: Uint8 = Field(alias="Category Name_Baby ")
    # cat_baseball_softball: Uint8 = Field(alias="Category Name_Baseball & Softball")
    # cat_basketball: Uint8 = Field(alias="Category Name_Basketball")
    # cat_books: Uint8 = Field(alias="Category Name_Books ")
    # cat_boxing_mma: Uint8 = Field(alias="Category Name_Boxing & MMA")
    # cat_cds: Uint8 = Field(alias="Category Name_CDs ")
    # cat_cameras: Uint8 = Field(alias="Category Name_Cameras ")
    # cat_camping_hiking: Uint8 = Field(alias="Category Name_Camping & Hiking")
    # cat_cardio_equipment: Uint8 = Field(alias="Category Name_Cardio Equipment")
    # cat_childrens_clothing: Uint8 = Field(alias="Category Name_Children's Clothing")
    # cat_cleats: Uint8 = Field(alias="Category Name_Cleats")
    # cat_computers: Uint8 = Field(alias="Category Name_Computers")
    # cat_consumer_electronics: Uint8 = Field(alias="Category Name_Consumer Electronics")
    # cat_crafts: Uint8 = Field(alias="Category Name_Crafts")
    # cat_dvds: Uint8 = Field(alias="Category Name_DVDs")
    # cat_electronics: Uint8 = Field(alias="Category Name_Electronics")
    # cat_fishing: Uint8 = Field(alias="Category Name_Fishing")
    # cat_fitness_accessories: Uint8 = Field(alias="Category Name_Fitness Accessories")
    # cat_garden: Uint8 = Field(alias="Category Name_Garden")
    # cat_girls_apparel: Uint8 = Field(alias="Category Name_Girls' Apparel")
    # cat_golf_apparel: Uint8 = Field(alias="Category Name_Golf Apparel")
    # cat_golf_bags_carts: Uint8 = Field(alias="Category Name_Golf Bags & Carts")
    # cat_golf_balls: Uint8 = Field(alias="Category Name_Golf Balls")
    # cat_golf_gloves: Uint8 = Field(alias="Category Name_Golf Gloves")
    # cat_golf_shoes: Uint8 = Field(alias="Category Name_Golf Shoes")
    # cat_health_beauty: Uint8 = Field(alias="Category Name_Health and Beauty")
    # cat_hockey: Uint8 = Field(alias="Category Name_Hockey")
    # cat_hunting_shooting: Uint8 = Field(alias="Category Name_Hunting & Shooting")
    # cat_indoor_outdoor_games: Uint8 = Field(alias="Category Name_Indoor/Outdoor Games")
    # cat_kids_golf_clubs: Uint8 = Field(alias="Category Name_Kids' Golf Clubs")
    # cat_lacrosse: Uint8 = Field(alias="Category Name_Lacrosse")
    # cat_mens_clothing: Uint8 = Field(alias="Category Name_Men's Clothing")
    # cat_mens_footwear: Uint8 = Field(alias="Category Name_Men's Footwear")
    # cat_mens_golf_clubs: Uint8 = Field(alias="Category Name_Men's Golf Clubs")
    # cat_music: Uint8 = Field(alias="Category Name_Music")
    # cat_pet_supplies: Uint8 = Field(alias="Category Name_Pet Supplies")
    # cat_shop_by_sport: Uint8 = Field(alias="Category Name_Shop By Sport")
    # cat_soccer: Uint8 = Field(alias="Category Name_Soccer")
    # cat_sporting_goods: Uint8 = Field(alias="Category Name_Sporting Goods")
    # cat_strength_training: Uint8 = Field(alias="Category Name_Strength Training")
    # cat_tennis_racquet: Uint8 = Field(alias="Category Name_Tennis & Racquet")
    # cat_toys: Uint8 = Field(alias="Category Name_Toys")
    # cat_trade_in: Uint8 = Field(alias="Category Name_Trade-In")
    # cat_video_games: Uint8 = Field(alias="Category Name_Video Games")
    # cat_water_sports: Uint8 = Field(alias="Category Name_Water Sports")
    # cat_womens_apparel: Uint8 = Field(alias="Category Name_Women's Apparel")
    # cat_womens_clothing: Uint8 = Field(alias="Category Name_Women's Clothing")
    # cat_womens_golf_clubs: Uint8 = Field(alias="Category Name_Women's Golf Clubs")

    # # --- Department Name ---
    # dept_book_shop: Uint8 = Field(alias="Department Name_Book Shop")
    # dept_discs_shop: Uint8 = Field(alias="Department Name_Discs Shop")
    # dept_fan_shop: Uint8 = Field(alias="Department Name_Fan Shop")
    # dept_fitness: Uint8 = Field(alias="Department Name_Fitness")
    # dept_footwear: Uint8 = Field(alias="Department Name_Footwear")
    # dept_golf: Uint8 = Field(alias="Department Name_Golf")
    # dept_health_beauty: Uint8 = Field(alias="Department Name_Health and Beauty ")
    # dept_outdoors: Uint8 = Field(alias="Department Name_Outdoors")
    # dept_pet_shop: Uint8 = Field(alias="Department Name_Pet Shop")
    # dept_technology: Uint8 = Field(alias="Department Name_Technology")

    # # --- Customer Segment ---
    # segment_corporate: Uint8 = Field(alias="Customer Segment_Corporate")
    # segment_home_office: Uint8 = Field(alias="Customer Segment_Home Office")

    # # --- Type ---
    # type_debit: Uint8 = Field(alias="Type_DEBIT")
    # type_payment: Uint8 = Field(alias="Type_PAYMENT")
    # type_transfer: Uint8 = Field(alias="Type_TRANSFER")

# # ============================================================
# # ENDPOINTS
# # ============================================================

# @app.get("/")
# def home():
    # return {
        # "message": "Supply Chain Decision Support API",
        # "endpoints": ["/predict/demand", "/predict/delay", "/health"]
    # }

# @app.get("/health")
# def health():
    # return {"status": "running"}

# @app.post("/predict/demand")
# def predict_demand(input: DemandInput):
    # # Build feature dataframe
    # features = pd.DataFrame([input.model_dump(by_alias=True)])
    
    # # Predict
    # predicted_demand = best_demand_model.predict(features)[0]
    # category_avg     = input.avg_price
    
    # # Decision logic
    # if predicted_demand > category_avg * 1.2:
        # decision = "Increase Stock"
    # elif predicted_demand < category_avg * 0.8:
        # decision = "Reduce Stock"
    # else:
        # decision = "Maintain"
    
    # return {
        # "predicted_demand":    round(float(predicted_demand), 2),
        # "category_avg":        round(category_avg, 2),
        # "inventory_decision":  decision
    # }

# @app.post("/predict/delay")
# def predict_delay(input: DelayInput):
    # # Build feature dataframe
    # features = pd.DataFrame([input.model_dump(by_alias=True)])
    
    # # Predict probability
    # delay_risk = best_delay_model.predict_proba(features)[0][1]
    
    # # Decision logic
    # if delay_risk > 0.7:
        # decision = "Expedite Shipment"
    # elif delay_risk > 0.4:
        # decision = "Monitor Closely"
    # else:
        # decision = "No Action"
    
    # return {
        # "delay_risk":      round(float(delay_risk), 3),
        # "delay_decision":  decision
    # }

# @app.post("/")
# def read_root(item: DataInput):
    # return {"Hello": item.name, "Status": "Live on BTP"}
   
# if __name__ == "__main__":
    # # BTP provides the port via an environment variable
    # port = int(os.environ.get("PORT", 8080))
    # uvicorn.run(app, host="0.0.0.0", port=port)
    
    
# import os
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# from pydantic import BaseModel, Field, ConfigDict
# import pandas as pd
# from typing import Annotated, List
# import joblib

# app = FastAPI(title="Supply Chain Decision Support API")

# ============================================================
# CORS — required for SAC JavaScript to call this API
# ============================================================
# app.add_middleware(
    # CORSMiddleware,
    # allow_origins=["*"],
    # allow_credentials=True,
    # allow_methods=["*"],
    # allow_headers=["*"],
# )

# ============================================================
# LOAD MODELS
# ============================================================
# best_demand_model = joblib.load("demand_model.pkl")
# best_delay_model  = joblib.load("delay_model.pkl")

# ============================================================
# TYPE DEFINITIONS
# ============================================================
# Uint8        = Annotated[int, Field(0, ge=0, le=255)]
# zipcode_type = Annotated[int, Field(0, ge=0, le=100000)]

# ============================================================
# INPUT SCHEMAS (unchanged from your original)
# ============================================================

# class DataInput(BaseModel):
    # name: str

# class DemandInput(BaseModel):
    # model_config = ConfigDict(populate_by_name=True)
    # order_year:   int
    # order_month:  int
    # avg_price:    float
    # order_count:  int
    # as_seen_on_tv:         Uint8 = Field(0, alias="Category Name_As Seen on  TV!")
    # baby:                  Uint8 = Field(0, alias="Category Name_Baby ")
    # baseball_softball:     Uint8 = Field(0, alias="Category Name_Baseball & Softball")
    # basketball:            Uint8 = Field(0, alias="Category Name_Basketball")
    # books:                 Uint8 = Field(0, alias="Category Name_Books ")
    # boxing_mma:            Uint8 = Field(0, alias="Category Name_Boxing & MMA")
    # cds:                   Uint8 = Field(0, alias="Category Name_CDs ")
    # cameras:               Uint8 = Field(0, alias="Category Name_Cameras ")
    # camping_hiking:        Uint8 = Field(0, alias="Category Name_Camping & Hiking")
    # cardio_equipment:      Uint8 = Field(0, alias="Category Name_Cardio Equipment")
    # childrens_clothing:    Uint8 = Field(0, alias="Category Name_Children's Clothing")
    # cleats:                Uint8 = Field(0, alias="Category Name_Cleats")
    # computers:             Uint8 = Field(0, alias="Category Name_Computers")
    # consumer_electronics:  Uint8 = Field(0, alias="Category Name_Consumer Electronics")
    # crafts:                Uint8 = Field(0, alias="Category Name_Crafts")
    # dvds:                  Uint8 = Field(0, alias="Category Name_DVDs")
    # electronics:           Uint8 = Field(0, alias="Category Name_Electronics")
    # fishing:               Uint8 = Field(0, alias="Category Name_Fishing")
    # fitness_accessories:   Uint8 = Field(0, alias="Category Name_Fitness Accessories")
    # garden:                Uint8 = Field(0, alias="Category Name_Garden")
    # girls_apparel:         Uint8 = Field(0, alias="Category Name_Girls' Apparel")
    # golf_apparel:          Uint8 = Field(0, alias="Category Name_Golf Apparel")
    # golf_bags_carts:       Uint8 = Field(0, alias="Category Name_Golf Bags & Carts")
    # golf_balls:            Uint8 = Field(0, alias="Category Name_Golf Balls")
    # golf_gloves:           Uint8 = Field(0, alias="Category Name_Golf Gloves")
    # golf_shoes:            Uint8 = Field(0, alias="Category Name_Golf Shoes")
    # health_beauty:         Uint8 = Field(0, alias="Category Name_Health and Beauty")
    # hockey:                Uint8 = Field(0, alias="Category Name_Hockey")
    # hunting_shooting:      Uint8 = Field(0, alias="Category Name_Hunting & Shooting")
    # indoor_outdoor_games:  Uint8 = Field(0, alias="Category Name_Indoor/Outdoor Games")
    # kids_golf_clubs:       Uint8 = Field(0, alias="Category Name_Kids' Golf Clubs")
    # lacrosse:              Uint8 = Field(0, alias="Category Name_Lacrosse")
    # mens_clothing:         Uint8 = Field(0, alias="Category Name_Men's Clothing")
    # mens_footwear:         Uint8 = Field(0, alias="Category Name_Men's Footwear")
    # mens_golf_clubs:       Uint8 = Field(0, alias="Category Name_Men's Golf Clubs")
    # music:                 Uint8 = Field(0, alias="Category Name_Music")
    # pet_supplies:          Uint8 = Field(0, alias="Category Name_Pet Supplies")
    # shop_by_sport:         Uint8 = Field(0, alias="Category Name_Shop By Sport")
    # soccer:                Uint8 = Field(0, alias="Category Name_Soccer")
    # sporting_goods:        Uint8 = Field(0, alias="Category Name_Sporting Goods")
    # strength_training:     Uint8 = Field(0, alias="Category Name_Strength Training")
    # tennis_racquet:        Uint8 = Field(0, alias="Category Name_Tennis & Racquet")
    # toys:                  Uint8 = Field(0, alias="Category Name_Toys")
    # trade_in:              Uint8 = Field(0, alias="Category Name_Trade-In")
    # video_games:           Uint8 = Field(0, alias="Category Name_Video Games")
    # water_sports:          Uint8 = Field(0, alias="Category Name_Water Sports")
    # womens_apparel:        Uint8 = Field(0, alias="Category Name_Women's Apparel")
    # womens_clothing:       Uint8 = Field(0, alias="Category Name_Women's Clothing")
    # womens_golf_clubs:     Uint8 = Field(0, alias="Category Name_Women's Golf Clubs")

# class DelayInput(BaseModel):
    # model_config = ConfigDict(populate_by_name=True)
    # cust_zip:       zipcode_type = Field(0, alias="Customer Zipcode")
    # order_year:     int
    # order_month:    int
    # order_week_day: int
    # order_hour:     int
    # ship_mode_same_day:       Uint8 = Field(0, alias="Shipping Mode_Same Day")
    # ship_mode_second_class:   Uint8 = Field(0, alias="Shipping Mode_Second Class")
    # ship_mode_standard_class: Uint8 = Field(0, alias="Shipping Mode_Standard Class")
    # market_europe:       Uint8 = Field(0, alias="Market_Europe")
    # market_latam:        Uint8 = Field(0, alias="Market_LATAM")
    # market_pacific_asia: Uint8 = Field(0, alias="Market_Pacific Asia")
    # market_usca:         Uint8 = Field(0, alias="Market_USCA")
    # region_caribbean:       Uint8 = Field(0, alias="Order Region_Caribbean")
    # region_central_africa:  Uint8 = Field(0, alias="Order Region_Central Africa")
    # region_central_america: Uint8 = Field(0, alias="Order Region_Central America")
    # region_central_asia:    Uint8 = Field(0, alias="Order Region_Central Asia")
    # region_east_africa:     Uint8 = Field(0, alias="Order Region_East Africa")
    # region_east_usa:        Uint8 = Field(0, alias="Order Region_East of USA")
    # region_eastern_asia:    Uint8 = Field(0, alias="Order Region_Eastern Asia")
    # region_eastern_europe:  Uint8 = Field(0, alias="Order Region_Eastern Europe")
    # region_north_africa:    Uint8 = Field(0, alias="Order Region_North Africa")
    # region_northern_europe: Uint8 = Field(0, alias="Order Region_Northern Europe")
    # region_oceania:         Uint8 = Field(0, alias="Order Region_Oceania")
    # region_south_america:   Uint8 = Field(0, alias="Order Region_South America")
    # region_south_asia:      Uint8 = Field(0, alias="Order Region_South Asia")
    # region_south_usa:       Uint8 = Field(0, alias="Order Region_South of  USA ")
    # region_southeast_asia:  Uint8 = Field(0, alias="Order Region_Southeast Asia")
    # region_southern_africa: Uint8 = Field(0, alias="Order Region_Southern Africa")
    # region_southern_europe: Uint8 = Field(0, alias="Order Region_Southern Europe")
    # region_us_center:       Uint8 = Field(0, alias="Order Region_US Center ")
    # region_west_africa:     Uint8 = Field(0, alias="Order Region_West Africa")
    # region_west_asia:       Uint8 = Field(0, alias="Order Region_West Asia")
    # region_west_usa:        Uint8 = Field(0, alias="Order Region_West of USA ")
    # region_western_europe:  Uint8 = Field(0, alias="Order Region_Western Europe")
    # cat_as_seen_on_tv:        Uint8 = Field(0, alias="Category Name_As Seen on  TV!")
    # cat_baby:                 Uint8 = Field(0, alias="Category Name_Baby ")
    # cat_baseball_softball:    Uint8 = Field(0, alias="Category Name_Baseball & Softball")
    # cat_basketball:           Uint8 = Field(0, alias="Category Name_Basketball")
    # cat_books:                Uint8 = Field(0, alias="Category Name_Books ")
    # cat_boxing_mma:           Uint8 = Field(0, alias="Category Name_Boxing & MMA")
    # cat_cds:                  Uint8 = Field(0, alias="Category Name_CDs ")
    # cat_cameras:              Uint8 = Field(0, alias="Category Name_Cameras ")
    # cat_camping_hiking:       Uint8 = Field(0, alias="Category Name_Camping & Hiking")
    # cat_cardio_equipment:     Uint8 = Field(0, alias="Category Name_Cardio Equipment")
    # cat_childrens_clothing:   Uint8 = Field(0, alias="Category Name_Children's Clothing")
    # cat_cleats:               Uint8 = Field(0, alias="Category Name_Cleats")
    # cat_computers:            Uint8 = Field(0, alias="Category Name_Computers")
    # cat_consumer_electronics: Uint8 = Field(0, alias="Category Name_Consumer Electronics")
    # cat_crafts:               Uint8 = Field(0, alias="Category Name_Crafts")
    # cat_dvds:                 Uint8 = Field(0, alias="Category Name_DVDs")
    # cat_electronics:          Uint8 = Field(0, alias="Category Name_Electronics")
    # cat_fishing:              Uint8 = Field(0, alias="Category Name_Fishing")
    # cat_fitness_accessories:  Uint8 = Field(0, alias="Category Name_Fitness Accessories")
    # cat_garden:               Uint8 = Field(0, alias="Category Name_Garden")
    # cat_girls_apparel:        Uint8 = Field(0, alias="Category Name_Girls' Apparel")
    # cat_golf_apparel:         Uint8 = Field(0, alias="Category Name_Golf Apparel")
    # cat_golf_bags_carts:      Uint8 = Field(0, alias="Category Name_Golf Bags & Carts")
    # cat_golf_balls:           Uint8 = Field(0, alias="Category Name_Golf Balls")
    # cat_golf_gloves:          Uint8 = Field(0, alias="Category Name_Golf Gloves")
    # cat_golf_shoes:           Uint8 = Field(0, alias="Category Name_Golf Shoes")
    # cat_health_beauty:        Uint8 = Field(0, alias="Category Name_Health and Beauty")
    # cat_hockey:               Uint8 = Field(0, alias="Category Name_Hockey")
    # cat_hunting_shooting:     Uint8 = Field(0, alias="Category Name_Hunting & Shooting")
    # cat_indoor_outdoor_games: Uint8 = Field(0, alias="Category Name_Indoor/Outdoor Games")
    # cat_kids_golf_clubs:      Uint8 = Field(0, alias="Category Name_Kids' Golf Clubs")
    # cat_lacrosse:             Uint8 = Field(0, alias="Category Name_Lacrosse")
    # cat_mens_clothing:        Uint8 = Field(0, alias="Category Name_Men's Clothing")
    # cat_mens_footwear:        Uint8 = Field(0, alias="Category Name_Men's Footwear")
    # cat_mens_golf_clubs:      Uint8 = Field(0, alias="Category Name_Men's Golf Clubs")
    # cat_music:                Uint8 = Field(0, alias="Category Name_Music")
    # cat_pet_supplies:         Uint8 = Field(0, alias="Category Name_Pet Supplies")
    # cat_shop_by_sport:        Uint8 = Field(0, alias="Category Name_Shop By Sport")
    # cat_soccer:               Uint8 = Field(0, alias="Category Name_Soccer")
    # cat_sporting_goods:       Uint8 = Field(0, alias="Category Name_Sporting Goods")
    # cat_strength_training:    Uint8 = Field(0, alias="Category Name_Strength Training")
    # cat_tennis_racquet:       Uint8 = Field(0, alias="Category Name_Tennis & Racquet")
    # cat_toys:                 Uint8 = Field(0, alias="Category Name_Toys")
    # cat_trade_in:             Uint8 = Field(0, alias="Category Name_Trade-In")
    # cat_video_games:          Uint8 = Field(0, alias="Category Name_Video Games")
    # cat_water_sports:         Uint8 = Field(0, alias="Category Name_Water Sports")
    # cat_womens_apparel:       Uint8 = Field(0, alias="Category Name_Women's Apparel")
    # cat_womens_clothing:      Uint8 = Field(0, alias="Category Name_Women's Clothing")
    # cat_womens_golf_clubs:    Uint8 = Field(0, alias="Category Name_Women's Golf Clubs")
    # dept_book_shop:     Uint8 = Field(0, alias="Department Name_Book Shop")
    # dept_discs_shop:    Uint8 = Field(0, alias="Department Name_Discs Shop")
    # dept_fan_shop:      Uint8 = Field(0, alias="Department Name_Fan Shop")
    # dept_fitness:       Uint8 = Field(0, alias="Department Name_Fitness")
    # dept_footwear:      Uint8 = Field(0, alias="Department Name_Footwear")
    # dept_golf:          Uint8 = Field(0, alias="Department Name_Golf")
    # dept_health_beauty: Uint8 = Field(0, alias="Department Name_Health and Beauty ")
    # dept_outdoors:      Uint8 = Field(0, alias="Department Name_Outdoors")
    # dept_pet_shop:      Uint8 = Field(0, alias="Department Name_Pet Shop")
    # dept_technology:    Uint8 = Field(0, alias="Department Name_Technology")
    # segment_corporate:   Uint8 = Field(0, alias="Customer Segment_Corporate")
    # segment_home_office: Uint8 = Field(0, alias="Customer Segment_Home Office")
    # type_debit:    Uint8 = Field(0, alias="Type_DEBIT")
    # type_payment:  Uint8 = Field(0, alias="Type_PAYMENT")
    # type_transfer: Uint8 = Field(0, alias="Type_TRANSFER")

# ============================================================
# SIMPLIFIED INPUT SCHEMAS FOR SAC
# (managers enter human-readable values, API converts)
# ============================================================

# class DemandSimpleInput(BaseModel):
    # """
    # Simplified demand input for SAC dashboard.
    # Manager selects category name and enters basic fields.
    # API handles one-hot encoding internally.
    # """
    # order_year:    int
    # order_month:   int
    # avg_price:     float
    # order_count:   int
    # category_name: str  # e.g. "Basketball", "Electronics", "Fishing"

# class DelaySimpleInput(BaseModel):
    # """
    # Simplified delay input for SAC dashboard.
    # Manager selects shipping mode, market, region.
    # API handles one-hot encoding internally.
    # """
    # order_year:    int
    # order_month:   int
    # order_week_day: int
    # order_hour:    int
    # customer_zipcode: int = 0
    # shipping_mode: str  # "Standard Class", "First Class", "Second Class", "Same Day"
    # market:        str  # "Europe", "LATAM", "Pacific Asia", "USCA", "Africa"
    # order_region:  str  # "Western Europe", "Central America", etc.
    # category_name: str  # "Basketball", "Electronics", etc.
    # department:    str  # "Fan Shop", "Technology", etc.
    # customer_segment: str = "Consumer"  # "Consumer", "Corporate", "Home Office"
    # payment_type:  str = "TRANSFER"     # "TRANSFER", "DEBIT", "PAYMENT", "CASH"

# ============================================================
# ENDPOINTS
# ============================================================

# @app.get("/")
# def home():
    # return {
        # "message": "Supply Chain Decision Support API",
        # "endpoints": ["/predict/demand", "/predict/delay",
                      # "/predict/demand/simple", "/predict/delay/simple",
                      # "/health"]
    # }

# @app.get("/health")
# def health():
    # return {"status": "running"}

# @app.post("/predict/demand")
# def predict_demand(input: DemandInput):
    # features         = pd.DataFrame([input.model_dump(by_alias=True)])
    # predicted_demand = best_demand_model.predict(features)[0]
    # category_avg     = input.avg_price

    # if predicted_demand > category_avg * 1.2:
        # decision = "Increase Stock"
    # elif predicted_demand < category_avg * 0.8:
        # decision = "Reduce Stock"
    # else:
        # decision = "Maintain"

    # return {
        # "predicted_demand":   round(float(predicted_demand), 2),
        # "category_avg":       round(category_avg, 2),
        # "inventory_decision": decision
    # }

# @app.post("/predict/delay")
# def predict_delay(input: DelayInput):
    # features   = pd.DataFrame([input.model_dump(by_alias=True)])
    # delay_risk = best_delay_model.predict_proba(features)[0][1]

    # if delay_risk > 0.7:
        # decision = "Expedite Shipment"
    # elif delay_risk > 0.4:
        # decision = "Monitor Closely"
    # else:
        # decision = "No Action"

    # return {
        # "delay_risk":     round(float(delay_risk), 3),
        # "delay_decision": decision
    # }

# ============================================================
# SIMPLIFIED ENDPOINTS — designed for SAC dashboard
# Manager enters human-readable values
# API handles encoding internally
# ============================================================

# @app.post("/predict/demand/simple")
# def predict_demand_simple(input: DemandSimpleInput):
    # """
    # SAC-friendly endpoint.
    # Manager selects category from dropdown,
    # enters year/month/price/count.
    # No need to know about one-hot encoding.
    # """
    # Start with all category columns as 0
    # base = {
        # "order_year":   input.order_year,
        # "order_month":  input.order_month,
        # "avg_price":    input.avg_price,
        # "order_count":  input.order_count,
        # "Category Name_As Seen on  TV!":        0,
        # "Category Name_Baby ":                  0,
        # "Category Name_Baseball & Softball":    0,
        # "Category Name_Basketball":             0,
        # "Category Name_Books ":                 0,
        # "Category Name_Boxing & MMA":           0,
        # "Category Name_CDs ":                   0,
        # "Category Name_Cameras ":               0,
        # "Category Name_Camping & Hiking":       0,
        # "Category Name_Cardio Equipment":       0,
        # "Category Name_Children's Clothing":    0,
        # "Category Name_Cleats":                 0,
        # "Category Name_Computers":              0,
        # "Category Name_Consumer Electronics":   0,
        # "Category Name_Crafts":                 0,
        # "Category Name_DVDs":                   0,
        # "Category Name_Electronics":            0,
        # "Category Name_Fishing":                0,
        # "Category Name_Fitness Accessories":    0,
        # "Category Name_Garden":                 0,
        # "Category Name_Girls' Apparel":         0,
        # "Category Name_Golf Apparel":           0,
        # "Category Name_Golf Bags & Carts":      0,
        # "Category Name_Golf Balls":             0,
        # "Category Name_Golf Gloves":            0,
        # "Category Name_Golf Shoes":             0,
        # "Category Name_Health and Beauty":      0,
        # "Category Name_Hockey":                 0,
        # "Category Name_Hunting & Shooting":     0,
        # "Category Name_Indoor/Outdoor Games":   0,
        # "Category Name_Kids' Golf Clubs":       0,
        # "Category Name_Lacrosse":               0,
        # "Category Name_Men's Clothing":         0,
        # "Category Name_Men's Footwear":         0,
        # "Category Name_Men's Golf Clubs":       0,
        # "Category Name_Music":                  0,
        # "Category Name_Pet Supplies":           0,
        # "Category Name_Shop By Sport":          0,
        # "Category Name_Soccer":                 0,
        # "Category Name_Sporting Goods":         0,
        # "Category Name_Strength Training":      0,
        # "Category Name_Tennis & Racquet":       0,
        # "Category Name_Toys":                   0,
        # "Category Name_Trade-In":               0,
        # "Category Name_Video Games":            0,
        # "Category Name_Water Sports":           0,
        # "Category Name_Women's Apparel":        0,
        # "Category Name_Women's Clothing":       0,
        # "Category Name_Women's Golf Clubs":     0,
    # }

    # Set the selected category to 1
    # cat_key = "Category Name_" + input.category_name
    # if cat_key in base:
        # base[cat_key] = 1

    # features         = pd.DataFrame([base])
    # predicted_demand = best_demand_model.predict(features)[0]
    # category_avg     = input.avg_price

    # if predicted_demand > category_avg * 1.2:
        # decision = "Increase Stock"
    # elif predicted_demand < category_avg * 0.8:
        # decision = "Reduce Stock"
    # else:
        # decision = "Maintain"

    # return {
        # "predicted_demand":   round(float(predicted_demand), 2),
        # "category_avg":       round(category_avg, 2),
        # "inventory_decision": decision,
        # "category_name":      input.category_name,
        # "order_year":         input.order_year,
        # "order_month":        input.order_month
    # }

# @app.post("/predict/delay/simple")
# def predict_delay_simple(input: DelaySimpleInput):
    # """
    # SAC-friendly endpoint.
    # Manager selects shipping mode, market, region, category.
    # API handles all one-hot encoding internally.
    # """
    # base = {
        # "Customer Zipcode":              input.customer_zipcode,
        # "order_year":                    input.order_year,
        # "order_month":                   input.order_month,
        # "order_week_day":                input.order_week_day,
        # "order_hour":                    input.order_hour,
        # "Shipping Mode_Same Day":        1 if input.shipping_mode == "Same Day"       else 0,
        # "Shipping Mode_Second Class":    1 if input.shipping_mode == "Second Class"   else 0,
        # "Shipping Mode_Standard Class":  1 if input.shipping_mode == "Standard Class" else 0,
        # "Market_Europe":                 1 if input.market == "Europe"       else 0,
        # "Market_LATAM":                  1 if input.market == "LATAM"        else 0,
        # "Market_Pacific Asia":           1 if input.market == "Pacific Asia" else 0,
        # "Market_USCA":                   1 if input.market == "USCA"         else 0,
        # "Order Region_Caribbean":        1 if input.order_region == "Caribbean"       else 0,
        # "Order Region_Central Africa":   1 if input.order_region == "Central Africa"  else 0,
        # "Order Region_Central America":  1 if input.order_region == "Central America" else 0,
        # "Order Region_Central Asia":     1 if input.order_region == "Central Asia"    else 0,
        # "Order Region_East Africa":      1 if input.order_region == "East Africa"     else 0,
        # "Order Region_East of USA":      1 if input.order_region == "East of USA"     else 0,
        # "Order Region_Eastern Asia":     1 if input.order_region == "Eastern Asia"    else 0,
        # "Order Region_Eastern Europe":   1 if input.order_region == "Eastern Europe"  else 0,
        # "Order Region_North Africa":     1 if input.order_region == "North Africa"    else 0,
        # "Order Region_Northern Europe":  1 if input.order_region == "Northern Europe" else 0,
        # "Order Region_Oceania":          1 if input.order_region == "Oceania"         else 0,
        # "Order Region_South America":    1 if input.order_region == "South America"   else 0,
        # "Order Region_South Asia":       1 if input.order_region == "South Asia"      else 0,
        # "Order Region_South of  USA ":   1 if input.order_region == "South of USA"    else 0,
        # "Order Region_Southeast Asia":   1 if input.order_region == "Southeast Asia"  else 0,
        # "Order Region_Southern Africa":  1 if input.order_region == "Southern Africa" else 0,
        # "Order Region_Southern Europe":  1 if input.order_region == "Southern Europe" else 0,
        # "Order Region_US Center ":       1 if input.order_region == "US Center"       else 0,
        # "Order Region_West Africa":      1 if input.order_region == "West Africa"     else 0,
        # "Order Region_West Asia":        1 if input.order_region == "West Asia"       else 0,
        # "Order Region_West of USA ":     1 if input.order_region == "West of USA"     else 0,
        # "Order Region_Western Europe":   1 if input.order_region == "Western Europe"  else 0,
        # "Category Name_As Seen on  TV!":      0,
        # "Category Name_Baby ":                0,
        # "Category Name_Baseball & Softball":  0,
        # "Category Name_Basketball":           0,
        # "Category Name_Books ":               0,
        # "Category Name_Boxing & MMA":         0,
        # "Category Name_CDs ":                 0,
        # "Category Name_Cameras ":             0,
        # "Category Name_Camping & Hiking":     0,
        # "Category Name_Cardio Equipment":     0,
        # "Category Name_Children's Clothing":  0,
        # "Category Name_Cleats":               0,
        # "Category Name_Computers":            0,
        # "Category Name_Consumer Electronics": 0,
        # "Category Name_Crafts":               0,
        # "Category Name_DVDs":                 0,
        # "Category Name_Electronics":          0,
        # "Category Name_Fishing":              0,
        # "Category Name_Fitness Accessories":  0,
        # "Category Name_Garden":               0,
        # "Category Name_Girls' Apparel":       0,
        # "Category Name_Golf Apparel":         0,
        # "Category Name_Golf Bags & Carts":    0,
        # "Category Name_Golf Balls":           0,
        # "Category Name_Golf Gloves":          0,
        # "Category Name_Golf Shoes":           0,
        # "Category Name_Health and Beauty":    0,
        # "Category Name_Hockey":               0,
        # "Category Name_Hunting & Shooting":   0,
        # "Category Name_Indoor/Outdoor Games": 0,
        # "Category Name_Kids' Golf Clubs":     0,
        # "Category Name_Lacrosse":             0,
        # "Category Name_Men's Clothing":       0,
        # "Category Name_Men's Footwear":       0,
        # "Category Name_Men's Golf Clubs":     0,
        # "Category Name_Music":                0,
        # "Category Name_Pet Supplies":         0,
        # "Category Name_Shop By Sport":        0,
        # "Category Name_Soccer":               0,
        # "Category Name_Sporting Goods":       0,
        # "Category Name_Strength Training":    0,
        # "Category Name_Tennis & Racquet":     0,
        # "Category Name_Toys":                 0,
        # "Category Name_Trade-In":             0,
        # "Category Name_Video Games":          0,
        # "Category Name_Water Sports":         0,
        # "Category Name_Women's Apparel":      0,
        # "Category Name_Women's Clothing":     0,
        # "Category Name_Women's Golf Clubs":   0,
        # "Department Name_Book Shop":          1 if input.department == "Book Shop"         else 0,
        # "Department Name_Discs Shop":         1 if input.department == "Discs Shop"        else 0,
        # "Department Name_Fan Shop":           1 if input.department == "Fan Shop"          else 0,
        # "Department Name_Fitness":            1 if input.department == "Fitness"           else 0,
        # "Department Name_Footwear":           1 if input.department == "Footwear"          else 0,
        # "Department Name_Golf":               1 if input.department == "Golf"              else 0,
        # "Department Name_Health and Beauty ": 1 if input.department == "Health and Beauty" else 0,
        # "Department Name_Outdoors":           1 if input.department == "Outdoors"          else 0,
        # "Department Name_Pet Shop":           1 if input.department == "Pet Shop"          else 0,
        # "Department Name_Technology":         1 if input.department == "Technology"        else 0,
        # "Customer Segment_Corporate":         1 if input.customer_segment == "Corporate"   else 0,
        # "Customer Segment_Home Office":       1 if input.customer_segment == "Home Office" else 0,
        # "Type_DEBIT":                         1 if input.payment_type == "DEBIT"    else 0,
        # "Type_PAYMENT":                       1 if input.payment_type == "PAYMENT"  else 0,
        # "Type_TRANSFER":                      1 if input.payment_type == "TRANSFER" else 0,
    # }

    # Set selected category
    # cat_key = "Category Name_" + input.category_name
    # if cat_key in base:
        # base[cat_key] = 1

    # features   = pd.DataFrame([base])
    # delay_risk = best_delay_model.predict_proba(features)[0][1]

    # if delay_risk > 0.7:
        # decision = "Expedite Shipment"
    # elif delay_risk > 0.4:
        # decision = "Monitor Closely"
    # else:
        # decision = "No Action"

    # return {
        # "delay_risk":      round(float(delay_risk), 3),
        # "delay_decision":  decision,
        # "shipping_mode":   input.shipping_mode,
        # "order_region":    input.order_region,
        # "category_name":   input.category_name
    # }

# @app.post("/")
# def read_root(item: DataInput):
    # return {"Hello": item.name, "Status": "Live on BTP"}

# if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 8080))
    # uvicorn.run(app, host="0.0.0.0", port=port)