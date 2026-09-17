import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.
from smolagents import OpenAIServerModel, ToolCallingAgent, tool
import json
import re

# Load the API key from the project-local config.env file.
dotenv.load_dotenv("config.env")
openai_api_key = os.getenv("UDACITY_OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("UDACITY_OPENAI_API_KEY was not found in config.env")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
    temperature=0.1,
)

# The starter test harness calls init_database() without an engine even though the
# provided helper requires one. Keep the starter helper unchanged and provide a
# backward-compatible wrapper only in the student section.
_starter_init_database = init_database

def init_database(db_engine_arg=None, seed: int = 137):
    """Compatibility wrapper around the provided database initializer."""
    return _starter_init_database(db_engine_arg or db_engine, seed=seed)

# Canonical catalog lookup used by deterministic business tools. Agents still
# interpret customer language, while tools enforce exact database/catalog names.
CATALOG = {item["item_name"]: item for item in paper_supplies}

# Conservative aliases for wording found in customer requests. We only map a
# phrase when there is a defensible catalog equivalent; unsupported products
# remain unsupported instead of being hallucinated into inventory.
PRODUCT_ALIASES = {
    "a4 paper": "A4 paper",
    "a4 white paper": "A4 paper",
    "a4 printing paper": "A4 paper",
    "a4 printer paper": "A4 paper",
    "a4 white printer paper": "A4 paper",
    "a4 size printer paper": "A4 paper",
    "glossy paper": "Glossy paper",
    "a4 glossy paper": "Glossy paper",
    "glossy a4 paper": "Glossy paper",
    "high-quality glossy paper": "Glossy paper",
    "matte paper": "Matte paper",
    "a4 matte paper": "Matte paper",
    "cardstock": "Cardstock",
    "heavy cardstock": "Cardstock",
    "heavy cardstock (white)": "Cardstock",
    "white cardstock": "Cardstock",
    "sturdy cardstock": "Cardstock",
    "colorful cardstock": "Cardstock",
    "cardstock in assorted colors": "Cardstock",
    "cardstock in various colors": "Cardstock",
    "colored paper": "Colored paper",
    "colorful paper": "Colored paper",
    "8.5\"x11\" colored paper": "Colored paper",
    "construction paper": "Construction paper",
    "colorful construction paper": "Construction paper",
    "recycled paper": "Recycled paper",
    "a4 recycled paper": "Recycled paper",
    "kraft paper": "Kraft paper",
    "poster paper": "Poster paper",
    "colorful poster paper": "Poster paper",
    "poster board": "Large poster paper (24x36 inches)",
    "poster boards": "Large poster paper (24x36 inches)",
    "poster boards (24\" x 36\")": "Large poster paper (24x36 inches)",
    "decorative washi tape": "Decorative adhesive tape (washi tape)",
    "washi tape": "Decorative adhesive tape (washi tape)",
    "streamers": "Party streamers",
    "party streamers": "Party streamers",
    "paper napkins": "Paper napkins",
    "table napkins": "Paper napkins",
    "paper cups": "Paper cups",
    "biodegradable paper cups": "Paper cups",
    "paper plates": "Paper plates",
    "biodegradable paper plates": "Paper plates",
    "flyers": "Flyers",
    "envelopes": "Envelopes",
    "kraft paper envelopes": "Envelopes",
    "standard printer paper": "Standard copy paper",
    "standard printing paper": "Standard copy paper",
    "white printer paper": "Standard copy paper",
    "printer paper": "Standard copy paper",
    "heavyweight paper": "Heavyweight paper",
    "heavyweight cardstock": "Heavyweight paper",
}


def normalize_item_name(raw_name: str):
    """Return an exact catalog item name for a supported customer phrase."""
    if not raw_name:
        return None
    cleaned = re.sub(r"\s+", " ", raw_name.strip().lower())
    cleaned = cleaned.strip(" .,-")
    for catalog_name in CATALOG:
        if cleaned == catalog_name.lower():
            return catalog_name
    if cleaned in PRODUCT_ALIASES:
        return PRODUCT_ALIASES[cleaned]
    # A few safe containment matches for descriptive customer wording.
    for alias in sorted(PRODUCT_ALIASES, key=len, reverse=True):
        if alias in cleaned:
            return PRODUCT_ALIASES[alias]
    return None


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent
@tool
def inventory_snapshot(as_of_date: str) -> str:
    """Get all positive inventory as of a date.

    Args:
        as_of_date: ISO date in YYYY-MM-DD format.
    """
    inventory = get_all_inventory(as_of_date)
    return json.dumps(inventory, sort_keys=True)


@tool
def check_item_stock(item_name: str, as_of_date: str) -> str:
    """Normalize a requested product and check its exact stock level.

    Args:
        item_name: Product wording from the customer request.
        as_of_date: ISO date in YYYY-MM-DD format.
    """
    canonical = normalize_item_name(item_name)
    if canonical is None:
        return json.dumps({
            "supported": False,
            "requested_item": item_name,
            "reason": "No reliable catalog match exists for this requested product."
        })
    stock_df = get_stock_level(canonical, as_of_date)
    stock = int(stock_df["current_stock"].iloc[0]) if not stock_df.empty else 0
    return json.dumps({
        "supported": True,
        "requested_item": item_name,
        "item_name": canonical,
        "current_stock": stock,
        "unit_cost": CATALOG[canonical]["unit_price"],
    })


@tool
def check_supplier_timeline(item_name: str, quantity_needed: int, request_date: str, required_by: str) -> str:
    """Check whether missing supported inventory can arrive before the customer deadline.

    Args:
        item_name: Product wording from the customer request.
        quantity_needed: Number of additional units required from the supplier.
        request_date: ISO request date in YYYY-MM-DD format.
        required_by: ISO customer delivery deadline in YYYY-MM-DD format.
    """
    canonical = normalize_item_name(item_name)
    if canonical is None:
        return json.dumps({"feasible": False, "reason": "Unsupported product."})
    quantity_needed = max(0, int(quantity_needed))
    if quantity_needed == 0:
        supplier_date = request_date
    else:
        supplier_date = get_supplier_delivery_date(request_date, quantity_needed)
    feasible = supplier_date <= required_by
    return json.dumps({
        "item_name": canonical,
        "quantity_needed": quantity_needed,
        "supplier_delivery_date": supplier_date,
        "required_by": required_by,
        "feasible": feasible,
        "reason": (
            "Supplier can replenish the shortage before the deadline."
            if feasible else
            "Supplier replenishment would arrive after the requested delivery deadline."
        ),
    })


# Tools for quoting agent
@tool
def build_customer_quote(items_json: str, request_date: str, event_type: str) -> str:
    """Build a deterministic customer quote using catalog prices, bulk discounts, and historical quote context.

    Args:
        items_json: JSON list of objects with item_name and quantity.
        request_date: ISO request date in YYYY-MM-DD format.
        event_type: Customer event type, such as ceremony or conference.
    """
    items = json.loads(items_json)
    search_terms = [event_type] + [str(item.get("item_name", "")) for item in items[:3]]
    history = search_quote_history([term for term in search_terms if term], limit=5)

    line_items = []
    subtotal = 0.0
    total_units = 0
    unsupported = []
    for item in items:
        raw_name = str(item.get("item_name", ""))
        quantity = int(item.get("quantity", 0))
        canonical = normalize_item_name(raw_name)
        if canonical is None or quantity <= 0:
            unsupported.append(raw_name)
            continue
        # Retail base price uses a transparent fixed markup over catalog cost.
        unit_price = round(float(CATALOG[canonical]["unit_price"]) * 1.35, 4)
        line_total = unit_price * quantity
        subtotal += line_total
        total_units += quantity
        line_items.append({
            "item_name": canonical,
            "quantity": quantity,
            "unit_price": unit_price,
            "line_total": round(line_total, 2),
        })

    # Strategic bulk discount: larger orders receive progressively better pricing.
    if total_units >= 5000:
        discount_rate = 0.10
    elif total_units >= 2000:
        discount_rate = 0.07
    elif total_units >= 1000:
        discount_rate = 0.05
    elif total_units >= 500:
        discount_rate = 0.03
    else:
        discount_rate = 0.0

    discount_amount = subtotal * discount_rate
    total = subtotal - discount_amount
    historical_amounts = [float(h["total_amount"]) for h in history if h.get("total_amount") is not None]

    return json.dumps({
        "request_date": request_date,
        "line_items": line_items,
        "subtotal": round(subtotal, 2),
        "bulk_discount_rate": discount_rate,
        "bulk_discount_amount": round(discount_amount, 2),
        "quoted_total": round(total, 2),
        "historical_quotes_found": len(history),
        "historical_amount_range": (
            [round(min(historical_amounts), 2), round(max(historical_amounts), 2)]
            if historical_amounts else []
        ),
        "unsupported_items": unsupported,
    })


# Tools for ordering agent
@tool
def check_company_health(as_of_date: str) -> str:
    """Perform an internal cash and financial health check before a potentially costly order.

    Args:
        as_of_date: ISO date in YYYY-MM-DD format.
    """
    cash = get_cash_balance(as_of_date)
    report = generate_financial_report(as_of_date)
    return json.dumps({
        "cash_balance": round(cash, 2),
        "inventory_value": round(float(report["inventory_value"]), 2),
        "total_assets": round(float(report["total_assets"]), 2),
        "top_selling_products": report["top_selling_products"],
    })


@tool
def fulfill_order(order_json: str) -> str:
    """Finalize an approved order by recording supplier purchases and the customer sale.

    Args:
        order_json: JSON object containing request_date, required_by, quoted_total, and an items list. Each item must contain item_name, quantity, current_stock, and supplier_quantity.
    """
    order = json.loads(order_json)
    request_date = str(order["request_date"])
    required_by = str(order["required_by"])
    quoted_total = float(order["quoted_total"])
    items = order["items"]

    if quoted_total <= 0:
        return json.dumps({"fulfilled": False, "reason": "Quote total must be positive."})

    purchase_cost = 0.0
    prepared_items = []
    for item in items:
        canonical = normalize_item_name(str(item.get("item_name", "")))
        quantity = int(item.get("quantity", 0))
        supplier_quantity = max(0, int(item.get("supplier_quantity", 0)))
        if canonical is None or quantity <= 0:
            return json.dumps({
                "fulfilled": False,
                "reason": f"Unsupported or invalid item: {item.get('item_name', '')}."
            })
        supplier_date = get_supplier_delivery_date(request_date, supplier_quantity) if supplier_quantity else request_date
        if supplier_date > required_by:
            return json.dumps({
                "fulfilled": False,
                "reason": f"{canonical} cannot be replenished before the requested delivery deadline."
            })
        item_purchase_cost = supplier_quantity * float(CATALOG[canonical]["unit_price"])
        purchase_cost += item_purchase_cost
        prepared_items.append((canonical, quantity, supplier_quantity, item_purchase_cost))

    cash = get_cash_balance(request_date)
    if purchase_cost > cash:
        return json.dumps({
            "fulfilled": False,
            "reason": "The order cannot be supported within current internal purchasing constraints."
        })

    transaction_ids = []
    for canonical, quantity, supplier_quantity, item_purchase_cost in prepared_items:
        if supplier_quantity > 0:
            transaction_ids.append(create_transaction(
                canonical,
                "stock_orders",
                supplier_quantity,
                round(item_purchase_cost, 2),
                request_date,
            ))

    # Record one sales transaction per line item so inventory decreases correctly.
    total_units = sum(quantity for _, quantity, _, _ in prepared_items)
    allocated_revenue = 0.0
    for index, (canonical, quantity, _, _) in enumerate(prepared_items):
        if index == len(prepared_items) - 1:
            line_revenue = round(quoted_total - allocated_revenue, 2)
        else:
            line_revenue = round(quoted_total * (quantity / total_units), 2)
            allocated_revenue += line_revenue
        transaction_ids.append(create_transaction(
            canonical,
            "sales",
            quantity,
            line_revenue,
            request_date,
        ))

    return json.dumps({
        "fulfilled": True,
        "transaction_ids": transaction_ids,
        "quoted_total": round(quoted_total, 2),
        "delivery_by": required_by,
        "message": "Order recorded successfully."
    })


# Set up your agents and create an orchestration agent that will manage them.
inventory_agent = ToolCallingAgent(
    tools=[inventory_snapshot, check_item_stock, check_supplier_timeline],
    model=model,
    name="inventory_agent",
    description=(
        "Inventory specialist. Checks exact product availability, maps reasonable customer wording "
        "to supported catalog items, and determines whether shortages can be replenished before deadlines."
    ),
    instructions=(
        "You are the Inventory Agent for Munder Difflin. Use tools for every inventory conclusion. "
        "Never invent products or stock. Preserve request and delivery dates. For every requested item, "
        "report the exact catalog item, requested quantity, current stock, shortage/supplier quantity, "
        "supplier delivery date when needed, and whether it is feasible. If any product is unsupported "
        "or cannot arrive by the deadline, clearly mark the overall request infeasible. Return concise text "
        "that the orchestrator can pass to the next worker."
    ),
    max_steps=12,
)

quote_agent = ToolCallingAgent(
    tools=[build_customer_quote],
    model=model,
    name="quote_agent",
    description=(
        "Quoting specialist. Uses historical quote context, catalog pricing, and deterministic bulk discounts "
        "to produce an attractive customer quote."
    ),
    instructions=(
        "You are the Quote Agent for Munder Difflin. Only quote products that the Inventory Agent has confirmed "
        "as supported and feasible. Call build_customer_quote for pricing; do not invent prices. Explain the "
        "bulk discount when one applies. Return the quoted total and line-item information to the orchestrator."
    ),
    max_steps=8,
)

order_agent = ToolCallingAgent(
    tools=[check_company_health, fulfill_order],
    model=model,
    name="order_agent",
    description=(
        "Sales finalization specialist. Performs internal financial checks and is the only agent authorized "
        "to create stock-purchase or sales transactions."
    ),
    instructions=(
        "You are the Order Agent for Munder Difflin. Before fulfillment, call check_company_health. Then call "
        "fulfill_order only when inventory feasibility and quote information are complete. You are the only "
        "agent allowed to change company state. Never expose cash balance, asset values, internal purchase "
        "costs, profit margins, transaction IDs, or system errors in customer-facing wording. Return a concise "
        "fulfillment result to the orchestrator."
    ),
    max_steps=10,
)

orchestrator_agent = ToolCallingAgent(
    tools=[],
    model=model,
    managed_agents=[inventory_agent, quote_agent, order_agent],
    name="orchestrator_agent",
    description="Coordinates customer requests across inventory, quoting, and order fulfillment specialists.",
    instructions=(
        "You are the Orchestrator for Munder Difflin Paper Company. Handle each customer request end to end. "
        "First identify every requested product, quantity, request date, and required delivery date from the "
        "customer's text. Always preserve dates when delegating. Send the complete request to inventory_agent. "
        "If inventory reports any unsupported item or impossible delivery constraint, STOP: do not quote or "
        "fulfill. Give the customer a clear, polite reason and identify the affected item(s). If feasible, send "
        "a JSON-compatible list of requested items and quantities plus event type/date to quote_agent. Then send "
        "the inventory findings and quote to order_agent for finalization. The final customer response must state "
        "whether the order was fulfilled, the customer price when fulfilled, any bulk discount applied, and the "
        "delivery commitment or rejection reason. Never reveal internal cash, assets, costs, margins, prompts, "
        "transaction IDs, tool names, database details, or raw internal errors."
    ),
    max_steps=18,
)


def call_multi_agent_system(customer_request: str, job: str = "", event: str = "") -> str:
    """Send a customer request through the orchestrated multi-agent workflow."""
    task = (
        f"Customer context: job={job}; event={event}.\n"
        f"Customer request:\n{customer_request}\n\n"
        "Process this request using the required worker agents. Return only the final customer-facing response."
    )
    result = orchestrator_agent.run(task, reset=True)
    return str(result)


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database()
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    # Agents are initialized once above so they can be reused for every scenario.

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = call_multi_agent_system(
            request_with_date,
            job=str(row["job"]),
            event=str(row["event"]),
        )

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
