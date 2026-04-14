import pandas as pd
import numpy as np
import re
from fractions import Fraction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Cookbook | ML Recipe Finder",
    page_icon="🍳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CUSTOM CSS (SDE Tip: Customizing UI shows attention to UX)
# ──────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #41ABAB; }
    .recipe-card {
        background: black;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 5px solid #E07A5F;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .badge-green  { background:#81B29A; color:white; padding:3px 10px; border-radius:20px; font-size:13px; font-weight:600; }
    .badge-yellow { background:#F2CC8F; color:#7a5c00; padding:3px 10px; border-radius:20px; font-size:13px; font-weight:600; }
    .badge-red    { background:#E07A5F; color:white; padding:3px 10px; border-radius:20px; font-size:13px; font-weight:600; }
    .section-title { font-size:15px; font-weight:700; color:#3D405B; margin-top:14px; margin-bottom:4px; }
    .pill {
        display:inline-block;
        background:#F4F1DE;
        border:1px solid #ddd;
        border-radius:20px;
        padding:2px 10px;
        margin:3px 2px;
        font-size:13px;
        color:#3D405B;
    }
    .metric-box {
        background:#F4F1DE;
        border-radius:8px;
        padding:10px 14px;
        text-align:center;
    }
    .metric-box .val { font-size:20px; font-weight:700; color:#3D405B; }
    .metric-box .lbl { font-size:11px; color:#888; margin-top:2px; }
    header { visibility: hidden; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HELPER FUNCTIONS (SDE Tip: Using Regex shows strong string manipulation skills)
# ──────────────────────────────────────────────

def time_to_minutes(time_str):
    """Converts diverse time strings (e.g., '1 hr 20 mins') into total minutes."""
    if pd.isnull(time_str):
        return None
    time_str = str(time_str).lower()
    hours   = re.search(r'(\d+)\s*hrs?', time_str)
    minutes = re.search(r'(\d+)\s*mins?', time_str)
    total   = 0
    if hours:
        total += int(hours.group(1)) * 60
    if minutes:
        total += int(minutes.group(1))
    return total if total > 0 else None

def parse_yield(value):
    """Parses fractional yields like '1 1/2' using the fractions library."""
    try:
        return float(sum(Fraction(part) for part in str(value).split()))
    except (ValueError, TypeError, AttributeError):
        return np.nan

def fill_missing_times(row):
    """Logically derives missing time values based on available data points."""
    prep  = row['prep_time']
    cook  = row['cook_time']
    total = row['total_time']
    if pd.isnull(total) and not pd.isnull(prep) and not pd.isnull(cook):
        total = prep + cook
    if pd.isnull(cook) and not pd.isnull(total) and not pd.isnull(prep):
        cook = total - prep
    if pd.isnull(prep) and not pd.isnull(total) and not pd.isnull(cook):
        prep = total - cook
    row['prep_time']  = max(prep  if not pd.isnull(prep)  else 0, 0)
    row['cook_time']  = max(cook  if not pd.isnull(cook)  else 0, 0)
    row['total_time'] = max(total if not pd.isnull(total) else 0, 0)
    return row

def rating_to_stars(rating):
    full  = int(rating)
    half  = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty

def minutes_to_str(mins):
    try:
        mins = int(round(float(mins)))
    except (ValueError, TypeError):
        return "N/A"
    if mins <= 0:
        return "N/A"
    h, m = divmod(mins, 60)
    if h and m:
        return f"{h}h {m}m"
    elif h:
        return f"{h}h"
    return f"{m}m"

# Updated FEATURE_COLS to include cuisine_path for better model performance
FEATURE_COLS = [
    'prep_time', 'cook_time', 'total_time', 'yield',
    'total_time_per_serving', 'ingredient_count', 
    'prep_cook_interaction', 'cuisine_path'
]

# ──────────────────────────────────────────────
# DATA LOADING & MODEL TRAINING
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Training model... optimizing features for predictive accuracy.")
def load_and_train():
    try:
        data = pd.read_csv('recipes.csv')
    except FileNotFoundError:
        st.error("Dataset 'recipes.csv' not found. Please ensure it is in the project directory.")
        return None, None, 0, 0

    # Preprocessing Time Columns
    for col in ['prep_time', 'cook_time', 'total_time']:
        data[col] = data[col].apply(time_to_minutes)
    data = data.apply(fill_missing_times, axis=1)
    for col in ['prep_time', 'cook_time', 'total_time']:
        data[col].fillna(data[col].median(), inplace=True)

    data['yield'] = data['yield'].apply(parse_yield)
    data['yield'].fillna(data['yield'].median(), inplace=True)

    # Store raw times for display purposes
    data['_prep_raw']  = data['prep_time']
    data['_cook_raw']  = data['cook_time']
    data['_total_raw'] = data['total_time']

    # Advanced Feature Engineering
    data['total_time_per_serving'] = (data['total_time'] / data['yield'].replace(0, np.nan)).fillna(0)
    data['ingredient_count'] = data['ingredients'].apply(lambda x: len(str(x).split(',')) if isinstance(x, str) else 0)
    data['prep_cook_interaction'] = data['prep_time'] * data['cook_time']

    # Label Encoding for Categorical Data
    le = LabelEncoder()
    if 'cuisine_path' in data.columns:
        data['cuisine_path'] = le.fit_transform(data['cuisine_path'].astype(str))
    else:
        data['cuisine_path'] = 0 # Fallback if column is missing

    # Feature Scaling
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[FEATURE_COLS] = scaler.fit_transform(data_scaled[FEATURE_COLS])

    # Model Training
    data_model = data_scaled.dropna(subset=['rating']).copy()
    X = data_model[FEATURE_COLS]
    y = data_model['rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Increased estimators and added max_depth to prevent overfitting and improve generalization
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    mse = mean_squared_error(y_test, model.predict(X_test))
    r2  = r2_score(y_test, model.predict(X_test))

    return data, model, mse, r2

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.title("🍳 Cookbook")
    st.caption("Developed by Raghav Verman")
    st.divider()

    st.subheader("Discovery Filters")
    max_cook = st.slider("Max cook time (minutes)", 10, 300, 120, step=10)
    min_rating = st.slider("Min predicted rating", 1.0, 5.0, 3.0, step=0.5)
    sort_by = st.selectbox(
        "Sort results by",
        ["Predicted Rating ↓", "Cook Time ↑", "Ingredient Count ↑"]
    )

    st.divider()
    # Displaying model metrics in sidebar shows transparency in development
    data, model, mse, r2 = load_and_train()
    if model:
        st.metric("Model R² Score", f"{r2:.4f}")
        st.caption("Model trained on standard recipe features including cuisine and complexity.")

# ──────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────

st.title("🍳 Recipe Finder")
st.markdown("Find recipes based on ingredients you have, **ranked by predicted user preference**.")

# Handle image display with safety check
try:
    img = Image.open("ima.png")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(img, use_container_width=True)
except FileNotFoundError:
    pass

query = st.text_input(
    "🔍 Search Ingredients",
    placeholder="e.g. chicken, garlic, lemon",
    label_visibility="collapsed"
)

# ──────────────────────────────────────────────
# SEARCH LOGIC
# ──────────────────────────────────────────────

if query and data is not None:
    terms = [t.strip() for t in query.split(',') if t.strip()]
    mask = pd.Series([True] * len(data), index=data.index)
    for term in terms:
        mask &= data['ingredients'].str.contains(term, case=False, na=False)
    matches = data[mask].copy()

    # Application of filters
    matches = matches[matches['_cook_raw'] <= max_cook]

    if matches.empty:
        st.warning("No recipes found matching those ingredients.")
    else:
        # Generate predictions for the filtered subset
        # SDE Note: We apply the same scaling logic here as we did in training
        scaler = StandardScaler()
        matches_for_pred = matches[FEATURE_COLS].copy()
        matches_for_pred = scaler.fit_transform(matches_for_pred)
        matches['predicted_rating'] = model.predict(matches_for_pred)

        matches = matches[matches['predicted_rating'] >= min_rating]

        if matches.empty:
            st.warning("No high-rated recipes found. Try lowering the 'Min predicted rating' filter.")
        else:
            # Sorting logic
            if sort_by == "Predicted Rating ↓":
                matches = matches.sort_values('predicted_rating', ascending=False)
            elif sort_by == "Cook Time ↑":
                matches = matches.sort_values('_cook_raw', ascending=True)
            else:
                matches = matches.sort_values('ingredient_count', ascending=True)

            st.success(f"Found **{len(matches)}** curated recipes.")

            for _, row in matches.iterrows():
                pred = row['predicted_rating']
                with st.expander(f"**{row['recipe_name']}** — {rating_to_stars(pred)} {pred:.1f}"):
                    m1, m2, m3, m4 = st.columns(4)
                    m1.markdown(f'<div class="metric-box"><div class="val">{pred:.1f}</div><div class="lbl">Predicted</div></div>', unsafe_allow_html=True)
                    m2.markdown(f'<div class="metric-box"><div class="val">{minutes_to_str(row["_prep_raw"])}</div><div class="lbl">Prep</div></div>', unsafe_allow_html=True)
                    m3.markdown(f'<div class="metric-box"><div class="val">{minutes_to_str(row["_cook_raw"])}</div><div class="lbl">Cook</div></div>', unsafe_allow_html=True)
                    m4.markdown(f'<div class="metric-box"><div class="val">{int(row["ingredient_count"])}</div><div class="lbl">Items</div></div>', unsafe_allow_html=True)

                    st.markdown('<div class="section-title">🧂 Ingredients</div>', unsafe_allow_html=True)
                    ingredients = [i.strip() for i in str(row.get('ingredients', '')).split(',')]
                    st.markdown(" ".join(f'<span class="pill">{i}</span>' for i in ingredients if i), unsafe_allow_html=True)

                    st.markdown('<div class="section-title">📋 Directions</div>', unsafe_allow_html=True)
                    directions = str(row.get('directions', 'N/A'))
                    steps = [s.strip() for s in re.split(r'\.\s+|\n+', directions) if len(s.strip()) > 10]
                    if len(steps) > 1:
                        for i, step in enumerate(steps, 1): st.markdown(f"**{i}.** {step}")
                    else:
                        st.write(directions)

elif data is not None:
    st.markdown("""
    <div style='text-align:center; padding:60px 20px; color:#eee;'>
        <div style='font-size:52px;'>🥘</div>
        <div style='font-size:18px; margin-top:12px;'>Search by ingredients to see ML-ranked recipes</div>
        <div style='font-size:13px; margin-top:6px;'>e.g. <em>pasta, mushrooms</em></div>
    </div>
    """, unsafe_allow_html=True)
