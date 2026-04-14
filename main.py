import pandas as pd
import numpy as np
import re
from fractions import Fraction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# ──────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Cookbook",
    page_icon="🍳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #41ABAB; }

    /* Recipe card */
    .recipe-card {
        background: black;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 5px solid #E07A5F;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }

    /* Rating badge */
    .badge-green  { background:#81B29A; color:white; padding:3px 10px; border-radius:20px; font-size:13px; font-weight:600; }
    .badge-yellow { background:#F2CC8F; color:#7a5c00; padding:3px 10px; border-radius:20px; font-size:13px; font-weight:600; }
    .badge-red    { background:#E07A5F; color:white; padding:3px 10px; border-radius:20px; font-size:13px; font-weight:600; }

    /* Section headings inside expander */
    .section-title { font-size:15px; font-weight:700; color:#3D405B; margin-top:14px; margin-bottom:4px; }

    /* Ingredient pill */
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

    /* Metric box */
    .metric-box {
        background:#F4F1DE;
        border-radius:8px;
        padding:10px 14px;
        text-align:center;
    }
    .metric-box .val { font-size:20px; font-weight:700; color:#3D405B; }
    .metric-box .lbl { font-size:11px; color:#888; margin-top:2px; }

    /* Hide default streamlit header decoration */
    header { visibility: hidden; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def time_to_minutes(time_str):
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
    try:
        return float(sum(Fraction(part) for part in str(value).split()))
    except (ValueError, TypeError, AttributeError):
        return np.nan


def fill_missing_times(row):
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


def rating_badge(rating):
    stars = rating_to_stars(rating)
    if rating >= 4.0:
        cls = "badge-green"
    elif rating >= 3.0:
        cls = "badge-yellow"
    else:
        cls = "badge-red"
    return f'<span class="{cls}">{stars} {rating:.1f}</span>'


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


FEATURE_COLS = [
    'prep_time', 'cook_time', 'total_time', 'yield',
    'total_time_per_serving', 'ingredient_count', 'prep_cook_interaction'
]


# ──────────────────────────────────────────────
# DATA LOADING & MODEL TRAINING (cached)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Training model… this takes ~10 seconds on first load.")
def load_and_train():
    data = pd.read_csv('recipes.csv')

    # Time columns
    for col in ['prep_time', 'cook_time', 'total_time']:
        data[col] = data[col].apply(time_to_minutes)
    data = data.apply(fill_missing_times, axis=1)
    for col in ['prep_time', 'cook_time', 'total_time']:
        data[col].fillna(data[col].median(), inplace=True)

    # Yield
    data['yield'] = data['yield'].apply(parse_yield)
    data['yield'].fillna(data['yield'].median(), inplace=True)

    # Store raw times before scaling (for display)
    data['_prep_raw']  = data['prep_time']
    data['_cook_raw']  = data['cook_time']
    data['_total_raw'] = data['total_time']

    # Feature engineering
    data['total_time_per_serving'] = (
        data['total_time'] / data['yield'].replace(0, np.nan)
    ).fillna(0)
    data['ingredient_count'] = data['ingredients'].apply(
        lambda x: len(str(x).split(',')) if isinstance(x, str) else 0
    )
    data['prep_cook_interaction'] = data['prep_time'] * data['cook_time']

    # Encode categoricals
    le = LabelEncoder()
    if 'cuisine_path' in data.columns:
        data['cuisine_path'] = le.fit_transform(data['cuisine_path'].astype(str))

    # Scale features
    scaler = StandardScaler()
    data[FEATURE_COLS] = scaler.fit_transform(data[FEATURE_COLS])

    # Train model on rows that have ratings
    data_model = data.dropna(subset=['rating']).copy()
    X = data_model[FEATURE_COLS]
    y = data_model['rating']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    mse = mean_squared_error(y_test, model.predict(X_test))
    r2  = r2_score(y_test, model.predict(X_test))

    return data, model, mse, r2


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.title("🍳 Cookbook")
    st.caption("ML-powered recipe finder")
    st.divider()

    st.subheader("Filters")
    max_cook = st.slider("Max cook time (minutes)", 10, 300, 120, step=10)
    min_rating = st.slider("Min predicted rating", 1.0, 5.0, 3.0, step=0.5)
    sort_by = st.selectbox(
        "Sort results by",
        ["Predicted Rating ↓", "Cook Time ↑", "Ingredient Count ↑"]
    )

    st.divider()
    st.caption("Built with scikit-learn + Streamlit")


# ──────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────

st.title("🍳 Recipe Finder")
st.markdown("Enter ingredients you have and get recipes **ranked by predicted rating**.")


from PIL import Image
img = Image.open("ima.png")
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image(img, use_container_width=True)
# Load data + model
data, model, mse, r2 = load_and_train()



# Search bar
query = st.text_input(
    "🔍 Ingredients",
    placeholder="e.g. chicken, garlic, lemon",
    label_visibility="collapsed"
)

# ──────────────────────────────────────────────
# SEARCH RESULTS
# ──────────────────────────────────────────────

if query:
    terms = [t.strip() for t in query.split(',') if t.strip()]

    # Filter: all terms must appear in ingredients
    mask = pd.Series([True] * len(data), index=data.index)
    for term in terms:
        mask &= data['ingredients'].str.contains(term, case=False, na=False)
    matches = data[mask].copy()

    # Apply cook time filter (use raw time column)
    matches = matches[matches['_cook_raw'] <= max_cook]

    if matches.empty:
        st.warning("No recipes found. Try fewer ingredients or increase the cook time filter.")
    else:
        # Predict ratings
        matches['predicted_rating'] = model.predict(matches[FEATURE_COLS])

        # Apply min rating filter
        matches = matches[matches['predicted_rating'] >= min_rating]

        if matches.empty:
            st.warning("No recipes meet your rating filter. Try lowering the minimum rating.")
        else:
            # Sort
            if sort_by == "Predicted Rating ↓":
                matches = matches.sort_values('predicted_rating', ascending=False)
            elif sort_by == "Cook Time ↑":
                matches = matches.sort_values('_cook_raw', ascending=True)
            else:
                matches = matches.sort_values('ingredient_count', ascending=True)

            st.success(f"Found **{len(matches)}** recipe(s) matching: _{', '.join(terms)}_")

            # ── Recipe cards ──
            for _, row in matches.iterrows():
                predicted = row['predicted_rating']
                badge     = rating_badge(predicted)
                cook_str  = minutes_to_str(row['_cook_raw'])
                prep_str  = minutes_to_str(row['_prep_raw'])
                total_str = minutes_to_str(row['_total_raw'])
                ing_count = int(row['ingredient_count'])

                with st.expander(f"**{row['recipe_name']}**   {rating_to_stars(predicted)} {predicted:.1f}"):

                    # Top metrics row
                    m1, m2, m3, m4 = st.columns(4)
                    m1.markdown(
                        f'<div class="metric-box"><div class="val">{predicted:.1f}</div>'
                        f'<div class="lbl">Predicted rating</div></div>',
                        unsafe_allow_html=True
                    )
                    m2.markdown(
                        f'<div class="metric-box"><div class="val">{prep_str}</div>'
                        f'<div class="lbl">Prep time</div></div>',
                        unsafe_allow_html=True
                    )
                    m3.markdown(
                        f'<div class="metric-box"><div class="val">{cook_str}</div>'
                        f'<div class="lbl">Cook time</div></div>',
                        unsafe_allow_html=True
                    )
                    m4.markdown(
                        f'<div class="metric-box"><div class="val">{ing_count}</div>'
                        f'<div class="lbl">Ingredients</div></div>',
                        unsafe_allow_html=True
                    )

                    st.markdown("")

                    # Ingredients as pills
                    st.markdown('<div class="section-title">🧂 Ingredients</div>', unsafe_allow_html=True)
                    ingredients = [i.strip() for i in str(row.get('ingredients', '')).split(',')]
                    pills_html  = " ".join(f'<span class="pill">{i}</span>' for i in ingredients if i)
                    st.markdown(pills_html, unsafe_allow_html=True)

                    # Directions as numbered steps
                    st.markdown('<div class="section-title">📋 Directions</div>', unsafe_allow_html=True)
                    directions_raw = str(row.get('directions', 'N/A'))
                    # Try to split on common delimiters
                    steps = re.split(r'\.\s+|\n+|(?<=\d)\.\s', directions_raw)
                    steps = [s.strip() for s in steps if len(s.strip()) > 10]
                    if len(steps) > 1:
                        for i, step in enumerate(steps, 1):
                            st.markdown(f"**{i}.** {step}")
                    else:
                        st.write(directions_raw)

                    # Nutrition
                    nutrition = str(row.get('nutrition', ''))
                    if nutrition and nutrition.lower() not in ('nan', 'none', ''):
                        st.markdown('<div class="section-title">🥗 Nutrition</div>', unsafe_allow_html=True)
                        st.caption(nutrition)

else:
    # Empty state
    st.markdown("""
    <div style='text-align:center; padding:60px 20px; color:#aaa;'>
        <div style='font-size:52px;'>🥘</div>
        <div style='font-size:18px; margin-top:12px;'>Enter ingredients above to find recipes</div>
        <div style='font-size:13px; margin-top:6px;'>Try: <em>chicken, garlic</em> or <em>pasta, tomato</em></div>
    </div>
    """, unsafe_allow_html=True)