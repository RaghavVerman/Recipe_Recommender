
#  Cookbook: ML-Powered Recipe Finder

**Cookbook** is a data-driven web application that leverages Machine Learning to predict recipe quality and rank culinary results based on user-provided ingredients. Unlike traditional search engines, this app uses a **Random Forest Regressor** to estimate a recipe's rating, ensuring that the most "highly-rated" options are prioritized.

##  Technical Highlights
* **Intelligent Ingredient Search:** Implemented a robust filtering system that handles multiple ingredients simultaneously.
* **Predictive Ranking:** Every result is passed through a trained ML model to generate a predicted preference score (1.0–5.0 stars).
* **Dynamic UI/UX:** Built a high-contrast "Cookbook" dashboard with custom CSS injection for a professional feel.

##  The Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Random Forest, StandardScaler, LabelEncoder)
* **UI/UX:** Streamlit with Custom CSS
* **Utilities:** Regex (for time parsing), Fractions (for yield calculation)

##  Software Engineering Logic
This project demonstrates several core SDE competencies:
1. **Robust Data Parsing:** Uses **Regular Expressions (Regex)** to convert strings like "1 hr 20 mins" into standardized integers.
2. **Advanced Feature Engineering:** Includes custom-calculated metrics such as **Prep-Cook Interaction** ($Prep \times Cook$) and **Cuisine Mapping** to capture complex data relationships.
3. **Data Imputation:** Implemented logical derivation to fill missing time values based on available prep/cook/total data points.
