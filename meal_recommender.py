"""
meal_recommender.py

Updated: integrates K-Means clustering (k=4) to automatically map clusters to meal types:
- cluster -> {breakfast, lunch, dinner, snack} by centroid calorie heuristic

Usage unchanged:
    from meal_recommender import MealRecommender
    mr = MealRecommender(food_csv="food1_cleaned.csv", users_csv="users_with_city.csv")
    plan = mr.recommend_meal_plan(calorie_target=2200, preference='veg', temp=30)
"""

import os
import re
import math
import pickle
import requests
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# -----------------------------
# Config & Aliases
# -----------------------------
DEFAULT_FEATURES = [
    "Data.Protein", "Data.Fat", "Data.Carbohydrate", "Data.Energy",
    "Data.Fiber", "Data.Sodium", "Data.Vitamins.VitaminC", "Data.Calcium", "Data.Iron"
]

COLUMN_ALIASES = {
    'calories': ['Data.Energy', 'Data.Calories', 'Calories', 'Energy', 'ENERC_KCAL', 'Data.Energy (kcal)'],
    'protein': ['Data.Protein', 'Protein', 'PROCNT'],
    'fat': ['Data.Fat', 'Fat', 'FAT'],
    'carbs': ['Data.Carbohydrate', 'Carbohydrate', 'Carbs', 'CHOCDF'],
    'fiber': ['Data.Fiber', 'Fiber', 'FIBTG'],
    'sodium': ['Data.Sodium', 'Sodium', 'NA', 'SODIUM'],
    'vitamin_c': ['Data.Vitamins.VitaminC', 'Vitamin C', 'VitC', 'VITC'],
    'calcium': ['Data.Calcium', 'Calcium', 'CA'],
    'iron': ['Data.Iron', 'Iron', 'FE'],
    # user aliases
    'age': ['age', 'Age', 'user_age'],
    'gender': ['gender', 'sex', 'Gender'],
    'height': ['height', 'height_cm', 'Height', 'Height_cm'],
    'weight': ['weight', 'weight_kg', 'Weight', 'Weight_kg'],
    'activity': ['activity_level', 'activity', 'Activity', 'physical_activity'],
    'goal': ['goal', 'Goal', 'diet_goal'],
    'preference': ['preference', 'pref', 'diet_pref'],
    'allergies': ['allergies', 'allergy', 'Allergies'],
    'city': ['city', 'City', 'user_city', 'location']
}

MEAL_CATEGORIES = {
    'breakfast': ['DAIRY', 'BREAD', 'CEREAL', 'FRUIT', 'EGG', 'YOGURT', 'PASTRY'],
    'lunch': ['RICE', 'PULSE', 'VEGETABLE', 'MEAT', 'FISH', 'CHEESE', 'BEAN'],
    'dinner': ['RICE', 'MEAT', 'FISH', 'VEGETABLE', 'CHEESE', 'PULSES'],
    'snack': ['NUTS', 'FRUIT', 'SWEET', 'SNACK', 'BEVERAGE']
}

GOAL_MACRO_RATIOS = {
    'lose': {'protein': 0.30, 'carbs': 0.35, 'fat': 0.35},
    'gain': {'protein': 0.25, 'carbs': 0.50, 'fat': 0.25},
    'maintain': {'protein': 0.25, 'carbs': 0.45, 'fat': 0.30}
}

DEFAULT_MEAL_SPLIT = {'breakfast': 0.25, 'lunch': 0.35, 'dinner': 0.30, 'snack': 0.10}

ACTIVITY_FACTORS = {
    'sedentary': 1.2,
    'light': 1.375,
    'moderate': 1.55,
    'active': 1.725,
    'very_active': 1.9
}

# -----------------------------
# Helper utilities
# -----------------------------
def _find_column(df: pd.DataFrame, token: str) -> Optional[str]:
    token = token.lower()
    if token in COLUMN_ALIASES:
        for alias in COLUMN_ALIASES[token]:
            if alias in df.columns:
                return alias
            normalized = [c for c in df.columns if c.lower().replace(' ', '').replace('_','') == alias.lower().replace(' ', '').replace('_','')]
            if normalized:
                return normalized[0]
    for c in df.columns:
        if token in c.lower():
            return c
    return None

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# -----------------------------
# Main Class
# -----------------------------
class MealRecommender:
    def __init__(self,
                 food_csv: str = '/mnt/data/food1_cleaned.csv',
                 users_csv: Optional[str] = '/mnt/data/users_with_city.csv',
                 feature_columns: Optional[List[str]] = None,
                 cluster_k: Optional[int] = 4):
        if not os.path.exists(food_csv):
            raise FileNotFoundError(f"Food CSV not found: {food_csv}")
        self.food_csv = food_csv
        self.users_csv = users_csv
        self.df = pd.read_csv(food_csv)
        self._normalize_columns()
        self._ensure_description_and_category()
        self._add_dietary_flags()
        self.feature_columns = self._resolve_features(feature_columns or DEFAULT_FEATURES)
        self._fill_missing_numeric()
        self.scaler = StandardScaler()
        self.food_matrix = None
        self.kmeans = None
        self.cluster_k = cluster_k
        self.cluster_meal_map = {}
        self._build_food_vectors()
        if cluster_k and cluster_k > 1:
            self._build_clusters(cluster_k)
        self.users_df = None
        if users_csv and os.path.exists(users_csv):
            try:
                self.users_df = pd.read_csv(users_csv)
                self._normalize_user_columns()
            except Exception as e:
                print("Failed to load users CSV:", e)
                self.users_df = None

    def _normalize_columns(self):
        self.df.columns = [c.strip() for c in self.df.columns]

    def _ensure_description_and_category(self):
        cols_lower = [c.lower() for c in self.df.columns]
        if 'description' not in cols_lower:
            candidates = [c for c in self.df.columns if 'desc' in c.lower()]
            if candidates:
                self.df.rename(columns={candidates[0]: 'Description'}, inplace=True)
            else:
                self.df['Description'] = self.df.get('Category', '').astype(str) + ' ' + self.df.index.astype(str)
        if 'category' not in cols_lower:
            candidates = [c for c in self.df.columns if 'cat' in c.lower()]
            if candidates:
                self.df.rename(columns={candidates[0]: 'Category'}, inplace=True)
        self.df['Description'] = self.df['Description'].astype(str)
        if 'Category' not in self.df.columns:
            self.df['Category'] = self.df['Description'].str.upper().str.split().str[0].fillna('MISC')

    def _add_dietary_flags(self):
        meat_tokens = ['BEEF','CHICKEN','CHICK','PORK','LAMB','MUTTON','DUCK','GOOSE','HAM','BACON','MEAT','VEAL']
        fish_seafood_tokens = ['FISH','SEAFOOD','TUNA','SALMON','COD','SHRIMP','CRAB','LOBSTER','OYSTER','CLAM','MUSSEL']
        egg_tokens = ['EGG','OMELET','OMELETTE']
        desc_upper = self.df['Description'].str.upper()
        self.df['is_meat_item'] = desc_upper.apply(lambda s: any(tok in s for tok in meat_tokens) and not any(tok in s for tok in fish_seafood_tokens))
        self.df['is_fish_item'] = desc_upper.apply(lambda s: any(tok in s for tok in fish_seafood_tokens))
        self.df['is_egg_item'] = desc_upper.apply(lambda s: any(tok in s for tok in egg_tokens))
        self.df['is_strictly_vegetarian'] = ~(self.df['is_meat_item'] | self.df['is_fish_item'] | self.df['is_egg_item'])

    def _resolve_features(self, features):
        resolved = []
        cols = set(self.df.columns)
        for f in features:
            if f in cols:
                resolved.append(f)
                continue
            for alias_list in COLUMN_ALIASES.values():
                for alias in alias_list:
                    if alias in cols and f.lower().replace('.','').replace('_','') in alias.lower().replace('.','').replace('_',''):
                        resolved.append(alias)
                        break
        if not any(r for r in resolved if f.lower() in r.lower()):
            for c in self.df.columns:
                if f.split('.')[-1].lower() in c.lower():
                    resolved.append(c)
                    break
        if not any(re.search('calor|ener', c.lower()) for c in resolved):
            for c in self.df.columns:
                if re.search('calor|ener', c.lower()):
                    resolved.append(c)
                    break
        for token in ['protein', 'fat', 'carbo']:
            for c in self.df.columns:
                if token in c.lower() and c not in resolved:
                    resolved.append(c)
                    break
        resolved = list(dict.fromkeys(resolved))
        if not resolved:
            raise ValueError("No numeric features resolved.")
        return resolved

    def _fill_missing_numeric(self):
        for c in self.feature_columns:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce').fillna(0.0)

    # -------------------------
    # Vectorization & clustering
    # -------------------------
    def _build_food_vectors(self):
        features = [c for c in self.feature_columns if c in self.df.columns]
        X = self.df[features].astype(float).values
        X_scaled = self.scaler.fit_transform(X)
        X_norm = normalize(X_scaled, axis=1)
        self.food_matrix = X_norm

    def _build_clusters(self, k=4):
        """
        Fit KMeans on normalized food vectors and map clusters -> meal types.
        Mapping heuristic:
          - cluster with highest centroid calories -> dinner
          - 2nd highest -> lunch
          - 3rd -> breakfast
          - lowest -> snack
        """
        if self.food_matrix is None:
            raise RuntimeError("Food matrix not built.")
        try:
            self.kmeans = KMeans(n_clusters=k, random_state=42).fit(self.food_matrix)
            self.df['food_cluster'] = self.kmeans.labels_
            centroids = self.kmeans.cluster_centers_
            cal_col = None
            for i, col in enumerate(self.feature_columns):
                if re.search('calor|ener', col.lower()):
                    cal_col = i
                    break
            cluster_scores = []
            for cid in range(k):
                if cal_col is not None:
                    score = centroids[cid, cal_col]
                else:
                    score = centroids[cid].sum()
                cluster_scores.append((cid, float(score)))
            cluster_scores.sort(key=lambda x: x[1], reverse=True)
            order = [c for c, _ in cluster_scores]
            meal_names = ['dinner', 'lunch', 'breakfast', 'snack']
            self.cluster_meal_map = {}
            for idx, cluster_id in enumerate(order):
                meal_name = meal_names[idx] if idx < len(meal_names) else f"cluster_{idx}"
                self.cluster_meal_map[cluster_id] = meal_name
            self.meal_to_clusters = {}
            for cid, meal in self.cluster_meal_map.items():
                self.meal_to_clusters.setdefault(meal, []).append(cid)
        except Exception as e:
            print("Clustering failed:", e)
            self.kmeans = None
            self.cluster_meal_map = {}
            self.meal_to_clusters = {}

    # -------------------------
    # Health score
    # -------------------------
    def _choose_col(self, token):
        token = token.lower()
        for key, alias_list in COLUMN_ALIASES.items():
            if token == key:
                for a in alias_list:
                    if a in self.df.columns:
                        return a
        for c in self.df.columns:
            if token in c.lower():
                return c
        return None

    def compute_health_score(self, row: pd.Series) -> float:
        protein = float(row.get(self._choose_col('protein'), 0.0) or 0.0)
        fiber = float(row.get(self._choose_col('fiber'), 0.0) or 0.0)
        sodium = float(row.get(self._choose_col('sodium'), 0.0) or 0.0)
        vitc = float(row.get(self._choose_col('vitamin_c'), 0.0) or 0.0)
        calcium = float(row.get(self._choose_col('calcium'), 0.0) or 0.0)
        iron = float(row.get(self._choose_col('iron'), 0.0) or 0.0)
        calories = float(row.get(self._choose_col('calories'), 100.0) or 100.0)
        score = 0.0
        score += protein * 4.0
        score += fiber * 3.0
        score += (vitc / 2.0)
        score += (calcium / 100.0)
        score += (iron / 1.0)
        score -= (sodium / 200.0)
        score -= max(0, (calories - 250) / 50.0)
        score = max(0.0, score)
        return score
    # -------------------------
    # Recommend wrappers for users
    # -------------------------
    def _normalize_user_columns(self):
        if self.users_df is None:
            return
        self.users_df.columns = [c.strip() for c in self.users_df.columns]

    def _extract_user_field(self, user_row: pd.Series, field: str, default=None):
        if self.users_df is None:
            if isinstance(user_row, pd.Series) and field in user_row.index:
                val = user_row.get(field)
                return default if pd.isna(val) else val
            return default
        col = _find_column(self.users_df, field)
        if col and col in user_row.index:
            val = user_row.get(col)
            if pd.isna(val):
                return default
            return val
        if field in user_row.index:
            val = user_row.get(field)
            if pd.isna(val):
                return default
            return val
        return default

    def _compute_bmr_and_target(self, user_row: pd.Series) -> Tuple[int, Dict[str, float]]:
        age = _safe_float(self._extract_user_field(user_row, 'age', default=30))
        gender = str(self._extract_user_field(user_row, 'gender', default='male')).lower()
        height = _safe_float(self._extract_user_field(user_row, 'height', default=170))
        weight = _safe_float(self._extract_user_field(user_row, 'weight', default=70))
        activity = str(self._extract_user_field(user_row, 'activity', default='moderate')).lower()
        goal = str(self._extract_user_field(user_row, 'goal', default='maintain')).lower()
        if gender.startswith('f') or gender.startswith('w'):
            bmr = 10.0 * weight + 6.25 * height - 5.0 * age - 161.0
        else:
            bmr = 10.0 * weight + 6.25 * height - 5.0 * age + 5.0
        act_factor = 1.55
        if 'sed' in activity or 'low' in activity:
            act_factor = ACTIVITY_FACTORS['sedentary']
        elif 'light' in activity:
            act_factor = ACTIVITY_FACTORS['light']
        elif 'mod' in activity or 'moder' in activity:
            act_factor = ACTIVITY_FACTORS['moderate']
        elif 'act' in activity or 'very' in activity:
            act_factor = ACTIVITY_FACTORS['active']
        maint_cal = bmr * act_factor
        if 'lose' in goal:
            target_cal = int(max(1100, maint_cal - 500))
        elif 'gain' in goal:
            target_cal = int(maint_cal + 300)
        else:
            target_cal = int(maint_cal)
        macro_ratios = GOAL_MACRO_RATIOS.get(goal, GOAL_MACRO_RATIOS['maintain']).copy()
        return target_cal, macro_ratios

    def fetch_city_temperature(self, city_name: str, api_key: Optional[str] = None) -> Optional[float]:
        if not api_key:
            return None
        try:
            q = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": city_name, "appid": api_key, "units": "metric"},
                timeout=8
            )
            if q.status_code == 200:
                data = q.json()
                temp = data.get('main', {}).get('temp')
                return float(temp) if temp is not None else None
            return None
        except Exception:
            return None

    def recommend_for_user(self,
                           user_id: Optional[Any] = None,
                           user_row: Optional[pd.Series] = None,
                           weather_api_key: Optional[str] = None,
                           verbose: bool = False) -> Dict:
        if self.users_df is None and user_row is None:
            raise ValueError("No users dataset loaded and no user_row provided.")
        if user_row is None:
            if isinstance(user_id, int):
                if user_id < 0 or user_id >= len(self.users_df):
                    raise IndexError("user_id index out of range.")
                user_row = self.users_df.iloc[user_id]
            else:
                id_col_candidates = [c for c in self.users_df.columns if 'id' == c.lower() or 'user' in c.lower() and 'id' in c.lower()]
                if id_col_candidates:
                    matches = self.users_df[self.users_df[id_col_candidates[0]] == user_id]
                    if len(matches) == 0:
                        raise ValueError("user_id not found.")
                    user_row = matches.iloc[0]
                else:
                    raise ValueError("Cannot resolve user by id; please pass user_row or ensure users CSV has an id column.")
        calorie_target, macro_ratios = self._compute_bmr_and_target(user_row)
        preference = str(self._extract_user_field(user_row, 'preference', default='any') or 'any')
        allergies_raw = self._extract_user_field(user_row, 'allergies', default='')
        if isinstance(allergies_raw, str):
            allergies = [a.strip().lower() for a in re.split('[,;|]', allergies_raw) if a.strip()]
        elif isinstance(allergies_raw, (list, tuple, set)):
            allergies = [str(a).lower() for a in allergies_raw]
        else:
            allergies = []
        city = str(self._extract_user_field(user_row, 'city', default='')).strip()
        temp = None
        if city:
            temp = self.fetch_city_temperature(city, api_key=weather_api_key)
        if verbose:
            print(f"[User] calorie_target={calorie_target}, preference={preference}, allergies={allergies}, city={city}, temp={temp}")
        plan = self.recommend_meal_plan(
            calorie_target=calorie_target,
            goal=str(self._extract_user_field(user_row, 'goal', default='maintain')),
            preference=preference,
            allergies=allergies,
            temp=temp,
            meal_split=DEFAULT_MEAL_SPLIT
        )
        plan['user_info'] = {
            'calorie_target': calorie_target,
            'preference': preference,
            'allergies': allergies,
            'city': city,
            'temp': temp
        }
        return plan

    def recommend_for_all_users(self, weather_api_key: Optional[str] = None, verbose: bool = False) -> Dict:
        if self.users_df is None:
            raise ValueError("No users dataset loaded.")
        outputs = {}
        for idx in range(len(self.users_df)):
            try:
                plan = self.recommend_for_user(user_id=idx, weather_api_key=weather_api_key, verbose=verbose)
                outputs[idx] = plan
            except Exception as e:
                outputs[idx] = {'error': str(e)}
        return outputs
    # -------------------------
    # Core recommendation (uses cluster mapping for meal pools)
    # -------------------------
    def recommend_meal_plan(self,
                            calorie_target: int,
                            goal: str = 'maintain',
                            preference: str = 'any',
                            allergies: Optional[List[str]] = None,
                            temp: Optional[float] = None,
                            meal_split: Dict[str, float] = None,
                            max_items_per_meal: int = 3,
                            verbose: bool = False) -> Dict[str, Any]:
        if meal_split is None:
            meal_split = DEFAULT_MEAL_SPLIT
        allergies = [a.lower() for a in (allergies or [])]
        macro_ratios = self._adjust_for_temp_and_goal(goal, temp)
        per_meal_calories = {m: max(50, int(calorie_target * frac)) for m, frac in meal_split.items()}
        self.df['health_score'] = self.df.apply(self.compute_health_score, axis=1)
        mask_pref = np.ones(len(self.df), dtype=bool)
        if preference.lower() == 'veg':
            mask_pref &= self.df['is_strictly_vegetarian'].values
        elif preference.lower() == 'pescatarian':
            mask_pref &= ~self.df['is_meat_item'].values
        elif preference.lower() == 'non-veg':
            pass
        if allergies:
            descs = self.df['Description'].str.lower()
            for a in allergies:
                mask_pref &= ~descs.str.contains(re.escape(a))
        candidate_idx = np.where(mask_pref)[0]
        if len(candidate_idx) == 0:
            raise ValueError("No candidate foods after applying preference/allergy filters.")
        plan = {}
        used_indices = set()
        cal_col = self._choose_col('calories')
        prot_col = self._choose_col('protein')
        fat_col = self._choose_col('fat')
        carb_col = self._choose_col('carbs') or self._choose_col('carbo')

        def nutrition_for(row_idx, grams):
            row = self.df.iloc[row_idx]
            kcal_per_100g = float(row.get(cal_col, 0.0) or 0.0)
            factor = grams / 100.0
            return {
                'calories': kcal_per_100g * factor,
                'protein': float(row.get(prot_col, 0.0) or 0.0) * factor,
                'fat': float(row.get(fat_col, 0.0) or 0.0) * factor,
                'carbs': float(row.get(carb_col, 0.0) or 0.0) * factor
            }

        for meal, target_cal in per_meal_calories.items():
            meal_plan = []
            pool_idx = candidate_idx.copy()
            if hasattr(self, 'meal_to_clusters') and meal in self.meal_to_clusters and len(self.meal_to_clusters[meal]) > 0:
                cluster_ids = self.meal_to_clusters[meal]
                cluster_mask = np.isin(self.df.get('food_cluster', np.full(len(self.df), -1)).values, cluster_ids)
                pool_idx = np.intersect1d(pool_idx, np.where(cluster_mask)[0])
                if len(pool_idx) < max_items_per_meal:
                    pool_idx = candidate_idx.copy()

            meal_cat_tokens = MEAL_CATEGORIES.get(meal, [])
            if len(pool_idx) > max_items_per_meal and meal_cat_tokens:
                descs = self.df['Description'].str.upper()
                catmask = np.zeros(len(self.df), dtype=bool)
                for tok in meal_cat_tokens:
                    catmask |= descs.str.contains(tok)
                pool_idx2 = np.intersect1d(pool_idx, np.where(catmask)[0])
                if len(pool_idx2) >= max_items_per_meal:
                    pool_idx = pool_idx2

            p_ratio = macro_ratios['protein']
            c_ratio = macro_ratios['carbs']
            f_ratio = macro_ratios['fat']
            target_protein_g = (target_cal * p_ratio) / 4.0
            target_carbs_g = (target_cal * c_ratio) / 4.0
            target_fat_g = (target_cal * f_ratio) / 9.0
            feature_vec = np.zeros(len(self.feature_columns))
            for i, col in enumerate(self.feature_columns):
                lc = col.lower()
                if prot_col and prot_col.lower() in lc:
                    feature_vec[i] = target_protein_g
                elif carb_col and carb_col.lower() in lc:
                    feature_vec[i] = target_carbs_g
                elif fat_col and fat_col.lower() in lc:
                    feature_vec[i] = target_fat_g
                elif cal_col and cal_col.lower() in lc:
                    feature_vec[i] = target_cal
                else:
                    feature_vec[i] = 0.0
            try:
                feature_vec_scaled = self.scaler.transform(feature_vec.reshape(1, -1))
                feature_vec_norm = normalize(feature_vec_scaled)[0]
            except Exception:
                feature_vec_norm = feature_vec / (np.linalg.norm(feature_vec) + 1e-8)
            sims = cosine_similarity(self.food_matrix, feature_vec_norm.reshape(1, -1)).flatten()
            candidate_sims = [(idx, sims[idx], self.df.at[idx, 'health_score']) for idx in pool_idx if idx not in used_indices]
            candidate_sims.sort(key=lambda x: (x[1], x[2]), reverse=True)
            remaining_cal = target_cal
            items_selected = 0
            i_ptr = 0
            while remaining_cal > 0 and items_selected < max_items_per_meal and i_ptr < len(candidate_sims):
                idx, sim_val, health = candidate_sims[i_ptr]; i_ptr += 1
                if idx in used_indices:
                    continue
                row = self.df.iloc[idx]
                kcal_100g = float(row.get(cal_col, 0.0) or 0.0)
                if kcal_100g <= 0:
                    continue
                propose_cal = remaining_cal * 0.6
                grams = max(10, (propose_cal / (kcal_100g + 1e-9)) * 100.0)
                grams = float(max(10.0, min(grams, 500.0)))
                nutr = nutrition_for(idx, grams)
                if nutr['calories'] > remaining_cal * 1.1:
                    scale_factor = remaining_cal / (nutr['calories'] + 1e-9)
                    grams *= scale_factor
                    nutr = nutrition_for(idx, grams)
                meal_plan.append({
                    'Description': row['Description'],
                    'Category': row.get('Category', ''),
                    'grams': round(grams, 1),
                    'calories': round(nutr['calories'], 1),
                    'protein_g': round(nutr['protein'], 1),
                    'fat_g': round(nutr['fat'], 1),
                    'carbs_g': round(nutr['carbs'], 1),
                    'health_score': round(float(row.get('health_score', 0.0)), 2),
                    'similarity': round(float(sim_val), 4)
                })
                used_indices.add(idx)
                items_selected += 1
                remaining_cal -= nutr['calories']
                if remaining_cal <= max(25, 0.05 * target_cal):
                    break
            if len(meal_plan) == 0:
                fallback_idx = [i for i in np.argsort(-sims) if i not in used_indices][:max_items_per_meal]
                for idx in fallback_idx:
                    row = self.df.iloc[idx]
                    kcal_100g = float(row.get(cal_col, 0.0) or 100.0)
                    grams = max(10.0, (target_cal / 2.0 / (kcal_100g + 1e-9)) * 100.0)
                    nutr = nutrition_for(idx, grams)
                    meal_plan.append({
                        'Description': row['Description'],
                        'Category': row.get('Category', ''),
                        'grams': round(grams, 1),
                        'calories': round(nutr['calories'], 1),
                        'protein_g': round(nutr['protein'], 1),
                        'fat_g': round(nutr['fat'], 1),
                        'carbs_g': round(nutr['carbs'], 1),
                        'health_score': round(float(row.get('health_score', 0.0)), 2),
                        'similarity': round(float(sims[idx]), 4)
                    })
                    used_indices.add(idx)
            plan[meal] = meal_plan
        summary = {'target_calories': calorie_target, 'by_meal_targets': per_meal_calories}
        totals = {'calories': 0.0, 'protein': 0.0, 'fat': 0.0, 'carbs': 0.0}
        for meal_items in plan.values():
            for item in meal_items:
                totals['calories'] += item['calories']
                totals['protein'] += item['protein_g']
                totals['fat'] += item['fat_g']
                totals['carbs'] += item['carbs_g']
        summary['totals'] = {k: round(v, 1) for k, v in totals.items()}
        return {'plan': plan, 'summary': summary}
    
    def _adjust_for_temp_and_goal(self, goal: str, temperature: Optional[float]):
        ratios = GOAL_MACRO_RATIOS.get(goal, GOAL_MACRO_RATIOS['maintain']).copy()
        if temperature is not None:
            if temperature >= 28:
                ratios['carbs'] = max(0.30, ratios.get('carbs', 0.0) - 0.05)
                ratios['protein'] = min(0.40, ratios.get('protein', 0.0) + 0.02)
            elif temperature <= 10:
                ratios['carbs'] = min(0.55, ratios.get('carbs', 0.0) + 0.05)
                ratios['fat'] = min(0.35, ratios.get('fat', 0.0) + 0.02)
        s = sum(ratios.values())
        if s <= 0:
            return GOAL_MACRO_RATIOS['maintain'].copy()
        return {k: v / s for k, v in ratios.items()}

    # -------------------------
    # Persistence
    # -------------------------
    def save_artifacts(self, path_prefix: str):
        os.makedirs(os.path.dirname(path_prefix) or '.', exist_ok=True)
        with open(f"{path_prefix}_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(f"{path_prefix}_food_matrix.pkl", "wb") as f:
            pickle.dump(self.food_matrix, f)
        with open(f"{path_prefix}_df.pkl", "wb") as f:
            pickle.dump(self.df, f)
        if self.kmeans is not None:
            with open(f"{path_prefix}_kmeans.pkl", "wb") as f:
                pickle.dump(self.kmeans, f)

    def load_artifacts(self, path_prefix: str):
        with open(f"{path_prefix}_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(f"{path_prefix}_food_matrix.pkl", "rb") as f:
            self.food_matrix = pickle.load(f)
        with open(f"{path_prefix}_df.pkl", "rb") as f:
            self.df = pickle.load(f)
        kmeans_path = f"{path_prefix}_kmeans.pkl"
        if os.path.exists(kmeans_path):
            with open(kmeans_path, "rb") as f:
                self.kmeans = pickle.load(f)
