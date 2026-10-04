import os
import json
import joblib
import numpy as np
import pandas as pd

class TitanicPredictor:
    def __init__(self, models_dir="models", data_dir="data"):
        self.models_dir = models_dir
        self.data_dir = data_dir
        
        # Load encoders & scaler
        self.le_sex = joblib.load(os.path.join(models_dir, 'le_sex.joblib'))
        self.le_embarked = joblib.load(os.path.join(models_dir, 'le_embarked.joblib'))
        self.scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        
        # Load metrics & meta
        with open(os.path.join(models_dir, 'model_metrics.json'), 'r') as f:
            self.model_metrics = json.load(f)
            
        with open(os.path.join(models_dir, 'feature_meta.json'), 'r') as f:
            self.feature_meta = json.load(f)
            
        # Load raw & processed data
        self.df_raw = pd.read_csv(os.path.join(data_dir, 'titanic_raw.csv'))
        self.df_proc = pd.read_csv(os.path.join(data_dir, 'titanic_processed.csv'))
        
        # Load all trained models
        self.models = {}
        for key in self.model_metrics.keys():
            model_path = os.path.join(models_dir, f"{key}.joblib")
            if os.path.exists(model_path):
                self.models[key] = joblib.load(model_path)

    def get_models_list(self):
        return self.model_metrics

    def _prepare_input_vector(self, pclass, sex, age, sibsp, parch, fare, embarked):
        # Normalize sex
        if isinstance(sex, str):
            sex_val = 0 if sex.lower().startswith('f') else 1
        else:
            sex_val = int(sex)
            
        # Normalize embarked
        if isinstance(embarked, str):
            emb = embarked.upper()[0]
            emb_map = {'C': 0, 'Q': 1, 'S': 2}
            emb_val = emb_map.get(emb, 2)
        else:
            emb_val = int(embarked)
            
        age_val = float(age) if age is not None else self.feature_meta['age_mean']
        fare_val = float(fare) if fare is not None else self.feature_meta['fare_mean']
        sibsp_val = int(sibsp)
        parch_val = int(parch)
        alone_val = 1 if (sibsp_val == 0 and parch_val == 0) else 0
        
        # 8 features matching notebook: pclass, sex, age, sibsp, parch, fare, embarked, alone
        features = np.array([[int(pclass), sex_val, int(age_val), sibsp_val, parch_val, int(fare_val), emb_val, alone_val]])
        
        return features, {
            'pclass': int(pclass),
            'sex': 'female' if sex_val == 0 else 'male',
            'age': age_val,
            'sibsp': sibsp_val,
            'parch': parch_val,
            'fare': fare_val,
            'embarked': ['Cherbourg', 'Queenstown', 'Southampton'][emb_val],
            'alone': alone_val
        }

    def predict_single(self, model_key='svc', pclass=3, sex='male', age=29.0, sibsp=0, parch=0, fare=14.0, embarked='S'):
        if model_key not in self.models:
            model_key = 'svc'
            
        model = self.models[model_key]
        is_scaled = self.model_metrics[model_key]['scaled']
        
        feat_vector, clean_dict = self._prepare_input_vector(pclass, sex, age, sibsp, parch, fare, embarked)
        
        if is_scaled:
            feat_input = self.scaler.transform(feat_vector)
        else:
            feat_input = feat_vector
            
        pred_class = int(model.predict(feat_input)[0])
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(feat_input)[0]
            prob_death = float(probs[0])
            prob_survive = float(probs[1])
        elif hasattr(model, 'decision_function'):
            score = float(model.decision_function(feat_input)[0])
            prob_survive = float(1 / (1 + np.exp(-score)))
            prob_death = float(1 - prob_survive)
        else:
            prob_survive = 1.0 if pred_class == 1 else 0.0
            prob_death = 1.0 - prob_survive
            
        # Analyze contributing factors
        factors = []
        if clean_dict['sex'] == 'female':
            factors.append({'factor': 'Gender: Female', 'impact': 'positive', 'strength': 'Critical', 'text': '"Women and children first" maritime priority gave female passengers a 74%+ historical survival rate.'})
        else:
            factors.append({'factor': 'Gender: Male', 'impact': 'negative', 'strength': 'Critical', 'text': 'Male passengers were held back from lifeboats, resulting in over 81% mortality.'})
            
        if clean_dict['pclass'] == 1:
            factors.append({'factor': 'Class: 1st Class Luxury', 'impact': 'positive', 'strength': 'High', 'text': 'Direct boat deck access and priority evacuation led to 63% survival.'})
        elif clean_dict['pclass'] == 3:
            factors.append({'factor': 'Class: 3rd Class Steerage', 'impact': 'negative', 'strength': 'High', 'text': 'Steerage passengers were located in bottom decks with locked gates, leading to 75% mortality.'})
        else:
            factors.append({'factor': 'Class: 2nd Class', 'impact': 'neutral', 'strength': 'Medium', 'text': 'Mid-deck position: high survival for women & children, low for men.'})
            
        if clean_dict['age'] < 12:
            factors.append({'factor': 'Age: Child (<12)', 'impact': 'positive', 'strength': 'High', 'text': 'Children received lifeboat boarding priority during distress procedures.'})
        elif clean_dict['age'] > 55:
            factors.append({'factor': 'Age: Senior (55+)', 'impact': 'negative', 'strength': 'Low', 'text': 'Cold North Atlantic temperature exposure and mobility barriers increased mortality.'})
            
        family_size = clean_dict['sibsp'] + clean_dict['parch']
        if family_size == 0:
            factors.append({'factor': 'Solo Traveler', 'impact': 'neutral', 'strength': 'Low', 'text': 'Solo travel enabled faster individual movement but lacked party support.'})
        elif 1 <= family_size <= 3:
            factors.append({'factor': f'Small Family ({family_size+1} members)', 'impact': 'positive', 'strength': 'Medium', 'text': 'Optimal unit size: family members assisted each other in reaching lifeboats.'})
        else:
            factors.append({'factor': f'Large Family ({family_size+1} members)', 'impact': 'negative', 'strength': 'Medium', 'text': 'Large groups were difficult to keep together in crowded stairwells.'})

        if clean_dict['fare'] > 50:
            factors.append({'factor': f'Luxury Fare (£{int(clean_dict["fare"])})', 'impact': 'positive', 'strength': 'Medium', 'text': 'Luxury ticket pricing correlated with upper deck suites and attendant assistance.'})

        # Realistic ticket metadata
        berth_map = {
            1: ['State Room A-36 (Promenade Deck)', 'Suite B-52 (Bridge Deck)', 'Deluxe Cabin C-68', 'Parlour Suite B-58'],
            2: ['Cabin D-24 (Saloon Deck)', 'Cabin E-18 (Upper Deck)', 'Cabin D-42 (Midship)'],
            3: ['Berth 118 (Lower Hull Section G)', 'Berth 84 (Steerage Forward)', 'Berth 42 (Section E)']
        }
        import random
        random.seed(int(clean_dict['age'] * 10 + clean_dict['fare'] + clean_dict['pclass']))
        assigned_berth = random.choice(berth_map.get(clean_dict['pclass'], berth_map[3]))
        
        # Format vintage British shillings fare
        fare_val = clean_dict['fare']
        pounds = int(fare_val)
        shillings = int((fare_val - pounds) * 20)
        pence = int((((fare_val - pounds) * 20) - shillings) * 12)
        vintage_fare_str = f"£{pounds} {shillings}s {pence}d"

        # Generate realistic contract number
        import hashlib
        ticket_seed = f"{pclass}-{clean_dict['sex']}-{age}-{fare}-{clean_dict['embarked']}"
        ticket_hash = hashlib.md5(ticket_seed.encode()).hexdigest()[:6].upper()
        contract_number = f"No. {int(ticket_hash, 16) % 900000 + 100000}"
        ticket_number = f"WSL-1912-{ticket_hash}"

        return {
            'model_used': {
                'key': model_key,
                'name': self.model_metrics[model_key]['name'],
                'accuracy': self.model_metrics[model_key]['accuracy']
            },
            'prediction': pred_class,
            'outcome': 'SURVIVED' if pred_class == 1 else 'PERISHED',
            'survival_probability': round(prob_survive * 100, 1),
            'death_probability': round(prob_death * 100, 1),
            'confidence': round(max(prob_survive, prob_death) * 100, 1),
            'passenger_summary': clean_dict,
            'ticket_number': ticket_number,
            'contract_number': contract_number,
            'assigned_berth': assigned_berth,
            'vintage_fare': vintage_fare_str,
            'risk_factors': factors
        }

    def predict_manifest_death_count(self, model_key='svc', pclass_filter=None, sex_filter=None, embarked_filter=None):
        if model_key not in self.models:
            model_key = 'svc'
            
        model = self.models[model_key]
        is_scaled = self.model_metrics[model_key]['scaled']
        
        df = self.df_proc.copy()
        
        if pclass_filter is not None and str(pclass_filter) != 'all':
            pclass_val = int(pclass_filter)
            df = df[df['pclass'] == pclass_val]
            
        if sex_filter is not None and str(sex_filter) != 'all':
            sex_val = 0 if str(sex_filter).lower().startswith('f') else 1
            df = df[df['sex'] == sex_val]
            
        if embarked_filter is not None and str(embarked_filter) != 'all':
            emb_map = {'C': 0, 'Q': 1, 'S': 2}
            emb_val = emb_map.get(str(embarked_filter).upper(), 2)
            df = df[df['embarked'] == emb_val]
            
        if len(df) == 0:
            return {'error': 'No passengers match the selected filter criteria.'}
            
        X = df.drop('survived', axis=1)
        y = df['survived']
        
        X_input = self.scaler.transform(X) if is_scaled else X
        preds = model.predict(X_input)
        
        total = len(df)
        pred_deaths = int((preds == 0).sum())
        pred_survivors = int((preds == 1).sum())
        actual_deaths = int((y == 0).sum())
        actual_survivors = int((y == 1).sum())
        
        death_error = abs(pred_deaths - actual_deaths)
        death_pct = round((pred_deaths / total) * 100, 1) if total > 0 else 0
        actual_death_pct = round((actual_deaths / total) * 100, 1) if total > 0 else 0
        
        # Breakdown by class
        class_breakdown = {}
        for c in [1, 2, 3]:
            c_mask = (df['pclass'] == c).values
            c_total = int(c_mask.sum())
            if c_total > 0:
                c_pred_d = int((preds[c_mask] == 0).sum())
                c_act_d = int((y.values[c_mask] == 0).sum())
                class_breakdown[f'Class {c}'] = {
                    'total': c_total,
                    'predicted_deaths': c_pred_d,
                    'actual_deaths': c_act_d,
                    'predicted_death_rate': round((c_pred_d / c_total) * 100, 1),
                    'actual_death_rate': round((c_act_d / c_total) * 100, 1)
                }
                
        # Breakdown by gender
        gender_breakdown = {}
        for s_name, s_val in [('Female', 0), ('Male', 1)]:
            s_mask = (df['sex'] == s_val).values
            s_total = int(s_mask.sum())
            if s_total > 0:
                s_pred_d = int((preds[s_mask] == 0).sum())
                s_act_d = int((y.values[s_mask] == 0).sum())
                gender_breakdown[s_name] = {
                    'total': s_total,
                    'predicted_deaths': s_pred_d,
                    'actual_deaths': s_act_d,
                    'predicted_death_rate': round((s_pred_d / s_total) * 100, 1),
                    'actual_death_rate': round((s_act_d / s_total) * 100, 1)
                }
                
        # Breakdown by age
        age_breakdown = {}
        age_categories = [
            ('Children (0-12)', df['age'] <= 12),
            ('Teens & Youths (13-21)', (df['age'] > 12) & (df['age'] <= 21)),
            ('Adults (22-50)', (df['age'] > 21) & (df['age'] <= 50)),
            ('Seniors (51+)', df['age'] > 50)
        ]
        for cat_name, a_mask_series in age_categories:
            a_mask = a_mask_series.values
            a_total = int(a_mask.sum())
            if a_total > 0:
                a_pred_d = int((preds[a_mask] == 0).sum())
                a_act_d = int((y.values[a_mask] == 0).sum())
                age_breakdown[cat_name] = {
                    'total': a_total,
                    'predicted_deaths': a_pred_d,
                    'actual_deaths': a_act_d,
                    'predicted_death_rate': round((a_pred_d / a_total) * 100, 1),
                    'actual_death_rate': round((a_act_d / a_total) * 100, 1)
                }

        return {
            'model_key': model_key,
            'model_name': self.model_metrics[model_key]['name'],
            'total_passengers': total,
            'predicted_deaths': pred_deaths,
            'actual_deaths': actual_deaths,
            'predicted_survivors': pred_survivors,
            'actual_survivors': actual_survivors,
            'predicted_mortality_rate': death_pct,
            'actual_mortality_rate': actual_death_pct,
            'death_count_error': death_error,
            'death_count_accuracy': round((1.0 - (death_error / max(1, actual_deaths))) * 100, 2),
            'class_breakdown': class_breakdown,
            'gender_breakdown': gender_breakdown,
            'age_breakdown': age_breakdown
        }

    def simulate_custom_group_deaths(self, model_key='svc', cohorts=None):
        if not cohorts:
            return {'error': 'No cohorts provided for death count simulation.'}
            
        if model_key not in self.models:
            model_key = 'svc'
            
        results = []
        total_people = 0
        total_predicted_deaths = 0
        total_predicted_survivors = 0
        
        for idx, cohort in enumerate(cohorts):
            count = int(cohort.get('count', 1))
            if count <= 0:
                continue
                
            pclass = int(cohort.get('pclass', 3))
            sex = cohort.get('sex', 'male')
            age = float(cohort.get('age', 28.0))
            sibsp = int(cohort.get('sibsp', 0))
            parch = int(cohort.get('parch', 0))
            fare = float(cohort.get('fare', 15.0))
            embarked = cohort.get('embarked', 'S')
            
            pred = self.predict_single(model_key, pclass, sex, age, sibsp, parch, fare, embarked)
            
            prob_death = pred['death_probability'] / 100.0
            expected_deaths = int(round(count * prob_death))
            expected_survivors = count - expected_deaths
            
            total_people += count
            total_predicted_deaths += expected_deaths
            total_predicted_survivors += expected_survivors
            
            results.append({
                'cohort_index': idx + 1,
                'name': cohort.get('label', f"Group {idx+1}: Class {pclass} {sex.title()}, Age {int(age)}"),
                'count': count,
                'pclass': pclass,
                'sex': sex,
                'age': age,
                'fare': fare,
                'death_probability_pct': pred['death_probability'],
                'survival_probability_pct': pred['survival_probability'],
                'expected_deaths': expected_deaths,
                'expected_survivors': expected_survivors,
                'mortality_rate_pct': round((expected_deaths / count) * 100, 1)
            })
            
        overall_mortality = round((total_predicted_deaths / max(1, total_people)) * 100, 1)
        
        return {
            'model_name': self.model_metrics[model_key]['name'],
            'total_passengers_simulated': total_people,
            'total_predicted_deaths': total_predicted_deaths,
            'total_predicted_survivors': total_predicted_survivors,
            'overall_mortality_rate': overall_mortality,
            'cohort_results': results
        }

    def get_historical_manifest_records(self, limit=100, offset=0, search=""):
        df = self.df_raw.copy().reset_index()
        if search:
            s = search.lower()
            mask = (
                df['who'].fillna('').astype(str).str.lower().str.contains(s) |
                df['sex'].fillna('').astype(str).str.lower().str.contains(s) |
                df['embark_town'].fillna('').astype(str).str.lower().str.contains(s)
            )
            df = df[mask]
            
        total = len(df)
        sliced = df.iloc[offset:offset+limit]
        
        records = []
        for _, row in sliced.iterrows():
            records.append({
                'passenger_id': int(row['index']) + 1,
                'survived': int(row['survived']),
                'pclass': int(row['pclass']),
                'sex': str(row['sex']),
                'age': float(row['age']) if pd.notnull(row['age']) else None,
                'sibsp': int(row['sibsp']),
                'parch': int(row['parch']),
                'fare': round(float(row['fare']), 2) if pd.notnull(row['fare']) else 0.0,
                'embarked': str(row['embarked']) if pd.notnull(row['embarked']) else 'S',
                'embark_town': str(row['embark_town']) if pd.notnull(row['embark_town']) else 'Southampton',
                'who': str(row['who']) if pd.notnull(row['who']) else 'adult'
            })
            
        return {
            'total': total,
            'offset': offset,
            'limit': limit,
            'records': records
        }
