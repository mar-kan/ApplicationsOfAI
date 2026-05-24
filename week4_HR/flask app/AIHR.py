from flask import Flask, render_template, request
import tensorflow as tf
import joblib
import pandas as pd
import re

model = joblib.load('checkpoints/hr_svm_model.pkl')
scaler = joblib.load(open('checkpoints/hr_scaler.pkl', 'rb'))
feature_columns = joblib.load(open('checkpoints/hr_features.pkl', 'rb'))

app = Flask(__name__)

# Build UI groups based on the original dataset so the form matches the original columns.
# If the original CSV is not available, fall back to deriving groups from the processed feature columns.
ui_groups = {}
try:
    df_raw = pd.read_csv('../HR-Employee-Attrition.csv')
    exclude_cols = {'Attrition'}
    form_columns = [c for c in df_raw.columns if c not in exclude_cols]

    for col in form_columns:
        display_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', col).title()
        if df_raw[col].dtype == 'object' or df_raw[col].dtype == 'string':
            options = sorted(df_raw[col].dropna().unique().astype(str).tolist())

            # add no for under18
            if col == 'Under18':
                options.append('N')

            ui_groups[col] = {'display_name': display_name, 'is_categorical': True, 'options': options}
        else:
            ui_groups[col] = {'display_name': display_name, 'is_categorical': False, 'options': []}
except Exception:
    # fallback: infer from feature_columns (one-hot encoded names like BusinessTravel_Travel_Rarely)
    for col in feature_columns:
        parts = col.split('_', 1)
        parent_name = parts[0]
        display_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', parent_name).title()
        if parent_name not in ui_groups:
            ui_groups[parent_name] = {'display_name': display_name, 'is_categorical': False, 'options': []}
        if len(parts) > 1:
            ui_groups[parent_name]['is_categorical'] = True
            ui_groups[parent_name]['options'].append(parts[1])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None

    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            input_dict = {}

            # Read the form data and map it back to feature_columns
            # For each trained feature column (one-hot columns for categoricals, numeric otherwise),
            # construct the dictionary expected by the scaler/model. For categoricals, note that
            # the training used drop_first=True, so the baseline category corresponds to all-zeros.
            for col in feature_columns:
                parts = col.split('_', 1)
                parent_name = parts[0]

                if len(parts) > 1:
                    # categorical one-hot column
                    category_option = parts[1]
                    user_selection = form_data.get(parent_name)
                    input_dict[col] = 1.0 if user_selection == category_option else 0.0
                else:
                    # numeric feature
                    user_input = form_data.get(parent_name, '')
                    input_dict[col] = float(user_input) if user_input not in (None, '') else 0.0

            input_df = pd.DataFrame([input_dict])
            input_scaled = scaler.transform(input_df)

            prob = model.predict_proba(input_scaled)[0, 1]
            prediction = 'High risk' if prob >= 0.5 else 'No risk'
            probability = round(prob * 100, 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"
            

    return render_template(
        'index.html',
        ui_groups=ui_groups,
        prediction=prediction,
        probability=probability
    )

if __name__ == '__main__':
    app.run(debug=True)