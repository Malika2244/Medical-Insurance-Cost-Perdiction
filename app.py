# Medical Insurance Cost Prediction App
# Big Brains Internship 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import gradio as gr

# Load and Train Model (auto)
df = pd.read_csv("insurance.csv")
# Simple encoding same as your notebook
df['sex'] = df['sex'].map({'female':0, 'male':1})
df['smoker'] = df['smoker'].map({'no':0, 'yes':1})
df['region'] = df['region'].map({'northeast':0, 'northwest':1, 'southeast':2, 'southwest':3})

X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

def predict_insurance(age, sex, bmi, children, smoker, region):
    sex_val = 1 if sex == "male" else 0
    smoker_val = 1 if smoker == "yes" else 0
    region_map = {"northeast":0, "northwest":1, "southeast":2, "southwest":3}
    region_val = region_map[region]
    
    input_df = pd.DataFrame([[age, sex_val, bmi, children, smoker_val, region_val]],
                            columns=['age','sex','bmi','children','smoker','region'])
    pred = model.predict(input_df)[0]
    return f"Estimated Insurance Cost: ${pred:.2f}"

# Gradio UI
app = gr.Interface(
    fn=predict_insurance,
    inputs=[
        gr.Slider(18, 65, value=30, label="Age"),
        gr.Radio(["male", "female"], label="Gender", value="female"),
        gr.Slider(15, 50, value=25, label="BMI"),
        gr.Slider(0, 5, step=1, value=0, label="Children"),
        gr.Radio(["yes", "no"], label="Smoker?", value="no"),
        gr.Radio(["northeast", "northwest", "southeast", "southwest"], label="Region", value="northeast")
    ],
    outputs="text",
    title="Medical Insurance Cost Predictor",
    description="Big Brains Internship Project - Predicts insurance charges. R2 Score: 0.8068"
)

if __name__ == "__main__":
    app.launch()
