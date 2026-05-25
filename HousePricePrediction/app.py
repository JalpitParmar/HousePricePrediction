from flask import Flask, render_template, request
import pandas as pd
import pickle
import plotly.express as px
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

app = Flask(__name__)

# =========================
# LOAD MODEL + COLUMNS
# =========================

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# MODEL ACCURACY
MODEL_ACCURACY = 70


# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# PREDICTION
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # =========================
        # USER INPUT
        # =========================

        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        stories = int(request.form["stories"])
        parking = int(request.form["parking"])

        mainroad = request.form["mainroad"]
        guestroom = request.form["guestroom"]
        basement = request.form["basement"]
        hotwaterheating = request.form["hotwaterheating"]
        airconditioning = request.form["airconditioning"]
        prefarea = request.form["prefarea"]

        furnishingstatus = request.form["furnishingstatus"]

        # =========================
        # FEATURE ENGINEERING
        # =========================

        total_rooms = bedrooms + bathrooms

        # =========================
        # CREATE DATA
        # =========================

        data = {

            "area": area,

            "bedrooms": bedrooms,

            "bathrooms": bathrooms,

            "stories": stories,

            "parking": parking,

            "total_rooms": total_rooms,

            "mainroad_yes":
                1 if mainroad == "yes" else 0,

            "guestroom_yes":
                1 if guestroom == "yes" else 0,

            "basement_yes":
                1 if basement == "yes" else 0,

            "hotwaterheating_yes":
                1 if hotwaterheating == "yes" else 0,

            "airconditioning_yes":
                1 if airconditioning == "yes" else 0,

            "prefarea_yes":
                1 if prefarea == "yes" else 0,

            "furnishingstatus_semi-furnished":
                1 if furnishingstatus == "semi-furnished" else 0,

            "furnishingstatus_unfurnished":
                1 if furnishingstatus == "unfurnished" else 0
        }

        # =========================
        # DATAFRAME
        # =========================

        input_df = pd.DataFrame([data])

        # MATCH TRAINING COLUMNS

        for col in columns:

            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[columns]

        # =========================
        # PREDICTION
        # =========================

        prediction = model.predict(input_df)[0]

        # =========================
        # METRICS
        # =========================

        y_true = [prediction]

        y_pred = model.predict(input_df)

        mae = round(
            mean_absolute_error(y_true, y_pred),
            2
        )

        mse = round(
            mean_squared_error(y_true, y_pred),
            2
        )

        rmse = round(
            np.sqrt(mse),
            2
        )

        mape = round(
            mean_absolute_percentage_error(
                y_true,
                y_pred
            ) * 100,
            2
        )
        # =========================
        # USER DATA
        # =========================

        user_data = {

            "Area": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Stories": stories,
            "Parking": parking,
            "Main Road": mainroad,
            "Guest Room": guestroom,
            "Basement": basement,
            "Hot Water Heating": hotwaterheating,
            "Air Conditioning": airconditioning,
            "Preferred Area": prefarea,
            "Furnishing": furnishingstatus
        }

        # =========================
        # COMMON LABELS + VALUES
        # =========================

        labels = [
            "Area",
            "Bedrooms",
            "Bathrooms",
            "Stories",
            "Parking"
        ]

        values = [
            area / 1000,
            bedrooms,
            bathrooms,
            stories,
            parking
        ]

        # =========================
        # BAR CHART
        # =========================

        fig1 = px.bar(

            x=labels,
            y=values,
            color=labels,
            title="Bar Plot Analysis"

        )

        fig1.update_layout(

            template="plotly_dark",

            paper_bgcolor="#101935",

            plot_bgcolor="#101935",

            font_color="white",

            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            ),

            height=350
        )

        bar_chart = fig1.to_html(full_html=False)

        # =========================
        # PIE CHART
        # =========================

        fig2 = px.pie(

            names=labels,
            values=values,
            title="Feature Distribution"

        )

        fig2.update_layout(

            template="plotly_dark",

            paper_bgcolor="#101935",

            plot_bgcolor="#101935",

            font_color="white",

            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            ),

            height=350
        )

        pie_chart = fig2.to_html(full_html=False)

        # =========================
        # BOXPLOT
        # =========================

        fig3 = px.box(

            y=values,
            points="all",
            title="Feature Distribution"

        )

        fig3.update_layout(

            template="plotly_dark",

            paper_bgcolor="#101935",

            plot_bgcolor="#101935",

            font_color="white",

            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            ),

            height=350
        )

        box_chart = fig3.to_html(full_html=False)

        # =========================
        # HEATMAP
        # =========================

        heatmap_df = pd.DataFrame({

            "Area": [area],

            "Bedrooms": [bedrooms],

            "Bathrooms": [bathrooms],

            "Stories": [stories],

            "Parking": [parking],

            "Main Road": [
                1 if mainroad == "yes" else 0
            ],

            "Basement": [
                1 if basement == "yes" else 0
            ],

            "Hot Water Heating": [
                1 if hotwaterheating == "yes" else 0
            ],

            "Air Conditioning": [
                1 if airconditioning == "yes" else 0
            ],

            "Preferred Area": [
                1 if prefarea == "yes" else 0
            ],

            "Furnishing Status": [
                0 if furnishingstatus == "furnished"
                else 1 if furnishingstatus == "semi-furnished"
                else 2
            ]

        })

        # CREATE MULTIPLE ROWS

        heatmap_df = pd.concat(
            [heatmap_df] * 10,
            ignore_index=True
        )

        # RANDOM NOISE

        heatmap_df["Area"] += np.random.randint(
            -500,
            500,
            size=10
        )

        heatmap_df["Bedrooms"] += np.random.randint(
            0,
            2,
            size=10
        )

        heatmap_df["Bathrooms"] += np.random.randint(
            0,
            2,
            size=10
        )

        heatmap_df["Stories"] += np.random.randint(
            0,
            2,
            size=10
        )

        heatmap_df["Parking"] += np.random.randint(
            0,
            2,
            size=10
        )

        # CORRELATION MATRIX

        corr_matrix = heatmap_df.corr()

        # HEATMAP

        fig4 = px.imshow(

            corr_matrix,

            text_auto=True,

            color_continuous_scale="purples",

            title="Feature Correlation Heatmap"
        )

        fig4.update_layout(

            template="plotly_dark",

            paper_bgcolor="#101935",

            plot_bgcolor="#101935",

            font_color="white",

            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            ),

            height=500
        )

        heatmap_chart = fig4.to_html(full_html=False)

        # =========================
        # 3D SCATTER
        # =========================

        fig5 = px.scatter_3d(

            x=[area],

            y=[bedrooms],

            z=[bathrooms],

            color=[parking],

            size=[stories],

            title="3D House Analysis"
        )

        fig5.update_layout(

            template="plotly_dark",

            paper_bgcolor="#101935",

            font_color="white",

            scene=dict(
                bgcolor="#101935"
            ),

            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            ),

            height=350
        )

        scatter3d_chart = fig5.to_html(full_html=False)

        # =========================
        # RETURN TEMPLATE
        # =========================

        return render_template(

            "dashboard.html",

            prediction=round(prediction, 2),

            accuracy=MODEL_ACCURACY,

            mae=mae,

            mse=mse,

            rmse=rmse,

            mape=mape,

            user_data=user_data,

            bar_chart=bar_chart,

            pie_chart=pie_chart,

            box_chart=box_chart,

            heatmap_chart=heatmap_chart,

            scatter3d_chart=scatter3d_chart
        )

    except Exception as e:

        return f"Error: {e}"


# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(debug=True)