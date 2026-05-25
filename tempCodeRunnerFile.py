        # HEATMAP
        # =========================

        # CREATE DATAFRAME

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

        # CREATE MORE ROWS

        heatmap_df = pd.concat(
            [heatmap_df] * 100,
            ignore_index=True
        )

        # ADD RANDOM VARIATION

        heatmap_df["Area"] += np.random.randint(
            -500,
            500,
            size=100
        )

        heatmap_df["Bedrooms"] += np.random.randint(
            0,
            3,
            size=100
        )

        heatmap_df["Bathrooms"] += np.random.randint(
            0,
            3,
            size=100
        )

        heatmap_df["Stories"] += np.random.randint(
            0,
            2,
            size=100
        )

        heatmap_df["Parking"] += np.random.randint(
            0,
            2,
            size=100
        )

        # CORRELATION MATRIX

        corr_matrix = heatmap_df.corr(numeric_only=True)

        # HEATMAP

        fig4 = px.imshow(

            corr_matrix,

            text_auto=".2f",

            color_continuous_scale="Purples",

            title="Feature Correlation Heatmap"
        )

        # UPDATE LAYOUT

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

            width=800,
            height=450
        )

        # BETTER CELL DISPLAY

        fig4.update_xaxes(side="bottom")

        # HTML

        heatmap_chart = fig4.to_html(full_html=False)