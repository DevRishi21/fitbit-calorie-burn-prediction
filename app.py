import streamlit as st
import pandas as pd
import numpy as np
import joblib

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Fitbit Workout Analytics",
    page_icon="🏋️",
    layout="wide"
)

# LOAD SAVED MODELS AND PREPROCESSING OBJECTS
@st.cache_resource
def load_models():

    # Supervised learning model
    xgb_model = joblib.load(
        "models/xgb_calorie_model.pkl"
    )

    # Exact feature order used during regression training
    regression_features = joblib.load(
        "models/regression_features.pkl"
    )

    # Unsupervised learning objects
    cluster_scaler = joblib.load(
        "models/clustering_scaler.pkl"
    )

    cluster_pca = joblib.load(
        "models/clustering_pca.pkl"
    )

    kmeans_model = joblib.load(
        "models/kmeans_model.pkl"
    )

    # Exact feature order used for clustering
    clustering_features = joblib.load(
        "models/clustering_features.pkl"
    )

    cluster_means = joblib.load(
        "models/cluster_means.pkl"
    )

    return (
        xgb_model,
        regression_features,
        cluster_scaler,
        cluster_pca,
        kmeans_model,
        clustering_features,
        cluster_means
    )


# Load everything
try:

    (
        xgb_model,
        regression_features,
        cluster_scaler,
        cluster_pca,
        kmeans_model,
        clustering_features,
        cluster_means

    ) = load_models()

    # After load_models() returns, validate clustering_features
    if not isinstance(clustering_features, list) or not all(isinstance(f, str) for f in clustering_features):
        st.error(
            "clustering_features.pkl did not load a list of column names. "
            "Please re-save it from the notebook."
        )
        st.stop()

except Exception as e:

    st.error(
        "Unable to load the saved model files."
    )

    st.info(
        "Please make sure all required .pkl files are inside the models/ folder."
    )

    st.stop()

# TITLE
st.title("🏋️ Fitbit Workout Analytics")

st.markdown(
    """
    This application uses Machine Learning to:

    - 🔥 Predict calories burned during a workout
    - 📊 Identify workout patterns using clustering

    The application uses the trained models saved from the
    Fitbit Machine Learning project.
    """
)


# SIDEBAR
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Application",
    [
        "🔥 Calorie Prediction",
        "📊 Workout Pattern Clustering"
    ]
)

def get_cluster_interpretation(
    cluster_id,
    cluster_means
):

    cluster = cluster_means.loc[cluster_id]

    # Get intensity-related values
    avg_bpm = cluster.get("Avg_BPM", np.nan)
    hr_intensity = cluster.get("HR_Intensity", np.nan)
    effective_met = cluster.get("Effective_MET", np.nan)
    base_met = cluster.get("Base_MET", np.nan)
    session_duration = cluster.get(
        "Session_Duration",
        np.nan
    )

    # --------------------------------------------------------
    # Calculate relative intensity using the available
    # intensity-related features.
    #
    # We use the rank of the cluster against other clusters
    # instead of hardcoding Cluster 0/1/2.
    # --------------------------------------------------------

    intensity_score = (
        pd.Series({
            "Avg_BPM": avg_bpm,
            "HR_Intensity": hr_intensity,
            "Effective_MET": effective_met,
            "Base_MET": base_met
        })
        .rank(pct=True)
        .mean()
    )

    if intensity_score >= 0.67:

        intensity_label = "Higher-intensity"

        description = (
            "This cluster represents relatively "
            "higher-intensity workout sessions, "
            "characterized by higher heart-rate and "
            "metabolic activity compared with the "
            "other identified clusters."
        )

    elif intensity_score <= 0.33:

        intensity_label = "Lower-intensity"

        description = (
            "This cluster represents relatively "
            "lower-intensity workout sessions, "
            "with comparatively lower heart-rate "
            "and metabolic activity."
        )

    else:

        intensity_label = "Moderate-intensity"

        description = (
            "This cluster represents relatively "
            "moderate-intensity workout sessions, "
            "falling between the lower- and "
            "higher-intensity groups."
        )

    return intensity_label, description

# COMMON INPUT FUNCTION
def get_workout_inputs():

    st.subheader("Workout Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=10,
            max_value=100,
            value=25,
            step=1
        )

        gender = st.selectbox(
            "Gender",
            options=[0, 1],
            format_func=lambda x:
                "Female" if x == 0 else "Male"
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=200.0,
            value=70.0,
            step=0.5
        )

        height = st.number_input(
            "Height (m)",
            min_value=1.0,
            max_value=2.5,
            value=1.70,
            step=0.01
        )

        fat_percentage = st.number_input(
            "Fat Percentage",
            min_value=1.0,
            max_value=60.0,
            value=20.0,
            step=0.5
        )

    with col2:

        max_bpm = st.number_input(
            "Maximum BPM",
            min_value=80,
            max_value=220,
            value=180,
            step=1
        )

        avg_bpm = st.number_input(
            "Average BPM",
            min_value=50,
            max_value=220,
            value=130,
            step=1
        )

        resting_bpm = st.number_input(
            "Resting BPM",
            min_value=30,
            max_value=120,
            value=70,
            step=1
        )

        session_duration = st.number_input(
            "Session Duration (hours)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1
        )

        water_intake = st.number_input(
            "Water Intake (liters)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1
        )

    with col3:

        workout_type = st.selectbox(
            "Workout Type",
            [
                "Cardio",
                "HIIT",
                "Mixed",
                "Strength",
                "Yoga"
            ]
        )

        workout_frequency = st.number_input(
            "Workout Frequency (days/week)",
            min_value=0,
            max_value=7,
            value=3,
            step=1
        )

        experience_level = st.selectbox(
            "Experience Level",
            options=[0, 1, 2, 3]
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=24.0,
            step=0.1
        )

        base_met = st.number_input(
            "Base MET",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.1
        )

    st.divider()

    st.subheader("Workout Intensity")

    col4, col5 = st.columns(2)

    with col4:

        hr_intensity = st.number_input(
            "HR Intensity",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.01
        )

    with col5:

        effective_met = st.number_input(
            "Effective MET",
            min_value=1.0,
            max_value=25.0,
            value=5.0,
            step=0.1
        )

    return {
        "Age": age,
        "Gender": gender,
        "Weight": weight,
        "Height": height,
        "Max_BPM": max_bpm,
        "Avg_BPM": avg_bpm,
        "Resting_BPM": resting_bpm,
        "Session_Duration": session_duration,
        "Workout_Type": workout_type,
        "Fat_Percentage": fat_percentage,
        "Water_Intake": water_intake,
        "Workout_Frequency": workout_frequency,
        "Experience_Level": experience_level,
        "BMI": bmi,
        "Base_MET": base_met,
        "HR_Intensity": hr_intensity,
        "Effective_MET": effective_met
    }

# CREATE MODEL INPUT DATAFRAME
def create_model_dataframe(inputs):

    input_df = pd.DataFrame([inputs])

    # --------------------------------------------------------
    # One-hot encoding for Workout_Type
    #
    # Cardio is the baseline category because the training
    # dataset used drop_first=True.
    # --------------------------------------------------------

    input_df["Workout_Type_HIIT"] = (
        input_df["Workout_Type"] == "HIIT"
    ).astype(int)

    input_df["Workout_Type_Mixed"] = (
        input_df["Workout_Type"] == "Mixed"
    ).astype(int)

    input_df["Workout_Type_Strength"] = (
        input_df["Workout_Type"] == "Strength"
    ).astype(int)

    input_df["Workout_Type_Yoga"] = (
        input_df["Workout_Type"] == "Yoga"
    ).astype(int)

    # Remove original categorical column
    input_df.drop(
        columns=["Workout_Type"],
        inplace=True
    )

    return input_df

# CALORIE PREDICTION
if page == "🔥 Calorie Prediction":

    st.header("🔥 Calories Burned Prediction")

    st.write(
        "Enter the workout details below to predict the "
        "calories burned during the workout session."
    )

    inputs = get_workout_inputs()

    st.divider()

    predict_button = st.button(
        "🔥 Predict Calories Burned",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        # Create dataframe
        input_df = create_model_dataframe(inputs)

        # ----------------------------------------------------
        # Arrange columns in exactly the same order used
        # during model training.
        # ----------------------------------------------------

        try:

            input_df = input_df[
                regression_features
            ]

        except Exception as e:

            st.error(
                "The input features do not match the "
                "features used during model training."
            )

            st.exception(e)

            st.stop()

        # ----------------------------------------------------
        # XGBoost prediction
        #
        # No StandardScaler is applied here because the
        # XGBoost model was trained using the unscaled
        # encoded features.
        # ----------------------------------------------------

        prediction = xgb_model.predict(
            input_df
        )[0]

        prediction = max(0, prediction)

        st.success(
            "Prediction completed successfully!"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🔥 Predicted Calories",
                f"{prediction:.2f} kcal"
            )

        with col2:

            st.metric(
                "⏱️ Session Duration",
                f"{inputs['Session_Duration']:.1f} hrs"
            )

        with col3:

            st.metric(
                "❤️ Average BPM",
                f"{inputs['Avg_BPM']}"
            )

        st.info(
            f"Based on the entered workout details, "
            f"the estimated calorie burn is approximately "
            f"**{prediction:.2f} kcal**."
        )

# ============================================================
# WORKOUT PATTERN CLUSTERING
# ============================================================

elif page == "📊 Workout Pattern Clustering":

    st.header("📊 Workout Pattern Clustering")

    st.write(
        """
        This section identifies the workout pattern of the
        entered session using the trained:

        **StandardScaler → PCA → KMeans**

        pipeline.
        """
    )

    inputs = get_workout_inputs()

    st.divider()

    cluster_button = st.button(
        "📊 Identify Workout Pattern",
        type="primary",
        use_container_width=True
    )

    if cluster_button:

        # ----------------------------------------------------
        # Create dataframe using the same preprocessing
        # used for the regression model.
        # ----------------------------------------------------

        input_df = create_model_dataframe(inputs)

        # ----------------------------------------------------
        # Clustering was performed without Workout_Type.
        #
        # Therefore remove all Workout_Type dummy columns.
        # ----------------------------------------------------

        workout_type_columns = [
            "Workout_Type_HIIT",
            "Workout_Type_Mixed",
            "Workout_Type_Strength",
            "Workout_Type_Yoga"
        ]

        clustering_input = input_df.drop(
            columns=workout_type_columns,
            errors="ignore"
        )

        # ----------------------------------------------------
        # Arrange columns in exactly the same order that was
        # used while training the clustering model.
        # ----------------------------------------------------

        try:

            clustering_input = clustering_input[
                clustering_features
            ]

        except Exception as e:

            st.error(
                "The clustering input features do not match "
                "the features used during clustering."
            )

            st.exception(e)

            st.stop()

        # ----------------------------------------------------
        # STEP 1: StandardScaler
        # ----------------------------------------------------

        scaled_data = cluster_scaler.transform(
            clustering_input
        )

        # ----------------------------------------------------
        # STEP 2: PCA
        # ----------------------------------------------------

        pca_data = cluster_pca.transform(
            scaled_data
        )

        # ----------------------------------------------------
        # STEP 3: KMeans
        # ----------------------------------------------------

        cluster_prediction = kmeans_model.predict(
            pca_data
        )[0]

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        st.success(
            "Workout pattern identified successfully!"
        )

        st.metric(
            "📊 Assigned Cluster",
            f"Cluster {cluster_prediction}"
        )

        # ============================================================
        # CLUSTER INTERPRETATION
        # ============================================================

        intensity_label, description = get_cluster_interpretation(
            cluster_prediction,
            cluster_means
        )

        st.subheader(
            f"🏋️ Workout Pattern: {intensity_label}"
        )

        st.write(description)

        st.info(
            f"The entered workout session belongs to "
            f"**Cluster {cluster_prediction}** based on its "
            f"physiological and workout-related characteristics."
        )

        # ============================================================
        # CLUSTER PROFILE
        # ============================================================

        st.subheader(
            f"📈 Cluster {cluster_prediction} Profile"
        )

        selected_profile = cluster_means.loc[
            cluster_prediction
        ].to_frame(name="Average Value")

        selected_profile.index.name = "Feature"

        st.dataframe(
            selected_profile,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Display the input values used for clustering
        # ----------------------------------------------------

        st.subheader("Workout Characteristics")

        display_data = {
            "Age": inputs["Age"],
            "Weight": inputs["Weight"],
            "Height": inputs["Height"],
            "Max BPM": inputs["Max_BPM"],
            "Average BPM": inputs["Avg_BPM"],
            "Resting BPM": inputs["Resting_BPM"],
            "Session Duration": inputs["Session_Duration"],
            "Fat Percentage": inputs["Fat_Percentage"],
            "Water Intake": inputs["Water_Intake"],
            "Workout Frequency": inputs["Workout_Frequency"],
            "Experience Level": inputs["Experience_Level"],
            "BMI": inputs["BMI"],
            "Base MET": inputs["Base_MET"],
            "HR Intensity": inputs["HR_Intensity"],
            "Effective MET": inputs["Effective_MET"]
        }

        display_df = pd.DataFrame(
            display_data.items(),
            columns=["Feature", "Value"]
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # PCA coordinates
        # ----------------------------------------------------

        st.subheader("PCA Representation")

        pca_col1, pca_col2 = st.columns(2)

        with pca_col1:

            st.metric(
                "Principal Component 1",
                f"{pca_data[0][0]:.3f}"
            )

        with pca_col2:

            st.metric(
                "Principal Component 2",
                f"{pca_data[0][1]:.3f}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fitbit: Calorie Burn Prediction & Workout Pattern "
    )