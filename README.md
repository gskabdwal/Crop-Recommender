# Crop Recommender

A machine learning-powered crop recommendation system built with Streamlit. This application leverages environmental and soil data to suggest the most suitable crop for a given set of conditions. The model is built using XGBoost and integrates seamlessly with a user-friendly Streamlit interface.

Check it out : https://crop-recommender-web.streamlit.app/

## Features

- **Interactive Web App:** Built with Streamlit for an intuitive user experience.
- **Robust Predictions:** Utilizes a pre-trained XGBoost model for accurate recommendations.
- **Efficient Data Handling:** Employs Pandas and NumPy for managing data.
- **Model Serialization:** Uses Joblib to load the pre-trained model quickly.
- **Streamlined Pipeline:** Integrates Scikit-learn for data preprocessing and modeling.

## Requirements

Ensure you have the following packages installed. You can install them all at once using the provided `requirements.txt` file.

### requirements.txt
- streamlit
- numpy
- pandas
- scikit-learn
- xgboost
- joblib
