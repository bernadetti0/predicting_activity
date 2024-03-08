# Data Exploratory:

# Import liabraries :

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Load dataset into pandas DataFrame:

def model_streamlit(watch):

    data_path = "raw_data/aw_fb_data.csv"

    df = pd.read_csv(data_path)


    # Data Cleaning :

    df1 = df.copy()

    df1.drop(['Unnamed: 0', 'X1'], axis=1, inplace=True)

    df1.rename(columns={'hear_rate': 'heart_rate', 'entropy_setps': 'entropy_steps'}, inplace=True)

    df1 = df1.dropna()

    # Feature Engineering :

    df1.insert(loc=0, column='participant_id', value = df1.set_index(['age', 'gender', 'height', 'weight']).index.factorize()[0]+1)

    df1['bmi'] = round(df1.weight / (df1.height/100)**2)

    df1_core_features = df1[['activity', 'device', 'participant_id','age', 'gender', 'height','weight', 'steps', 'heart_rate', 'calories', 'distance', 'bmi']]




    if watch == "apple":

        # Splitting Dataset into Apple and Fitbit :
        df1_apple = df1_core_features[df1_core_features['device']=='apple watch']

        # Features and Target Split :
        X_apple = df1_apple.drop(columns = 'activity').drop(columns="device")
        y_apple = df1_apple['activity']

        # Lable Encoding of Target Value :
        label_encoder_apple = LabelEncoder()
        label_encoder_apple.fit(y_apple)
        y_apple_encoded = label_encoder_apple.transform(y_apple)

        # Features scaling :
        scaler_apple = StandardScaler()
        df_scaled_apple = pd.DataFrame(scaler_apple.fit_transform(X_apple),columns= scaler_apple.get_feature_names_out())

        # Train-Test Split Data :
        X_train_apple, X_test_apple, y_train_apple, y_test_apple = train_test_split(df_scaled_apple, y_apple_encoded, test_size=0.2, random_state=42)


        # Building Model :

        # Random Forest(Apple) :
        rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
        rf_model.fit(X_train_apple, y_train_apple)
        y_apple_pred_rf = rf_model.predict(X_test_apple)

        accuracy_apple = accuracy_score(y_test_apple, y_apple_pred_rf)

        return rf_model


    if watch == "fitbit":

        # Splitting Data into Apple and Fitbit :
        df1_fitbit = df1_core_features[df1_core_features['device']=='fitbit']

        # Features and Target Split :
        X_fitbit = df1_fitbit.drop(columns = 'activity').drop(columns="device")
        y_fitbit = df1_fitbit['activity']

        # Lable Encoding of Target Value :
        label_encoder_fitbit = LabelEncoder()
        label_encoder_fitbit.fit(y_fitbit)
        y_fitbit_encoded = label_encoder_fitbit.transform(y_fitbit)

        # Features scaling :
        scaler_fitbit = StandardScaler()
        df_scaled_fitbit = pd.DataFrame(scaler_fitbit.fit_transform(X_fitbit),columns= scaler_fitbit.get_feature_names_out())

        # Train-Test Split Data :
        X_train_fitbit, X_test_fitbit, y_train_fitbit, y_test_fitbit = train_test_split(df_scaled_fitbit, y_fitbit_encoded, test_size=0.2, random_state=42)

        # Building Model :

        # Random Forest(Fitbit) :

        rf_model2 = RandomForestClassifier(n_estimators=100, random_state=0)
        rf_model2.fit(X_train_fitbit, y_train_fitbit)
        y_fitbit_pred_rf = rf_model2.predict(X_test_fitbit)

        accuracy_fitbit = accuracy_score(y_test_fitbit, y_fitbit_pred_rf)

        return rf_model2
