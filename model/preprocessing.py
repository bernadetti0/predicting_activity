import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset into pandas DataFrame:

def test_preprocessor(watch, df):
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

        # Features scaling :
        scaler_apple = StandardScaler()
        df_scaled_apple = pd.DataFrame(scaler_apple.fit_transform(X_apple),columns= scaler_apple.get_feature_names_out())

        return df_scaled_apple


    if watch == "fitbit":

        # Splitting Data into Apple and Fitbit :
        df1_fitbit = df1_core_features[df1_core_features['device']=='fitbit']

        # Features and Target Split :
        X_fitbit = df1_fitbit.drop(columns = 'activity').drop(columns="device")

        # Features scaling :
        scaler_fitbit = StandardScaler()
        df_scaled_fitbit = pd.DataFrame(scaler_fitbit.fit_transform(X_fitbit),columns= scaler_fitbit.get_feature_names_out())

        return df_scaled_fitbit
