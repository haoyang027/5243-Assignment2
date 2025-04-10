import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    Preprocess the dataset by separating features and target variable,
    handling categorical data, and performing normalization.

    Parameters:
    data (pd.DataFrame): The input dataset.

    Returns:
    tuple: A tuple containing the standardized features and target variable.
    """
    # Drop rows with missing values
    data = data.dropna()

    # Assuming the target variable is 'Eg (eV)' and all other columns are features
    target = data['Eg (eV)']
    features = data.drop(columns=['Eg (eV)'])

    # Identify categorical columns
    categorical_columns = features.select_dtypes(include=['object']).columns

    # Debug: Print data types before encoding
    print("Data types before encoding:")
    print(features.dtypes)

    # Check unique values in the `composition` column
    if 'composition' in features.columns:
        print("Unique values in 'composition':")
        print(features['composition'].unique())

    # Perform one-hot encoding for categorical features
    if len(categorical_columns) > 0:
        print(f"Encoding categorical columns: {categorical_columns}")
        features = pd.get_dummies(features, columns=categorical_columns, drop_first=True)
        print("Features after one-hot encoding:")
        print(features.head())

    # Debug: Print data types after encoding
    print("Data types after encoding:")
    print(features.dtypes)

    # Standardize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, target