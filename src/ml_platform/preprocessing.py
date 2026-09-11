from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(
        X, y,
        test_size=0.25,
        random_state=42
    )

def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
