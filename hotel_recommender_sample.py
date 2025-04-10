import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load Expedia dataset (you can modify the path)
# df = pd.read_csv('train.csv')
df = pd.read_csv('train.csv',
                 usecols=['user_location_country', 'srch_destination_id', 'is_mobile', 'orig_destination_distance', 'hotel_cluster'],
                 nrows=50000)

# Select only necessary columns
df = df[['user_location_country', 'srch_destination_id', 'is_mobile', 'orig_destination_distance', 'hotel_cluster']]

# Drop rows with missing values
df.dropna(inplace=True)

# Features and Target
X = df[['user_location_country', 'srch_destination_id', 'is_mobile', 'orig_destination_distance']]
y = df['hotel_cluster']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=30, max_depth=7, random_state=42)
model.fit(X_train, y_train)

# Save the model
pickle.dump(model, open('hotel_model.pkl', 'wb'))

print("✅ Model trained and saved as hotel_model.pkl")
