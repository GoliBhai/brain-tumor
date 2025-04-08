import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load dataset
try:
    df = pd.read_csv("brain_tumor_dataset.csv")
except FileNotFoundError:
    print("❌ Error: 'brain_tumor_dataset.csv' not found in this directory.")
    exit()

# Encode categorical variables
le_gender = LabelEncoder()
le_location = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Tumor Location"] = le_location.fit_transform(df["Tumor Location"])

# Prepare features and target
X = df.drop("Brain Tumor", axis=1)
y = df["Brain Tumor"]

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Function to collect patient input
def get_patient_data():
    print("\n🧾 Please answer the following questions:")
    age = int(input("Age: "))
    gender = input("Gender (Male/Female): ")
    headache = int(input("Headache (1-Yes, 0-No): "))
    nausea = int(input("Nausea (1-Yes, 0-No): "))
    vomiting = int(input("Vomiting (1-Yes, 0-No): "))
    vision = int(input("Vision Problems (1-Yes, 0-No): "))
    seizures = int(input("Seizures (1-Yes, 0-No): "))
    balance = int(input("Balance Issues (1-Yes, 0-No): "))
    cognitive = int(input("Cognitive Decline (1-Yes, 0-No): "))
    tumor_size = float(input("Tumor Size in cm: "))
    location = input("Tumor Location (Frontal Lobe / Parietal Lobe / Temporal Lobe / Occipital Lobe / Cerebellum / Brain Stem): ")
    genetic = int(input("Family Genetic History (1-Yes, 0-No): "))

    try:
        gender_enc = le_gender.transform([gender])[0]
        location_enc = le_location.transform([location])[0]
    except ValueError:
        print("❌ Invalid input for gender or tumor location. Please try again with the correct labels.")
        exit()

    features = np.array([[
        age, gender_enc, headache, nausea, vomiting, vision,
        seizures, balance, cognitive, tumor_size, location_enc, genetic
    ]])

    return features

# Get new input and predict
patient_data = get_patient_data()
prediction = model.predict(patient_data)[0]

# Display result
print("\n📊 Prediction Result:")
if prediction == 1:
    print("⚠️ The patient is likely to have a brain tumor.")
else:
    print("✅ The patient is not likely to have a brain tumor.")
