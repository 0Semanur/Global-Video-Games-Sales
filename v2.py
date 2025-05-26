import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv(r"C:\\Users\\Semanur\\Desktop\\vgsales.csv")

# Display the first few rows
print(df.head())

# Check for missing values
print('Missing values in each column:')
print(df.isnull().sum())

# Clean 'Year' column
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Year'].fillna(int(df['Year'].median()), inplace=True)
#Fill missing value in the Publisher column
df['Publisher'].fillna('Unknown', inplace=True)
print('\nAfter cleaning, missing values:')
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Descriptive statistics
print('Descriptive statistics:')
print(df.describe())

# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# Pairplot
sns.pairplot(numeric_df)
plt.suptitle('Pair Plot of Numeric Features', y=1.02)
plt.tight_layout()
plt.show()

# Histogram of Global Sales
plt.figure(figsize=(8, 6))
sns.histplot(df['Global_Sales'], bins=30, kde=True)
plt.title('Distribution of Global Sales')
plt.xlabel('Global Sales (millions)')
plt.tight_layout()
plt.show()

# Count plot for Genre
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='Genre', order=df['Genre'].value_counts().index)
plt.title('Count of Games by Genre')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Boxen plot for Platform and Global Sales
plt.figure(figsize=(12, 8))
sns.boxenplot(data=df, x='Platform', y='Global_Sales')
plt.title('Global Sales by Platform')
plt.xlabel('Platform')
plt.ylabel('Global Sales (millions)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create binary target: Hit or not
df['Hit'] = (df['Global_Sales'] > 1.0).astype(int)

# Features and target
features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Genre', 'Platform', 'Publisher', 'Year']
X = df[features]
y = df['Hit']

# Preprocessing: scale numerical and encode categorical features
numeric_features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Year']
categorical_features = ['Genre', 'Platform', 'Publisher']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Build pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Fit model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("[Logistic Regression]")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
print(f'Cross-validated F1 Score (Balanced Logistic Regression): {cv_scores.mean():.2f} ± {cv_scores.std():.2f}')

# Trend analysis: yearly hit rate
yearly_hits = df.groupby('Year')['Hit'].mean()
plt.figure(figsize=(10, 5))
plt.plot(yearly_hits.index, yearly_hits.values, marker='o')
plt.title('Hit Rate by Year')
plt.xlabel('Year')
plt.ylabel('Proportion of Hit Games')
plt.tight_layout()
plt.show()
print("Train F1:", f1_score(y_train, model.predict(X_train)))
print("Test F1:", f1_score(y_test, y_pred))
