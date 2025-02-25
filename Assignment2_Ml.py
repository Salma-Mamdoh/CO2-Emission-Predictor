# %% [markdown]
# ## About the Dataset
# 
# This dataset captures details on how **weather-related features** such as temperature, humidity, wind speed, cloud cover, and pressure relate to the likelihood of **rain**. The dataset contains information on weather conditions and is compiled across a period, providing insights into how various weather attributes influence precipitation. The dataset includes **2,500 rows** and **6 columns**.
# 
# ### Key Information
# 
# - **Weather Features**:
#   - **Temperature**: The ambient temperature in degrees Celsius.
#   - **Humidity**: The percentage of moisture in the air.
#   - **Wind Speed**: The speed of the wind in meters per second.
#   - **Cloud Cover**: The percentage of sky covered by clouds.
#   - **Pressure**: The atmospheric pressure in hectopascals (hPa).
# 
# - **Target Variable**:
#   - **Rain**: Indicates whether it rained or not (binary classification: "rain" or "no rain").
# 
# ### Column Descriptions
# 
# - **Temperature**: Ambient temperature in degrees Celsius.
# - **Humidity**: The percentage of moisture present in the air.
# - **Wind Speed**: The speed of wind measured in meters per second.
# - **Cloud Cover**: The percentage of sky covered by clouds.
# - **Pressure**: The atmospheric pressure recorded in hectopascals.
# - **Rain**: The target variable, indicating whether it rained (1) or did not rain (0) based on the weather conditions. 
# 
# This dataset can be used to predict the likelihood of rain based on various weather parameters like temperature, humidity, and wind speed, which can be valuable for weather forecasting and climate studies.
# 
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 6)  
import warnings
warnings.filterwarnings('ignore') 

# %% [markdown]
# ## Task 1: Preprocessing 

# %%
df = pd.read_csv("weather_forecast_data.csv")
df.head()

# %%
df.shape

# %%
print("Dataset Information: \n")
df.info()

# %% [markdown]
# #### Unique Values and Value Counts

# %%
for col in df.select_dtypes(include='object').columns:
    print(f"\nUnique values in {col}:")
    print(df[col].value_counts())

# %% [markdown]
# # Data Cleaning

# %%
def Missing_Data_Check(df):
    print("\nMissing Data Check:")
    missing_data = df.isnull().sum()
    print(missing_data)


# %%
Missing_Data_Check(df)


# %% [markdown]
# #### Data Have missing Values lets identify them 

# %%
# display rows with missing data
print("\nRows with Missing Data:")
print(df[df.isnull().any(axis=1)])

# %%
# Duplicated data
print("Duplicates in df :", df.duplicated().sum())

# %%
# Print the mean of the numeric columns
print("Dataset Mean Summary: \n")
print(df.select_dtypes(include=['float64']).mean())


# %%
# Display the number of unique values in each column of the dataset
print("Unique values in dataset:")
df_unique_counts = df.nunique()
print(df_unique_counts)

# %% [markdown]
# ### Apply the two techniques to handle missing data, dropping missing values and replacing them with the average of the feature.

# %%
def handle_missing_data(df, method='replace'):
    df_copy = df.copy()
    
    if method == 'replace':
        df_copy.fillna(df_copy.select_dtypes(include=['float64']).mean(), inplace=True)
        print("Missing values have been replaced with the mean of each feature.")
        return df_copy
    elif method == 'drop':
        df_copy.dropna(inplace=True)
        print(f"Rows with missing values have been dropped. Remaining rows: {len(df_copy)}.")
        return df_copy
    else:
        print("Invalid method! Please use 'replace' or 'drop'.")
        return df_copy


# %%
df_cleaned_using_Replace = handle_missing_data(df, method='replace')

# %%
df_cleaned_using_drop = handle_missing_data(df, method='drop') # original data 2500 row 

# %%
Missing_Data_Check(df_cleaned_using_Replace)


# %%
Missing_Data_Check(df_cleaned_using_drop)


# %%
# Identifying numerical and non-numerical columns in the dataset
numerical_df = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
non_numerical_df = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

print("\nNumerical columns in the dataset:")
print(numerical_df)

print("\nNon-numerical columns in the dataset:")
print(non_numerical_df)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

def plot_bar_graphs(df, columns):
    for column in columns:
        fig, ax = plt.subplots(figsize=(15, 5))

        sns.countplot(x=column, data=df, order=df[column].value_counts().index, ax=ax, palette="crest")

        total = len(df[column])  
        for p in ax.patches:
            count = int(p.get_height())
            percentage = f'{count / total:.1%}'  
            ax.annotate(f'{count}\n{percentage}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=8, color='black',
                        xytext=(0, 5), textcoords='offset points')

        ax.set_xlabel(column.replace('_', ' ').title(), fontsize=12, labelpad=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'Distribution of {column.replace("_", " ").title()}', fontsize=18, pad=15)

        plt.xticks(rotation=45, ha='right', fontsize=12)
        
        plt.tight_layout()
        plt.show()

# Categorical feature 'Rain' from your dataset
cat_features = ['Rain']

plot_bar_graphs(df, cat_features)


# %%
def plot_histograms(df, columns):
    for column in columns:
        fig, ax = plt.subplots(figsize=(15, 5))

        sns.histplot(df[column], kde=True, ax=ax, color="blue", bins=20)  # You can adjust bins for better clarity
        ax.set_xlabel(column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Distribution of {column.replace("_", " ").title()}', fontsize=18, pad=15)

        plt.tight_layout()
        plt.show()

# Numerical features from your dataset
num_features = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']

plot_histograms(df, num_features)


# %% [markdown]
# # Check whether numeric features have the same scale
# 

# %%
df_cleaned_using_Replace.describe().T

# %%
def plot_Box_plot(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df.select_dtypes(include='float64'))
    plt.xticks(rotation=45)
    plt.title("Comparison of Numeric Feature Scales")
    plt.show()

# %%
plot_Box_plot(df_cleaned_using_Replace)

# %% [markdown]
# ### Check whether Numeric Features Have the Same Scale
# 
# The numeric features do not appear to be on the same scale. Here’s why:
# 
# | Feature      | Mean       | Min        | Max        |
# |--------------|------------|------------|------------|
# | Temperature  | 22.573777  | 10.001842  | 34.995214  |
# | Humidity     | 64.366909  | 30.005071  | 99.997481  |
# | Wind_Speed   | 9.911826   | 0.009819   | 19.999132  |
# | Cloud_Cover  | 49.808770  | 0.015038   | 99.997795  |
# | Pressure     | 1014.409327| 980.014486 | 1049.985593|
# 
# 
# The features have different ranges, means, and standard deviations, confirming that they are not on the same scale. This could affect certain analyses and models. To improve model performance, you might need to normalize or standardize these features to bring them to the same scale.
# 
# ### Note : We will apply scaling After **Spliting Data Into Train and Test to Avoid Data Leakage**

# %% [markdown]
# #### Sperate Data Into Train and Test

# %%
from sklearn.model_selection import train_test_split

def Sepearating_features_and_targets(df):
    X = df.drop(columns=['Rain'])  
    y = df['Rain']   

    print("Features : \n")
    print(X.head())
    print(X.shape)

    print("\n Targets :")
    print(y.head())
    print(y.shape)
    return X,y

# %%
X,y=Sepearating_features_and_targets(df_cleaned_using_Replace)

# %%
def Split_the_data_into_training_and_testing_sets(X,y):
    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training data shape (X_train): ", X_train.shape)
    print("Testing data shape (X_test): ", X_test.shape)
    print("Training target shape (y_train): ", y_train.shape)
    print("Testing target shape (y_test): ", y_test.shape)
    return X_train, X_test, y_train, y_test

# %%
X_train, X_test, y_train, y_test = Split_the_data_into_training_and_testing_sets(X,y)

# %% [markdown]
# #### Encoding For Target Column

# %%
from sklearn.preprocessing import LabelEncoder


def Encode_Target(y_train,y_test,label_encoder):

    y_train = pd.DataFrame(y_train)  
    y_test = pd.DataFrame(y_test)    

    y_train['Rain'] = label_encoder.fit_transform(y_train['Rain'])
    y_test['Rain'] = label_encoder.transform(y_test['Rain'])


    print("\nEncoded Training Target (y_train):")
    print(y_train)

    print("\nEncoded Test Target (y_test):")
    print(y_test)
    return y_train,y_test

# %%
label_encoder = LabelEncoder()
y_train,y_test=Encode_Target(y_train,y_test,label_encoder)


# %% [markdown]
# ### Scaling numeric features 
# 

# %% [markdown]
# We use **StandardScaler**  to standardize numeric columns in the dataset. Standardization is the process of scaling features so they have a mean of 0 and a standard deviation of 1, which helps algorithms perform better by ensuring that features contribute equally.
# The formula for standardization is:
# 
# The standardization equation is:
# 
# $$
# z = \frac{x - \mu}{\sigma}
# $$
# 
# where:
# - $x$ is the original feature value,
# - $\mu$ is the mean of the feature in the training set,
# - $\sigma$ is the standard deviation of the feature in the training set,
# - $z$ is the standardized value.
# 
# 

# %%
from sklearn.preprocessing import StandardScaler
def Scale_Data(X_train,X_test):
    numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns

    scaler = StandardScaler()

    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])

    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

    print("Standardized Training Data:")
    print(X_train.head())

    print("\nStandardized Test Data:")
    print(X_test.head())
    return X_train,X_test


# %%
X_train,X_test=Scale_Data(X_train,X_test)


# %%
# display the mean and standard deviation after standardization
numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
print("\nMean after Standardization:")
print(X_train[numeric_columns].mean())
print("\nStandard Deviation after Standardization:")
print(X_train[numeric_columns].std())

# %%
plot_Box_plot(X_train)

# %%
plot_Box_plot(X_test)

# %% [markdown]
# ## Task 2: Implement Decision Tree, k-Nearest Neighbors (kNN) and naïve Bayes 
# ### Note This Models using Dataframe which handled missing values using -- **Replace By Average Technique**

# %%
from sklearn.metrics import classification_report

def print_classification_report(model_name,y_test, y_pred):
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred))


# %%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(model_name,y_test, y_pred):
    y_true_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred)
    cm = confusion_matrix(y_true_original, y_pred_original)
    unique_classes = sorted(set(y_true_original) | set(y_pred_original))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

# %% [markdown]
# ### Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def Decision_Tree(X_train,X_test,y_train,y_test):
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_dt)
    print(f'Accuracy of the Decsion Tree model on the test set: {accuracy:.4f}')
    return y_pred_dt

# %%
y_pred_dt=Decision_Tree(X_train,X_test,y_train,y_test)

# %%
print_classification_report("Decsion Tree",y_test,y_pred_dt)
    



# %%
plot_confusion_matrix("Decision Tree",y_test,y_pred_dt)

# %% [markdown]
# ### k-Nearest Neighbors (kNN)

# %%
from sklearn.neighbors import KNeighborsClassifier
def Knn(X_train,X_test,y_train,y_test, n_neighbors):
    knn_model = KNeighborsClassifier(n_neighbors)  
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_knn)
    print(f'Accuracy of the KNN model on the test set: {accuracy:.4f}')
    return y_pred_knn

# %%
y_pred_knn=Knn(X_train,X_test,y_train,y_test, 5)

# %%
print_classification_report("kNN using Skit-learn",y_test,y_pred_knn)

# %%
plot_confusion_matrix("kNN using Skit-learn",y_test,y_pred_knn)

# %% [markdown]
# ### Naïve Bayes

# %%
from sklearn.naive_bayes import GaussianNB
def Naïve_Bayes(X_train,X_test,y_train,y_test):
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_nb)
    print(f'Accuracy of the Naïve Bayes model on the test set: {accuracy:.4f}')
    return y_pred_nb

# %%
y_pred_nb=Naïve_Bayes(X_train,X_test,y_train,y_test)

# %%
print_classification_report("Naïve Bayes",y_test,y_pred_nb)

# %%
plot_confusion_matrix("Naïve Bayes",y_test,y_pred_nb)

# %% [markdown]
# ### Compare Performace of 3 Algorithms:
# 
# 1. **Decision Tree**:
#    - **Precision**: Perfect (1.00) for both "no rain" and "rain" classes.
#    - **Recall**: Perfect for "no rain" (1.00) but slightly lower for "rain" (0.96).
#    - **F1-Score**: Near-perfect for "rain" (0.98) and perfect for "no rain" (1.00).
#    - **Accuracy**: 100% (best among the three).
#    - **Confusion Matrix**:
#      - Only 2 false negatives (classifying "rain" as "no rain").
#    - **Overall**: The Decision Tree is the most accurate, performing well for both classes.
# 
# 2. **kNN**:
#    - **Precision**: 0.97 for "no rain" and 0.92 for "rain."
#    - **Recall**: 0.99 for "no rain," but only 0.79 for "rain."
#    - **F1-Score**: 0.98 for "no rain" and 0.85 for "rain."
#    - **Accuracy**: 97%.
#    - **Confusion Matrix**:
#      - 4 false positives (classifying "no rain" as "rain").
#      - 12 false negatives (classifying "rain" as "no rain").
#    - **Overall**: Performs well but struggles more with the "rain" class compared to the Decision Tree.
# 
# 3. **Naïve Bayes**:
#    - **Precision**: 0.96 for "no rain" and 1.00 for "rain."
#    - **Recall**: 1.00 for "no rain," but only 0.68 for "rain."
#    - **F1-Score**: 0.98 for "no rain" and 0.81 for "rain."
#    - **Accuracy**: 96% (lowest among the three).
#    - **Confusion Matrix**:
#      - 18 false negatives (classifying "rain" as "no rain").
#    - **Overall**: Struggles significantly with detecting "rain," despite high precision for the "rain" class.
# 
# The **Decision Tree** is the best overall performer, with perfect accuracy and minimal false negatives.
# 
# 

# %% [markdown]
# ### Implement k-Nearest Neighbors (kNN) algorithm from scratch
# 

# %%
def initialize_knn(k=3):
    return {"k": k, "X_train": None, "y_train": None}


# %%
def fit_knn(model, X_train, y_train):
    model["X_train"] = np.array(X_train)
    model["y_train"] = np.array(y_train)

# %% [markdown]
# ### Euclidean Distance
# 
# The Euclidean distance is a measure of the straight-line distance between two points in multi-dimensional space. It is given by the formula:
# 
# $$
# d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
# $$
# 
# 

# %%
def euclidean_distance(X_train, x_test):
    X_train = np.array(X_train, dtype=np.float64)
    x_test = np.array(x_test, dtype=np.float64)
    
    differences = X_train - x_test
    squared_differences = differences ** 2
    sum_squared_differences = np.sum(squared_differences, axis=1)
    distances = np.sqrt(sum_squared_differences)
    return distances


# %%
def get_k_neighbors(distances, y_train, k):
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]
    
    return k_labels  



# %%
from collections import Counter
import numpy as np

def predict_knn(model, X_test):
    predictions = []
    
    X_test = np.array(X_test)
    
    for i in range(X_test.shape[0]):  
        x_test = X_test[i]          
        distances = euclidean_distance(model["X_train"], x_test)  
        neighbors = get_k_neighbors(distances, model["y_train"], model["k"])  
        neighbors = [label for label in neighbors]
        most_common = Counter(neighbors).most_common(1)
        predictions.append(most_common[0][0])  # Append the predicted label
    
    return np.array(predictions)


# %%
knn_model = initialize_knn(k=5)

fit_knn(knn_model, X_train, np.array(y_train).ravel())

y_pred_knn_from_Scratch = predict_knn(knn_model, X_test)
accuracy = accuracy_score(y_test, y_pred_knn_from_Scratch)
print(f'Accuracy of the KNN model from scratch on the test set: {accuracy:.4f}')


# %%
print_classification_report("Knn model From Scratch",y_test,y_pred_knn_from_Scratch)

# %%
plot_confusion_matrix("Knn model From Scratch",y_test,y_pred_knn_from_Scratch)

# %% [markdown]
# ##### Report the results and compare the performance of your custom k Nearest Neighbors (kNN) implementation with the pre-built kNN algorithms in scikit-learn using Replacing Technique in Handling Missing values
# 

# %% [markdown]
# 
# ### **Classification Report Comparison**
# Both implementations produced identical classification metrics, indicating that their performances are identical in terms of precision, recall, F1-score, and overall accuracy.
# 
# | Metric        | Class 0 (No Rain) | Class 1 (Rain) | Accuracy |
# |---------------|-------------------|----------------|----------|
# | **Precision** | 0.97              | 0.92           | 0.97     |
# | **Recall**    | 0.99              | 0.79           |          |
# | **F1-score**  | 0.98              | 0.85           |          |
# 
# 
# 
# ### **Confusion Matrix Comparison**
# Both implementations produced the same confusion matrix:
# 
# | **Predicted →**    | **No Rain** | **Rain** |
# |--------------------|-------------|----------|
# | **Actual No Rain** | 440         | 4        |
# | **Actual Rain**    | 12          | 44       |
# 
# 
# ***Both implementations performed identically on this dataset***
# 
# 
# 
# 
# 
# 

# %% [markdown]
# ## Task 3: Interpreting the Decision Tree and Evaluation Metrics Report

# %% [markdown]
# ### Trying The Same Models But Using Different Missing Values Handling Technique -- Drop Missing Values 

# %% [markdown]
# #### Processing 

# %%
X2,y2=Sepearating_features_and_targets(df_cleaned_using_drop)

# %%
X2_train, X2_test, y2_train, y2_test = Split_the_data_into_training_and_testing_sets(X2,y2)

# %%
label_encoder2=LabelEncoder()
y2_train,y2_test=Encode_Target(y2_train,y2_test,label_encoder2)

# %%
X2_train,X2_test=Scale_Data(X2_train,X2_test)

# %%
plot_Box_plot(X2_train)

# %%
plot_Box_plot(X2_test)

# %% [markdown]
# ### Models using DF handeled by Drop missing Values

# %% [markdown]
# #### Descion Tree

# %%
y_pred_dt2=Decision_Tree(X2_train,X2_test,y2_train,y2_test)

# %%
print_classification_report("Descion Tree",y2_test,y_pred_dt2)

# %%
plot_confusion_matrix("Descion Tree",y2_test,y_pred_dt2)

# %% [markdown]
# #### Knn

# %%
y_pred_knn2=Knn(X2_train,X2_test,y2_train,y2_test,5)

# %%
print_classification_report("knn",y2_test,y_pred_knn2)

# %%
plot_confusion_matrix("knn",y2_test,y_pred_knn2)

# %% [markdown]
# #### Naïve Bayes

# %%
y_pred_nb2=Naïve_Bayes(X2_train,X2_test,y2_train,y2_test)

# %%
print_classification_report("Naïve Bayes",y2_test,y_pred_nb2)

# %%
plot_confusion_matrix("Naïve Bayes",y2_test,y_pred_nb2)

# %% [markdown]
# ### **1. The effect of different data handling**

# %% [markdown]
# ## Comparison of Decision Tree Performance with Different Missing Value Handling Techniques
# 
# ### 1. Replacing Missing Values with Average
# 
# **Classification Report:**
# 
# | Class | Precision | Recall | F1-Score | Support |
# |-------|-----------|--------|----------|---------|
# | 0     | 1.00      | 1.00   | 1.00     | 444     |
# | 1     | 1.00      | 0.96   | 0.98     | 56      |
# 
# - **Accuracy:** 1.00  
#  
# 
# **Confusion Matrix:**
# 
# | True Label / Predicted Label | No Rain | Rain |
# |------------------------------|---------|------|
# | No Rain                      | 444     | 0    |
# | Rain                         | 2       | 54   |
# 
# 
# ### 2. Dropping Missing Values
# 
# **Classification Report:**
# 
# | Class | Precision | Recall | F1-Score | Support |
# |-------|-----------|--------|----------|---------|
# | 0     | 1.00      | 1.00   | 1.00     | 402     |
# | 1     | 1.00      | 0.99   | 0.99     | 68      |
# 
# **Overall Metrics:**
# - **Accuracy:** 1.00  
# 
# 
# **Confusion Matrix:**
# 
# | True Label / Predicted Label | No Rain | Rain |
# |------------------------------|---------|------|
# | No Rain                      | 402     | 0    |
# | Rain                         | 1       | 67   |
# 
# ---
# 
# Both techniques achieved perfect accuracy (1.00), but subtle differences in other metrics were observed:
# 
# 1. **Recall for "Rain" Class**: Slightly higher when dropping missing values (0.99) compared to replacing them with the average (0.96).
# 2. **F1-Score for "Rain" Class**: Marginally better when dropping missing values (0.99 vs. 0.98).
# 3. **Confusion Matrix**: Both models showed excellent classification of "No Rain" samples, with minor differences in misclassification rates for the "Rain" class.
# 
# Replacing missing values allowed for a larger training dataset, which may benefit generalization in other scenarios.
# 
# 

# %% [markdown]
# ## Comparison of kNN Performance with Different Missing Value Handling Techniques
# 
# ### 1. Replacing Missing Values with Average
# 
# **Classification Report:**
# 
# | Class | Precision | Recall | F1-Score | Support |
# |-------|-----------|--------|----------|---------|
# | 0     | 0.97      | 0.99   | 0.98     | 444     |
# | 1     | 0.92      | 0.79   | 0.85     | 56      |
# 
# - **Accuracy:** 0.97  
#  
# 
# **Confusion Matrix:**
# 
# | True Label / Predicted Label | No Rain | Rain |
# |------------------------------|---------|------|
# | No Rain                      | 440     | 4    |
# | Rain                         | 12      | 44   |
# 
# 
# ### 2. Dropping Missing Values
# 
# **Classification Report:**
# 
# | Class | Precision | Recall | F1-Score | Support |
# |-------|-----------|--------|----------|---------|
# | 0     | 0.97      | 0.98   | 0.98     | 402     |
# | 1     | 0.89      | 0.84   | 0.86     | 68      |
# 
# **Overall Metrics:**
# - **Accuracy:** 0.96  
# 
# 
# **Confusion Matrix:**
# 
# | True Label / Predicted Label | No Rain | Rain |
# |------------------------------|---------|------|
# | No Rain                      | 395     | 7    |
# | Rain                         | 11      | 57   |
# 
# ---
# 
# 
# 1. **Replacing Missing Values with Average**:
#    - Slightly improved overall accuracy (0.97 vs. 0.96 when dropping values).
#    - Precision and recall for the "Rain" class were lower, with a higher number of misclassified "Rain" samples (12 vs. 11).
# 
# 2. **Dropping Missing Values**:
#    - Lower accuracy (0.96), but better recall (0.84) and F1-score (0.86) for the "Rain" class.
#    - More balanced results, with fewer false positives for the "Rain" class (7 vs. 4).
# 
# 3. **Trade-offs**:
#    - Replacing missing values preserves a larger dataset, potentially enhancing model generalization.
#    - Dropping missing values improves performance for the minority "Rain" class, making it more effective at detecting "Rain."
# 

# %% [markdown]
# ## Comparison of Naïve Bayes Performance with Different Missing Value Handling Techniques
# 
# ### 1. Replacing Missing Values with Average
# 
# **Classification Report:**
# 
# | Class | Precision | Recall | F1-Score | Support |
# |-------|-----------|--------|----------|---------|
# | 0     | 0.96      | 1.00   | 0.98     | 444     |
# | 1     | 1.00      | 0.68   | 0.81     | 56      |
# 
# **Overall Metrics:**
# - **Accuracy:** 0.96  
# 
# **Confusion Matrix:**
# 
# | True Label / Predicted Label | No Rain | Rain |
# |------------------------------|---------|------|
# | No Rain                      | 444     | 0    |
# | Rain                         | 18      | 38   |
# 
# ---
# 
# ### 2. Dropping Missing Values
# 
# **Classification Report:**
# 
# | Class | Precision | Recall | F1-Score | Support |
# |-------|-----------|--------|----------|---------|
# | 0     | 0.96      | 1.00   | 0.98     | 402     |
# | 1     | 1.00      | 0.74   | 0.85     | 68      |
# 
# **Overall Metrics:**
# - **Accuracy:** 0.96  
# 
# **Confusion Matrix:**
# 
# | True Label / Predicted Label | No Rain | Rain |
# |------------------------------|---------|------|
# | No Rain                      | 402     | 0    |
# | Rain                         | 18      | 50   |
# 
# ---
# 
# 
# 1. **Replacing Missing Values with Average**:
#    - Maintains **96% accuracy**.
#    - **Lower recall and F1-score for "Rain"** (0.68 and 0.81 respectively), leading to more misclassified "Rain" cases (18 false negatives).
# 
# 2. **Dropping Missing Values**:
#    - Keeps **96% accuracy**.
#    - **Improved recall and F1-score for "Rain"** (0.74 and 0.85), with fewer false negatives.
# 
# 3. **Trade-offs**:
#    - **Replacing missing values** keeps more data but sacrifices recall for the minority class.
#    - **Dropping missing values** reduces dataset size but better detects "Rain", improving the model's effectiveness for the minority class.
# 
# 

# %% [markdown]
# ### 2. Decision Tree Explanation Report

# %% [markdown]
# A decision tree is a supervised learning algorithm used for classification and regression tasks.
# It makes predictions by splitting the data into smaller subsets based on specific criteria until a prediction can be made.
# The splits are made using measures like **Entropy and information Gain**.
# 

# %%
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

clf = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=8)
clf.fit(X_train, y_train)

# Visualize the decision tree layer-by-layer
def plot_tree_by_depth(clf, feature_names, max_depth):
    for depth in range(1, max_depth + 1):
        plt.figure(figsize=(16, 10))
        plot_tree(clf, max_depth=depth, feature_names=feature_names, 
                  class_names=label_encoder.classes_, filled=True, rounded=True)
        plt.title(f"Decision Tree Visualization - Depth {depth}")
        plt.show()

tree_max_depth = clf.get_depth()

plot_tree_by_depth(clf, X_train.columns, max_depth=tree_max_depth)



# %% [markdown]
# ### **Layer-by-Layer Analysis**
# 
# #### **Root Node (Depth 0):**
# - **Feature Used**: Humidity
# - **Threshold**: Humidity <= 0.287
# - **Samples**: 2000
# - **Class Distribution**: [1742, 258] (1742 samples of `No`, 258 samples of `Yes`)
# - **Entropy**: 0.555
# - **Majority Class**: No
# 
# At this level, the dataset is split into two branches:
# - **Left Branch (True)**: Humidity <= 0.287 (1187 samples)
# - **Right Branch (False)**: Humidity > 0.287 (813 samples)
# 
# ---
# 
# #### **Depth 1**:
# **Left Branch (Humidity <= 0.287):**
# - **Feature Used**: Humidity
# - **Threshold**: Humidity <= -0.005
# - **Samples**: 1187
# - **Entropy**: 0.025
# - **Majority Class**: No
# 
# This branch is further split based on humidity:
# - **Left Child (True)**: All 983 samples are classified as `No`.
# - **Right Child (False)**: Remaining samples (204) continue splitting.
# 
# **Right Branch (Humidity > 0.287):**
# - **Feature Used**: Cloud Cover
# - **Threshold**: Cloud Cover <= 0.011
# - **Samples**: 813
# - **Entropy**: 0.897
# - **Class Distribution**: [558, 255]
# 
# This branch splits further:
# - **Left Child (True)**: Cloud Cover <= 0.011 (408 samples)
# - **Right Child (False)**: Cloud Cover > 0.011 (405 samples)
# 
# ---
# 
# #### **Depth 2**:
# **For Left Branch of Humidity <= 0.287:**
# - Nodes with homogeneous data stop splitting (entropy = 0). Example:
#   - The first left node contains 983 samples, all of which are `No`.
# 
# **For Cloud Cover Branch:**
# - Left Child (Cloud Cover <= 0.011):
#   - Splits further using **Pressure <= -1.55** and other thresholds.
# 
# - Right Child (Cloud Cover > 0.011):
#   - Splits further based on **Temperature <= 0.308**.
# 
# ---
# 
# #### **Depth 3-6 (Leaf Nodes)**:
# At deeper levels, the tree continues to split until it achieves complete homogeneity or reaches the maximum depth. For example:
# - **Leaf Node (Entropy = 0)**: Indicates pure nodes, where all samples belong to one class.
# - **Class Predictions**:
#   - Each leaf node represents a prediction (`Yes` or `No`), based on the majority class within that node.
# 
# ---
# 
# ### **How the Tree Makes Predictions**
# 1. A sample is passed through the tree, starting from the root.
# 2. At each internal node:
#    - The sample's feature value is compared with the threshold (e.g., `Humidity <= 0.287`).
#    - Depending on the result, the sample is directed to the left or right child.
# 3. This process continues until the sample reaches a leaf node.
# 4. The predicted class is the majority class of the samples in that leaf.
# 
# 

# %%
import numpy as np
import pandas as pd



def explain_prediction(clf, sample, feature_names):
    tree = clf.tree_
    
    print(f"Decision path for the sample: {sample}")
    print("Step-by-step explanation of the prediction:")
    
    node = 0  
    while tree.children_left[node] != tree.children_right[node]:  
        feature_index = tree.feature[node]
        threshold = tree.threshold[node]
        feature_name = feature_names[feature_index]
        
        # Make the decision
        if sample[feature_index] <= threshold:
            print(f"At node {node}, feature '{feature_name}' <= {threshold:.2f} (Sample value: {sample[feature_index]:.2f})")
            node = tree.children_left[node]  # Go to the left child
        else:
            print(f"At node {node}, feature '{feature_name}' > {threshold:.2f} (Sample value: {sample[feature_index]:.2f})")
            node = tree.children_right[node]  # Go to the right child

    predicted_class = np.argmax(tree.value[node])  # Majority class in leaf node
    print(f"Predicted class: {label_encoder.classes_[predicted_class]}")


sample = X.iloc[0].values  
explain_prediction(clf, sample, X.columns)
print("_________________________________")
sample = X.iloc[912].values  
explain_prediction(clf, sample, X.columns)


# %% [markdown]
# #### Feature Importance Across Desion Tree

# %%
import numpy as np
from sklearn.tree import export_text

tree_text = export_text(clf, feature_names=list(X_train.columns))
print(tree_text) 

importances = clf.feature_importances_
features = X_train.columns
plt.figure(figsize=(10, 5))
sns.barplot(x=features, y=importances, palette="viridis")
plt.title("Feature Importance Across Decision Tree")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

# %% [markdown]
# ### **3.Performance Metrics Report**
# 

# %% [markdown]
# #### Knn using Sckit learn with different 5 k

# %%
for k in range(3, 12, 2):
    y_pred_knn = Knn(X_train, X_test, y_train, y_test, k)  
    print_classification_report(f"Knn_using_built_in_while_k_is_{k}", y_test, y_pred_knn)  
    plot_confusion_matrix(f"Knn_using_built_in_while_k_is_{k}", y_test, y_pred_knn)  


# %% [markdown]
# ### Summary Table:
# 
# | **k Value** | **Accuracy** | **Precision (No Rain)** | **Precision (Rain)** | **Recall (No Rain)** | **Recall (Rain)** | **F1-Score (Rain)** | **False Positives (No Rain)** | **False Negatives (Rain)** |
# |-------------|--------------|-------------------------|----------------------|----------------------|-------------------|---------------------|-----------------------------|----------------------------|
# | **3**       | 0.9700       | 0.97                    | 0.88                 | 0.99                 | 0.84              | 0.86                | 4                           | 9                          |
# | **5**       | 0.9680       | 0.97                    | 0.92                 | 0.99                 | 0.79              | 0.83                | 4                           | 12                         |
# | **7**       | 0.9660       | 0.97                    | 0.90                 | 0.99                 | 0.80              | 0.84                | 5                           | 11                         |
# | **9**       | 0.9700       | 0.97                    | 0.90                 | 0.99                 | 0.82              | 0.86                | 5                           | 10                         |
# | **11**      | 0.9720       | 0.97                    | 0.90                 | 0.99                 | 0.84              | 0.87                | 7                           | 9                          |
# 
# ---
# 
# ### Key Insights:
# - **Best accuracy**: \( k = 11 \) with **0.9720**.
# - **Best precision (Rain)**: \( k = 5 \) with **0.92**.
# - **Best recall (Rain)**: \( k = 11 \) with **0.84**.
# - **Best F1-score (Rain)**: \( k = 11 \) with **0.87**.
# - **Best false negatives**: \( k = 11 \) with **9**.
# - **Best false positives (No Rain)**: \( k = 5 \) with **4**. 
# 
# 

# %% [markdown]
# ### Accuracy Comparison Plot

# %%
import matplotlib.pyplot as plt

k_values = [3, 5, 7, 9, 11]
accuracies = [0.9700, 0.9680, 0.9660, 0.9700, 0.9720]

best_accuracy = max(accuracies)
best_k = k_values[accuracies.index(best_accuracy)]

plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', label="Accuracy")
plt.axhline(y=best_accuracy, color='r', linestyle='--', label=f"Best Accuracy = {best_accuracy:.3f}")
plt.scatter([best_k], [best_accuracy], color='red', zorder=5)  

plt.title("Accuracy vs. K-values for KNN", fontsize=14)
plt.xlabel("K-values", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(k_values)  
plt.legend(loc="lower right")
plt.grid(True)

plt.annotate(f"Best k = {best_k}", (best_k, best_accuracy),
             textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='red')

plt.tight_layout()
plt.show()


# %% [markdown]
# #### Knn from Scratch with different 5 k

# %%
for k in range(3,12,2):
    knn_model = initialize_knn(k)
    fit_knn(knn_model, X_train, np.array(y_train).ravel())
    y_pred_knn_from_Scratch = predict_knn(knn_model, X_test)
    accuracy = accuracy_score(y_test, y_pred_knn_from_Scratch)
    print(f'Accuracy of the KNN model from scratch on the test set while_k_is_{k}: {accuracy:.4f}')
    print_classification_report(f"Knn model From Scratch while_k_is_{k}",y_test,y_pred_knn_from_Scratch)
    plot_confusion_matrix(f"Knn model From Scratch while_k_is_{k}",y_test,y_pred_knn_from_Scratch)
    


# %% [markdown]
# 
# | **k Value** | **Accuracy** | **Precision (No Rain)** | **Precision (Rain)** | **Recall (No Rain)** | **Recall (Rain)** | **F1-Score (Rain)** | **False Positives (No Rain)** | **False Negatives (Rain)** |
# |-------------|--------------|-------------------------|----------------------|----------------------|-------------------|---------------------|-----------------------------|----------------------------|
# | **3**       | 0.9700       | 0.97                    | 0.89                 | 0.99                 | 0.84              | 0.86                | 6                           | 9                          |
# | **5**       | 0.9680       | 0.97                    | 0.92                 | 0.99                 | 0.79              | 0.83                | 4                           | 12                         |
# | **7**       | 0.9660       | 0.97                    | 0.88                 | 0.99                 | 0.80              | 0.84                | 6                           | 11                         |
# | **9**       | 0.9700       | 0.97                    | 0.90                 | 0.99                 | 0.82              | 0.86                | 5                           | 10                         |
# | **11**      | 0.9720       | 0.97                    | 0.90                 | 0.99                 | 0.84              | 0.87                | 5                           | 9                          |
# 
# ---
# 
# ### Key Insights:
# 
# * **Best accuracy**: \(k=11\) with **0.9720**.
# * **Best precision (Rain)**: \(k=5\) with **0.92**.
# * **Best recall (Rain)**: \(k=11\) with **0.84**.
# * **Best F1-score (Rain)**: \(k=11\) with **0.87**.
# * **Best false negatives**: \(k=11\) with **9**.
# * **Best false positives (No Rain)**: \(k=5\) with **4**.

# %%
import matplotlib.pyplot as plt

k_values = [3, 5, 7, 9, 11]
accuracies = [0.9700, 0.9680, 0.9660, 0.9700, 0.9720]
best_accuracy = max(accuracies)
best_k = k_values[accuracies.index(best_accuracy)]

plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', label="Accuracy")
plt.axhline(y=best_accuracy, color='r', linestyle='--', label=f"Best Accuracy = {best_accuracy:.3f}")
plt.scatter([best_k], [best_accuracy], color='red', zorder=5)  

plt.title("Accuracy vs. K-values for KNN", fontsize=14)
plt.xlabel("K-values", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(k_values)  
plt.legend(loc="lower right")
plt.grid(True)

plt.annotate(f"Best k = {best_k}", (best_k, best_accuracy),
             textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='red')

plt.tight_layout()
plt.show()



# %% [markdown]
# Comaprison Between Knn buikt in and Knn implemented From Scratch
# ### **1. Accuracy:**
# - Both results report the **same accuracy** for each \(k\)-value. 
#     - The highest accuracy is consistently observed at \(k=11\) with **0.9720**.
# 
# ### **2. Precision (Rain):**
# - The **precision for Rain** is the same in both results:
#   - \(k=5\) has the highest precision at **0.92**.
# 
# ### **3. Recall (Rain):**
# - Both results show the **same recall for Rain**:
#   - The highest recall for Rain is observed at \(k=11\) with **0.84**.
# 
# ### **4. F1-Score (Rain):**
# - Both results report the **same F1-score for Rain**:
#   - The highest F1-score for Rain is at \(k=11\) with **0.87**.
# 
# ### **5. False Positives (No Rain):**
# - The **false positives** for No Rain are identical across both results:
#   - \(k=5\) has the lowest false positives at **4**.
# 
# ### **6. False Negatives (Rain):**
# - Both results show the **same false negatives for Rain**:
#   - \(k=11\) has the lowest false negatives at **9**.
# 
# ---
# There are **no differences** in the analysis, metrics, or key insights between the two results. The values and the interpretation are **identical** in both tables.
# 
# 


