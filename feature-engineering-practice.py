# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def feature_eng():
# %%
    bnkdat_df = pd.read_csv("banking_synthetic_data.csv")
    bnkdat_df.head()

    # %%
    #check missing values
    bnkdat_df.isnull().sum()
    # no missing values

    # %%
    all_mean_num_cols = bnkdat_df.select_dtypes(include='number').mean(skipna=False)
    all_mean_num_cols

    # %%
    # Numeric columns
    #handle missing values.
    bnkdat_df.fillna(all_mean_num_cols, inplace=True)
    bnkdat_df

    # %%
    bnkdat_df.select_dtypes(exclude='number') #only categorical columns

    # %%
    all_categorical_cols = bnkdat_df.select_dtypes(exclude='number').mode() # Categorical columns
    all_categorical_cols

    # %%
    # Numeric columns
    # Task 1: Data Cleaning and Preprocessing
    #handle missing values.
    bnkdat_df.fillna(all_categorical_cols, inplace=True)
    bnkdat_df

    # %%
    # 2. Convert Categorical Variables into numerical values.
    bnkdat_df['Sex'] = LabelEncoder().fit_transform(bnkdat_df['Sex'])  # Male=1, Female=0
    bnkdat_df['CustomerStatus'] = LabelEncoder().fit_transform(bnkdat_df['CustomerStatus'])  # Active=0, Inactive=1
    bnkdat_df['AccountType'] = LabelEncoder().fit_transform(bnkdat_df['AccountType']) 
    bnkdat_df

    # %%
    # 3. Normalize Numerical Features . performing feature scaling on selected numeric columns
    scaler = StandardScaler()
    bnkdat_df[['Age', 'AccountDuration', 'AverageMonthlySpending']] = scaler.fit_transform(
        bnkdat_df[['Age', 'AccountDuration', 'AverageMonthlySpending']])

    bnkdat_df

    # %% [markdown]
    # StandardScaler Formula
    # For each value x in a column:
    # z = \frac{x - \mu}{\sigma}
    # Where:
    # - \mu = mean of the column
    # - \sigma = standard deviation of the column
    # - z = standardized value
    # 

    # %%
    bnkdat_df.head()

    # %%
    bnkdat_df['InvestmentAccounts'] = LabelEncoder().fit_transform(bnkdat_df['InvestmentAccounts'])  # Male=1, Female=0
    bnkdat_df['FixedDeposits'] = LabelEncoder().fit_transform(bnkdat_df['FixedDeposits'])  # Active=0, Inactive=1
    bnkdat_df['MutualFunds'] = LabelEncoder().fit_transform(bnkdat_df['MutualFunds']) 
    bnkdat_df['StockInvestments'] = LabelEncoder().fit_transform(bnkdat_df['StockInvestments']) 
    bnkdat_df['BondInvestments'] = LabelEncoder().fit_transform(bnkdat_df['BondInvestments']) 
    bnkdat_df['TaxSavings'] = LabelEncoder().fit_transform(bnkdat_df['TaxSavings']) 
    bnkdat_df['TradingAccounts'] = LabelEncoder().fit_transform(bnkdat_df['TradingAccounts']) 
    bnkdat_df['SpecialtyFunds'] = LabelEncoder().fit_transform(bnkdat_df['SpecialtyFunds'])
    bnkdat_df['BalancedFunds'] = LabelEncoder().fit_transform(bnkdat_df['BalancedFunds'])
    bnkdat_df 

    # %%
    bnkdat_df

    # %%
    # 4. Feature Transformation
    bnkdat_df['TotalInvestments'] = bnkdat_df[['InvestmentAccounts', 'FixedDeposits', 'MutualFunds', 'StockInvestments',
                                'BondInvestments', 'BalancedFunds', 'TaxSavings',
                                'TradingAccounts', 'SpecialtyFunds']].sum(axis=1)
    bnkdat_df['TotalInvestments'] 
    # 4. Feature Transformation

    # %%
    # Task 2: Feature Engineering

    # 1. Create Interaction Features
    bnkdat_df['Age_AvgMonthlySpending'] = bnkdat_df['Age'] * bnkdat_df['AverageMonthlySpending']


    # %%
    # 2. Generate Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(bnkdat_df[['AccountType']])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['AccountType']))
    bnkdat_df = pd.concat([bnkdat_df, poly_df], axis=1)

    bnkdat_df
    #generate column AccountType^2

    # %%
    # 3. Feature Binning - feature binning — a technique to convert continuous numeric data into categorical buckets

    bnkdat_df['SpendingCategory'] = pd.cut(bnkdat_df['AverageMonthlySpending'], bins=[-np.inf, -0.5, 0.5, np.inf],
                                    labels=['Low', 'Medium', 'High'])

    # %% [markdown]
    # 🔹 pd.cut(...)
    # - This function bins continuous values into discrete intervals.
    # - It’s useful for creating categories like “Low”, “Medium”, “High” from a numeric column.
    # 🔹 bnkdat_df['AverageMonthlySpending']
    # - The numeric column being binned — likely already standardized (mean = 0, std = 1).
    # 🔹 bins=[-np.inf, -0.5, 0.5, np.inf]
    # np.inf - positive infinity
    # -np.inf - negative infinity
    # - Defines the edges of the bins:
    # - -np.inf to -0.5 → Low
    # - -0.5 to 0.5 → Medium
    # - 0.5 to np.inf → High
    # These thresholds assume the data is standardized. So values near 0 are average, below -0.5 are low, and above 0.5 are high.
    # 
    # 🔹 labels=['Low', 'Medium', 'High']
    # - Assigns category names to each bin.
    # 🔹 Assignment bnkdat_df['SpendingCategory'] = ...
    # - Creates a new column with the binned labels.
    # 

    # %%
    bnkdat_df

    # %%
    bnkdat_df[['Age', 'AccountDuration', 'AverageMonthlySpending', 'InvestmentAccounts',
                                    'FixedDeposits', 'MutualFunds', 'StockInvestments', 'BondInvestments',
                                    'BalancedFunds', 'TaxSavings', 'TradingAccounts',
                                    'SpecialtyFunds']]

    # %%
    # Task 3: Dimensionality Reduction: Number of components will depend on result. This is just an example

    # Apply PCA
    pca = PCA(n_components=2)
    pca

    # %%
    pca_result = pca.fit_transform(bnkdat_df[['Age', 'AccountDuration', 'AverageMonthlySpending', 'InvestmentAccounts',
                                    'FixedDeposits', 'MutualFunds', 'StockInvestments', 'BondInvestments',
                                    'BalancedFunds', 'TaxSavings', 'TradingAccounts',
                                    'SpecialtyFunds']])
    pca_result

    # %%
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    df_pca

    # %% [markdown]
    # Principal Component Analysis (PCA) — a powerful technique for simplifying high-dimensional data while preserving its structure
    # - You're creating a PCA object that will reduce your data to 2 principal components.
    # - These components are linear combinations of the original features that capture the maximum variance.
    # - You're applying PCA to a subset of columns:
    #   'Age', 'AccountDuration', 'AverageMonthlySpending', and various investment-related features.
    # 
    # - fit_transform does two things:
    #   - Fit: Learns the directions (principal components) that capture the most variance.
    #   - Transform: Projects the original data onto these directions.
    # 
    #   Think of it like rotating and flattening a cloud of points in 12D space down to 2D, while keeping the most meaningful structure
    # 
    # - You now have a new DataFrame with just two columns:
    #   - PC1: First principal component (captures most variance)
    #   - PC2: Second principal component (captures next most)
    # 
    # 🧠 Analogy: PCA as a Camera Angle
    # Imagine you're photographing a sculpture:
    # - From the front, you see the most detail (PC1).
    # - From the side, you get a different but still meaningful view (PC2).
    # - You ignore other angles that add little new information.
    # 
    # 

    # %%
    # Save the cleaned dataset and reports
    bnkdat_df.to_csv('cleaned_bank_data.csv', index=False)
    df_pca.to_csv('pca_results_bank_data.csv', index=False)

    print("Data cleaning, feature engineering, and dimensionality reduction completed. Results saved.")

    # %%


    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization')
    plt.grid(True)
    plt.show()
    plt.savefig('pca_visualization.png')


    # %%
    bnkdat_df[['Age', 'AccountDuration', 'AverageMonthlySpending', 'InvestmentAccounts',
                                    'FixedDeposits', 'MutualFunds', 'StockInvestments', 'BondInvestments',
                                    'BalancedFunds', 'TaxSavings', 'TradingAccounts',
                                    'SpecialtyFunds']]

    # %%
    # Apply t-SNE
    tsne = TSNE(n_components=2)
    tsne

    # %%
    tsne_results = tsne.fit_transform(bnkdat_df[['Age', 'AccountDuration', 'AverageMonthlySpending', 'InvestmentAccounts',
                                    'FixedDeposits', 'MutualFunds', 'StockInvestments', 'BondInvestments',
                                    'BalancedFunds', 'TaxSavings', 'TradingAccounts',
                                    'SpecialtyFunds']])

    tsne_results

    # %%
    tsne_df = pd.DataFrame(data=tsne_results, columns=['tsne-2d-one', 'tsne-2d-two'])
    tsne_df

    # %%
    tsne_df['CustomerStatus'] = bnkdat_df['CustomerStatus']
    tsne_df

    # %%
    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='tsne-2d-one', y='tsne-2d-two',
        hue='CustomerStatus',
        data=tsne_df,
        legend='full',
        alpha=0.7
    )

    plt.title('t-SNE Visualization of Customer Data')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig('tsne_visualization.png')


if __name__ == "__main__":
    feature_eng()
