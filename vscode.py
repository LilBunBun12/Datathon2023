# import matplotlib.pyplot as plt 
import pandas as pd
# from sklearn.decomposition import PCA


  
df = pd.read_csv('TD_HOSPITAL_TRAIN.csv')
# feature_data = df[['meals', 'blood']]

def removeBlanks(df,cat1,cat2):
  ar = df[cat1].tolist()
  ar2 = df[cat2].tolist()
  for i in range(len(ar) - 1,-1,-1):
    if(ar[i] == None or ar2[i] == None):
      df.drop(i)
  



def data_preprocessing(df):
    
    col_to_keep = ['death', 'age', 'blood', 'reflex', 'bloodchem1', 'bloodchem2', 'psych1', 'glucose']
    df = df[col_to_keep]
    # removeBlanks(df,"blood","meals")
    ar = df["blood"].tolist()
    print(ar)
    df.replace('', 0, inplace=True)
    df.fillna(0, inplace=True)
    return df

cleaned_data = data_preprocessing(df)

# Perform PCA
# pca = PCA(n_components=2)
# pca.fit(feature_data)

# Explained variance ratio
# explained_variance = pca.explained_variance_ratio_
# print("Explained Variance Ratios:", explained_variance)


# transformed_data = pca.transform(feature_data)

# Scatter plot of the original data
# plt.scatter(feature_data.values[:, 0], feature_data.values[:, 1], c='b', alpha=0.5)
# plt.axhline(0, color='k', linestyle='--', linewidth=1)  # Add a horizontal line at y=0
# plt.axvline(0, color='k', linestyle='--', linewidth=1)  # Add a vertical line at x=0
# plt.xlabel('meals')
# plt.ylabel('blood')
# plt.title('Original Data')
# plt.show()


# plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c='r', alpha=0.5)
# plt.axhline(0, color='k', linestyle='--', linewidth=1)
# plt.axvline(0, color='k', linestyle='--', linewidth=1)
# plt.xlabel('meals')
# plt.ylabel('blood')
# plt.title('PCA-Transformed Data')
# plt.show()

# plt.bar(['blood', 'meals'], explained_variance, alpha=0.7, color=['blue', 'red'])
# plt.xlabel('Principal Component')
# plt.ylabel('Explained Variance Ratio')
# plt.title('Explained Variance by Principal Component')
# plt.show()