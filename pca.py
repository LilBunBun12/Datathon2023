import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.decomposition import PCA
import math

  
df = pd.read_csv('TD_HOSPITAL_TRAIN.csv')
feature_data = df[['bp', 'meals', 'death', 'temperature']]


def fillNumberBlanks(df):
    for cat in df:
      i = 0
      print(cat)
      if(cat == "sex"):
        for i in range(len(df["sex"])):
          if(df["sex"][i] == "1"):
            df["sex"][i] = 1.0
          elif(df["sex"][i].lower().find("f") < 0):
            df["sex"][i] = 1.0
          else:
            df["sex"][i] = 0.0
      elif("1234567890".find(str(df[cat][i])[0].lower()) >= 0):
        while(math.isnan(df[cat][i])):
          i += 1
        ar = df[cat].values.tolist()
        medianar = []
        for i in range(len(df[cat])):
          if(not(math.isnan(df[cat][i]))):
            medianar.append(df[cat][i])
        medianar.sort()
        median = medianar[len(medianar)//2]
        for i in range(len(df[cat])):
          if(math.isnan(df[cat][i])):
            df[cat][i] = median
        continue
  



def data_preprocessing(df):
    
    col_to_keep = ['temperature', 'bp', 'meals', 'timeknown', 'age', 'blood', 'bloodchem1', 'bloodchem2', 'psych1', 'glucose']
    df = df[col_to_keep]
    print(df["timeknown"])
    fillNumberBlanks(df)
    # removeBlanks(df,"temperature","temperature")
    # print(ar)
    # df.replace('', 0, inplace=True)
    # df.fillna(0, inplace=True)
    return df

cleaned_data = data_preprocessing(df)

# Perform PCA
pca = PCA(n_components=2)
pca.fit(feature_data)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
# print("Explained Variance Ratios:", explained_variance)


transformed_data = pca.transform(feature_data)

dead_group = feature_data.loc[feature_data['death'] == 1]
alive_group = feature_data.loc[feature_data['death'] == 0]
dead_group = dead_group[['bp', 'meals']]
alive_group = alive_group[['bp', 'meals']]



# Scatter plot of the original data
plt.scatter(alive_group.values[:, 0], alive_group.values[:, 1], c='b', alpha=0.5, s = .5)
plt.scatter(dead_group.values[:, 0], dead_group.values[:, 1], c='r', alpha=0.5, s = .5)
plt.axhline(0, color='k', linestyle='--', linewidth=1)  # Add a horizontal line at y=0
plt.axvline(0, color='k', linestyle='--', linewidth=1)  # Add a vertical line at x=0
plt.xlabel('bp')
plt.ylabel('meals')
plt.title('Original Data')
plt.show()



print(transformed_data)
bp = transformed_data[:,0]
meals = transformed_data[:,1]

deaths = feature_data['death']
temperature_new = [bp[x] for x in range(len(deaths)) if deaths[x] == 1]
meals_new = [meals[x] for x in range(len(deaths)) if deaths[x] == 1]
plt.scatter(temperature_new,  meals_new, c='r', alpha=0.5, s = 3)
temperature_new = [bp[x] for x in range(len(deaths)) if deaths[x] == 0]
meals_new = [meals[x] for x in range(len(deaths)) if deaths[x] == 0]
plt.scatter(temperature_new,  meals_new, c='b', alpha=0.5, s = 3)
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.axvline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel('meals')
plt.ylabel('bp')
plt.title('PCA-Transformed Data')
plt.show()

plt.bar(['bp', 'meals'], explained_variance, alpha=0.7, color=['blue', 'red'])
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Component')
plt.show()