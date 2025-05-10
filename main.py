
import numpy as np
import pandas as pd

movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

movies.head(1)

credits.head(1)

# Merging two df based on title
movies=movies.merge(credits,on='title')

movies.head()

movies.info()

# Keeping only the important col
# genres, ID, keywords,title, overview, cast, crew
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.head()

# Overview, genres, cast, crew will be merged and new col as tag will be created, final ds with 3 colunns as Movie_id, title and tags

# Missing values
movies.isnull().sum()

# drop null values
movies.dropna(inplace=True)

# Checking for duplicated values
movies.duplicated().sum()

movies.iloc[0].genres
# Its a list of dict so need to convert in proper format as ['Action', 'Adventure', 'Fantasy', 'Scifi']

import ast
def convert(obj):
  L=[]
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L

movies['genres']=movies['genres'].apply(convert)

movies['keywords']=movies['keywords'].apply(convert)

movies.head()

def counvert3(obj):
  L=[]
  counter=0
  for i in ast.literal_eval(obj):
    if counter!=3:
      L.append(i['name'])
      counter+=1
    else:
      break
  return L

movies['cast']=movies['cast'].apply(counvert3)

def fetch_director(obj):
  L=[]
  for i in ast.literal_eval(obj):
    if i['job']=='Director':
      L.append(i['name'])
      break
  return L

movies['crew']=movies['crew'].apply(fetch_director)

# overview to be converted into string
movies['overview']=movies['overview'].apply(lambda x:x.split())

movies.head(1)

# Now we need to remove the space bet words, here sam worthibgton shd be samworthington
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies.head(2)

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
movies.head(1)

new_df=movies[['movie_id','title','tags']]

new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))

new_df.head(2)

new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')

vectors=cv.fit_transform(new_df['tags']).toarray()

vectors

cv.get_feature_names_out()

# Need to do stemming('dance', 'dancing', output is dance )
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))

  return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)

# har movie ha other movie ke sath cosine dist.. which means similarity
# Using cosine similarity function
from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)

# Top 5 movies based on the cosine similarity
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

def recommend(movie):
  index = new_df[new_df['title'] == movie].index[0]
  distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
  for i in distances[1:6]:
    print(new_df.iloc[i[0]].title)

recommend('Avatar')

import pickle
pickle.dump(new_df,open('movies.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
# pickle.load(open('movies.pkl','rb'))          