import pandas as pd

from wordcloud import WordCloud
from matplotlib import pyplot as plt


recipes = pd.read_csv('data/processed_recipes_1.csv')
print(recipes.head())