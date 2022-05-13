import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def heat_map(df, figsize=(20,20)):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()
