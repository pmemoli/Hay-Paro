#%%
import pandas as pd
import numpy as np

#%% proporcion clarin
df = pd.read_csv('./data/clarin-paro.csv')
df['paro'].mean()  # 72% 

#%% proporcion la nacion
df = pd.read_csv('./Data/raw/lanacion-paro.csv')
df['paro'].mean()  # 82% 

#%% proporcion infobae
df = pd.read_csv('./data/infobae-paro.csv')
df['paro'].mean()  # 77% 

#%% proporcion gpt
df = pd.read_csv('./Data/balance_gpt_train.csv')
df['paro'].mean()  # 40% 

#%% proporcion oversample
df = pd.read_csv('./Data/oversample_train.csv')
df['paro'].mean()  # 40% 

#%% proporcion train
df = pd.read_csv('./Data/basic_train.csv')
df['paro'].mean()  # 25% 

# Que quiero: que el dataset de oversample tenga ~50% clase positiva
# habiendo sampleado de paros relacionados. Lo mismo con gpt (van a
# terminar con mas datos relacionados que no).