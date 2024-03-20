# La idea seria dividir 60-20-20 del total y guardarlo en un unico csv 

#%% Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

#%% Load and concatenate data into a single df
path = 'Data/raw/'
clarin_paro = pd.read_csv(path + 'clarin-paro.csv')
infobae_paro = pd.read_csv(path + 'infobae-paro.csv')
lanacion_paro = pd.read_csv(path + 'lanacion-paro.csv')

infobae_noparo = pd.read_csv(path + 'infobae-noparo.csv')
clarin_noparo = pd.read_csv(path + 'clarin-noparo.csv')
lanacion_noparo = pd.read_csv(path + 'lanacion-noparo.csv')

# Sample from the non-strike related news no more than twice the size of strike related news
clarin_noparo = clarin_noparo.sample(n=2*len(clarin_paro), random_state=1)
infobae_noparo = infobae_noparo.sample(n=2*len(infobae_paro), random_state=1)
lanacion_noparo = lanacion_noparo.sample(n=2*len(lanacion_paro), random_state=1)

combined_df = pd.concat([clarin_noparo, clarin_paro,
                         infobae_noparo, infobae_paro,
                         lanacion_noparo, lanacion_paro])

combined_df['stratify_col'] = combined_df['relacionado'].astype(str) + "_" + combined_df['medio'].astype(str)

#%% Split the data 
train_df, temp_df = train_test_split(combined_df, test_size=0.4, stratify=combined_df['stratify_col'])
validation_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['stratify_col'])

path = 'Data/'
validation_df.to_csv(path + 'validation.csv', index=False)
test_df.to_csv(path + 'test.csv', index=False)

#%% Save the base dataset 
train_df.to_csv('Data/' + 'basic_train.csv', index=False)

positive_class_df = train_df[train_df['relacionado'] == 1]
negative_class_df = train_df[train_df['relacionado'] == 0]

# 1206 402 804
print(len(train_df), len(positive_class_df), len(negative_class_df))

#%% Oversample data to balance relacionado classes
# 60% relacionadas a paro (40% pos, 20% neg), 30% no relacionadas
relacionado_pos = train_df[train_df['relacionado'] == 1]
relacionado_paro_pos = relacionado_pos[relacionado_pos['paro'] == 1]
relacionado_paro_neg = relacionado_pos[relacionado_pos['paro'] == 0]

tot_0 = len(train_df)
x_rel_pos_0 = len(relacionado_paro_pos)
x_rel_neg_0 = len(relacionado_paro_neg)
p_paro_pos = 0.4
p_paro_neg = 0.2

A = np.array([
    [1 - p_paro_pos, -1 * p_paro_pos],
    [-1 * p_paro_neg, 1 - p_paro_neg],
])

b = np.array([
    p_paro_pos * tot_0 - x_rel_pos_0,
    p_paro_neg * tot_0 - x_rel_neg_0,
])

amount_to_add_pos, amount_to_add_neg = np.linalg.solve(A, b)

df_pos_paro_oversampled = resample(relacionado_paro_pos,
                                   replace=True,
                                   n_samples=int(amount_to_add_pos),
                                   random_state=123)

df_neg_paro_oversampled = resample(relacionado_paro_neg,
                                   replace=True,
                                   n_samples=int(amount_to_add_neg),
                                   random_state=123)

oversample_train_df = pd.concat([df_pos_paro_oversampled,
                                 df_neg_paro_oversampled,
                                 train_df])

relacionado = oversample_train_df[oversample_train_df['relacionado'] == 1]
no_relacionado = oversample_train_df[oversample_train_df['relacionado'] == 0]
positivo = oversample_train_df[oversample_train_df['paro'] == 1]
negativo = oversample_train_df[oversample_train_df['paro'] == 0]

oversample_train_df.to_csv('Data/' + 'oversample_train.csv', index=False)

print(len(relacionado), len(no_relacionado), len(positivo), len(negativo))

# 1875 804 1340 1339

#%% Concatenate bootstrapped GPT news
# Balance dataset
gpt_dataset = pd.read_csv('Data/raw/' + 'chatgpt-balance.csv')

gpt_dataset['texto'] = gpt_dataset['texto'].replace({'"', ''})
gpt_train_df = pd.DataFrame()
rows_train_gpt = len(gpt_dataset)

gpt_train_df['texto'] = pd.concat([train_df['texto'], gpt_dataset['texto']], ignore_index=True)
gpt_train_df['paro'] = pd.concat([train_df['paro'], gpt_dataset['paro']], ignore_index=True)
gpt_train_df['medio'] = pd.concat([train_df['medio'], pd.Series(['' for i in range(rows_train_gpt)])], ignore_index=True)
gpt_train_df['link'] = pd.concat([train_df['link'], pd.Series(['' for i in range(rows_train_gpt)])], ignore_index=True)
gpt_train_df['relacionado'] = pd.concat([train_df['relacionado'], pd.Series([1 for i in range(rows_train_gpt)])], ignore_index=True)

print(len(gpt_train_df[gpt_train_df['relacionado'] == 1]))
print(len(gpt_train_df[gpt_train_df['relacionado'] == 0]))
print(len(gpt_train_df[gpt_train_df['paro'] == 1]))
print(len(gpt_train_df[gpt_train_df['paro'] == 0]))

gpt_train_df.to_csv('Data/' + 'balance_gpt_train.csv')

# El balance y GPT tienen proporciones aproximadamente iguales
# %%
