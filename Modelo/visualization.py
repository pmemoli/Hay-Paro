#%% load data
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

sns.set_theme()

path = 'train_graphs/'

with open(path + 'resultado_simple.pkl', 'rb') as fp:
    data_simple = pickle.load(fp)

with open(path + 'resultado_lstm_gpt_oversample.pkl', 'rb') as fp:
    data_lstm_gpt_oversample = pickle.load(fp)

with open(path + 'resultado_lstm_basic.pkl', 'rb') as fp:
    data_lstm_basic = pickle.load(fp)

with open(path + 'resultado_lstm_continuacion.pkl', 'rb') as fp:
    data_lstm_continuacion = pickle.load(fp)

data = {**data_simple}
data['lstm'] = {**data_lstm_gpt_oversample['lstm'], **data_lstm_basic['lstm']}

for key in data['lstm']:
    data['lstm'][key] = data['lstm'][key] + data_lstm_continuacion['lstm'][key]

#%% visualize
def f1(precision, recall):
    return 2 * precision * recall/ (precision + recall)

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].set_xlim(0, 70)
ax[0].set_ylim(0.6, 0.95)
ax[1].set_ylim(0.6, 0.95)
ax[1].set_xlim(0, 1100)

i = 0
for model_name, results in data.items():
    ax[i].title.set_text(model_name)
    ax[i].set_xlabel('Epoch')
    ax[i].set_ylabel('F1 score')

    for dataset_name, results in data[model_name].items():
        f1_score = [f1(result['precision'], result['recall']) if result != None
                    else 0 for result in results]
        precision = [result['precision'] if result != None
                    else 0 for result in results]
        recall = [result['recall'] if result != None
                    else 0 for result in results]
        accuracy = [result['accuracy'] if result != None
                    else 0 for result in results]

        if dataset_name == 'gpt':
            dataset_name = 'gpt (synthetic data)'

        ax[i].plot(accuracy, label=dataset_name)
        ax[i].legend()

    i += 1

# fig.savefig(f'train graph')

'''
El modelo entrenado con el dataset de sobresampleo presenta el mejor rendimiento
y parece entrenarse en su punto justo con menos epochs. Sorprendentemente, 
el modelo entrenado con los datasets de chatgpt y el basico terminan dando
aproximadamente igual. 

Para el dataset de sobresampleo la lstm da considerablemente mejor que un modelo simple
de dos layers, mientras que para GPT y BASIC dan lo mismo.

mas variados y ver como es el desempeño cuando testeamos.
Para afinar el testing, podemos armar un dataset de 100 titulos escritos a mano
'''

# %%
