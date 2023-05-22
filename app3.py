import streamlit as st
import pandas as pd

columns_test = ['unit_ID','time_in_cycles','setting_1', 'setting_2','setting_3','T2','T24','T30','T50','P2','P15','P30','Nf','Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]
first_feature_to_select = "unit_ID"
second_features_list = [x for x in columns_test if x not in ["unit_ID", "time_in_cycles"]]
soglia = 20

import numpy as np
import os
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import keras.backend as K
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import plotly.express as px


soglia = 20

def colors(val, soglia):
    if val != "serie storica fornita non sufficientemente lunga" and float(val) < soglia:
        return "color: red"
    else:
        return "color: black"

def r2_keras(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

sensor_cols = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
       'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
       'PCNfR_dmd', 'W31', 'W32']
sequence_cols = ['setting_1', 'setting_2', 'setting_3', 'cycle_norm']
sequence_cols.extend(sensor_cols)
model_path = "model_lstm.h5"
sequence_length = 50

# Carica lo scaler da file pickle
with open('min_max_scaler.pkl', 'rb') as file:
    min_max_scaler = pickle.load(file)  


def preprocessing(dataset):
    dataset = dataset.drop(columns=[26,27], axis=1)
    dataset.columns = columns_test
    #normalizzazione
    dataset['cycle_norm'] = dataset['time_in_cycles']
    cols_normalize_2 = dataset.columns.difference(['unit_ID','time_in_cycles','RUL'])
    norm_test_df = pd.DataFrame(min_max_scaler.transform(dataset[cols_normalize_2]), 
                            columns=cols_normalize_2, 
                            index=dataset.index)
    test_join_df = dataset[dataset.columns.difference(cols_normalize_2)].join(norm_test_df)
    dataset = test_join_df.reindex(columns = dataset.columns)
    dataset = dataset.reset_index(drop=True)
    
    return dataset

 def fare_previsioni(dataset):
    dataset = preprocessing(dataset)
    
    #calcolo unit id corrispondenti alle serie troppo corte
    serie_troppo_corte = dataset[["unit_ID", "time_in_cycles"]].groupby("unit_ID").count()
    serie_troppo_corte = list(serie_troppo_corte[serie_troppo_corte["time_in_cycles"] < 50].index)

    #calcolo unit_id rimanenti e quindi usati per le predizioni
    unit_id_predicted = list(set(dataset["unit_ID"]) - set(serie_troppo_corte)) 
    
    # We pick the last sequence for each id in the test data
    seq_array_test_last = [dataset[dataset['unit_ID']==id][sequence_cols].values[-sequence_length:] 
                          for id in dataset['unit_ID'].unique() if len(dataset[dataset['unit_ID']==id]) >= sequence_length]
    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

    estimator = keras.models.load_model(model_path,custom_objects={'r2_keras': r2_keras})
    y_pred_test = estimator.predict(seq_array_test_last)
    
    #creazione dataframe per risultati
    results = pd.DataFrame({"unit_ID": unit_id_predicted, "previsioni": y_pred_test.flatten()})

    #aggiunta dati non calcolati
    for s in serie_troppo_corte:
        results.loc[len(results)] = [s, "serie storica fornita non sufficientemente lunga"]

    results = results.sort_values(by = "unit_ID").reset_index(drop = True)
    
    results.index = previsioni["unit_ID"]
    
    results.drop(columns = "unit_ID", inplace = True)

    return results
  
  
def scarica_csv(dataframe):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="previsioni.csv">Scarica file CSV</a>'
    return href
  
def interactive_chart_creator(dataset):
    cnt_train = dataset[[0,1]].groupby(0).max().sort_values(by=1, ascending=False)
    cnt_ind = [str(i) for i in cnt_train.index.to_list()]
    cnt_val = list(cnt_train[1].values)

    plt.figure(figsize=(12, 30))
    sns.barplot(x=list(cnt_val), y=list(cnt_ind), palette='Spectral') #controllare casting
    plt.xlabel('Numbero di cicli')
    plt.ylabel('Id unità')
    plt.title('Numero di cicli per unità', fontweight='bold', fontsize=24, pad=15)
    return plt


def elaboratore_previsioni(previsioni):
    return previsioni.style.applymap(colors)
  
  



st.title("Forecasting app")

file = st.file_uploader("Carica il dataset", type=["txt"])

if file is not None:
    # Leggi il file CSV in un DataFrame pandas
    dataset = pd.read_csv(file, sep=" ", header=None)
    
    # Mostra il dataset
    st.subheader("Dataset caricato")
    st.write(dataset)

    # Visualizza grafico
    st.subheader("Ispeziona il dataset")
    
    selected_unit_id = st.selectbox("Select {}".format(first_feature_to_select), dataset[columns_test.index(first_feature_to_select)].unique())
    st.subheader(f"Filtered DataFrame (Unit ID {selected_unit_id})")
    
    selected_column = st.selectbox("Select feature", second_features_list)
    
    interactive_chart = interactive_chart_creator(dataset)
    st.plotly_chart(interactive_chart)
    
    
    # Visualizza altri grafici
    if st.button("Visualizza altri grafici"):
        bar_plot = bar_plot_creator(dataset)
        st.pyplot(bar_plot)
        
     
# Esegui previsioni sul dataset caricato
    if st.button("Fai previsioni"):
        previsioni = fare_previsioni(dataset)
        
        # Mostra le previsioni
        st.subheader("Previsioni (soglia di allerta fissata a {})".format(soglia))
        st.dataframe(elaboratore_previsioni(previsioni), width = 500)
        
        # Bottone per scaricare il dataset delle previsioni
        st.markdown(scarica_csv(previsioni), unsafe_allow_html=True)
