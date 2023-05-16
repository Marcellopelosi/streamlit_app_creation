import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Carica il modello da file pickle
with open('model.pkl', 'rb') as file:
    modello = pickle.load(file)

# Carica lo scaler da file pickle
with open('min_max_scaler.pkl', 'rb') as file:
    min_max_scaler = pickle.load(file)    
    
def preprocessing(dataset):
    dataset = dataset.drop(columns=[26,27], axis=1)
    columns_test = ['unit_ID','time_in_cycles','setting_1', 'setting_2','setting_3','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]
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

# Funzione per fare previsioni sul dataset caricato
def fare_previsioni(dataset):
    dataset = preprocessing(dataset)
    
    previsioni = modello.predict(dataset)
    return previsioni

# Configurazione dell'applicazione Streamlit
st.title("Forecasting app")

# Caricamento del dataset
file = st.file_uploader("Carica il dataset", type=["txt"])

if file is not None:
    # Leggi il file CSV in un DataFrame pandas
    dataset = pd.read_csv(file, sep=" ", header=None)
    

    # Mostra il dataset
    st.subheader("Dataset caricato")
    st.write(dataset)

    # Esegui previsioni sul dataset caricato
    if st.button("Fai previsioni"):
        previsioni = fare_previsioni(dataset)

        # Mostra le previsioni
        st.subheader("Previsioni")
        st.write(previsioni)
