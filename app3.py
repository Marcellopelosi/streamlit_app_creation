import streamlit as st

columns_test = ['unit_ID','time_in_cycles','setting_1', 'setting_2','setting_3','T2','T24','T30','T50','P2','P15','P30','Nf',
                'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]
first_feature_to_select = "unit_ID"
second_features_list = [x for x in columns_test if x not in ["unit_ID", "time_in_cycles"]]
soglia = 20

_, _ = select_box_preparator(first_feature_to_select, second_feature_to_select)
interactive_chart = interactive_chart_creator(dataset)



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
