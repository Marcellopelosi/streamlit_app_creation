import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Create sample DataFrame
data = {
    'Name': ['John', 'Jane', 'Alice', 'Bob'],
    'Age': [32, 28, 35, 41],
    'Income': [50000, 60000, 70000, 55000],
    'Expenses': [25000, 20000, 30000, 35000]
}
df = pd.DataFrame(data)

# Streamlit app
def main():
    st.title("Party Visualization")
    
    # Display DataFrame
    st.subheader("Data")
    st.dataframe(df)

    # Select party
    party_selection = st.selectbox("Select Party", df['Name'])
    
    # Filter DataFrame based on party selection
    filtered_df = df[df['Name'] == party_selection]
    
    # Display filtered DataFrame
    st.subheader(f"Filtered DataFrame (Party {party_selection})")
    st.dataframe(filtered_df)

    # Plotting line graph for each column
    for column in filtered_df.columns:
        if column != 'Name':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df['Name'], y=filtered_df[column], mode='lines', name=column))
            fig.update_layout(title=f"{column} over Names", xaxis_title="Name", yaxis_title=column)
            st.plotly_chart(fig)

if __name__ == '__main__':
    main()
