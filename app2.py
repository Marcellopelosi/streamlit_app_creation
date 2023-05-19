import pandas as pd
import streamlit as st
import plotly.express as px

# Create sample DataFrame
data = {
    'Name': ['John', 'Jane', 'Alice', 'Bob'],
    'Age': [32, 28, 35, 41],
    'Party': ['A', 'B', 'A', 'B']
}
df = pd.DataFrame(data)

# Streamlit app
def main():
    st.title("Party Visualization")
    
    # Display DataFrame
    st.subheader("Data")
    st.dataframe(df)

    # Select party
    party_selection = st.selectbox("Select Party", df['Party'].unique())
    
    # Filter DataFrame based on party selection
    filtered_df = df[df['Party'] == party_selection]
    
    # Plotting
    st.subheader(f"Party {party_selection} Members")
    fig = px.bar(filtered_df, x='Name', y='Age', color='Name')
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()
