import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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
    
    # Display filtered DataFrame
    st.subheader(f"Filtered DataFrame (Party {party_selection})")
    st.dataframe(filtered_df)

    # Plotting line graph for each column
    for column in filtered_df.columns:
        if column != 'Party' and column != 'Name':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df['Name'], y=filtered_df[column], mode='lines', name=column))
            fig.update_layout(title=f"{column} over Names", xaxis_title="Name", yaxis_title=column)
            st.plotly_chart(fig)

if __name__ == '__main__':
    main()
