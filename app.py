import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
np.float_ = np.float64
from prophet import Prophet
import plotly.graph_objects as go
import json

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai_api_key"])

def get_market_size_data(input1, input2):
    prompt = f"""Can you list the historic market size figures of {input1} in {input2} industry for the last five years?
    You can use market cap or revenue figures of {input1} or related companies in {input2} industry.
    Please provide the data in a JSON format with years as keys and market size as values. For example:
    {{
        "2018": 1000000000,
        "2019": 1100000000,
        "2020": 1200000000,
        "2021": 1300000000,
        "2022": 1400000000
    }}
    Only provide the JSON data, no additional text."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides market size data in JSON format."},
            {"role": "user", "content": prompt}
        ]
    )

    try:
        data = json.loads(response.choices[0].message.content)
        df = pd.DataFrame(list(data.items()), columns=['ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds'])
        return df
    except json.JSONDecodeError:
        st.error("Error: Could not parse the response from OpenAI. Please try again.")
        return None

def forecast_market_size(df):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=5, freq='Y')
    forecast = model.predict(future)
    return forecast

def plot_market_size(historical_df, forecast_df):
    fig = go.Figure()

    # Plot historical data
    fig.add_trace(go.Scatter(x=historical_df['ds'], y=historical_df['y'],
                             mode='markers+lines', name='Historical', line=dict(color='blue')))

    # Plot forecast
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'],
                             mode='lines', name='Forecast', line=dict(color='red')))

    # Add confidence interval
    fig.add_trace(go.Scatter(x=forecast_df['ds'].tolist() + forecast_df['ds'].tolist()[::-1],
                             y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'].tolist()[::-1],
                             fill='toself', fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'),
                             name='Confidence Interval'))

    fig.update_layout(title='Market Size Prediction',
                      xaxis_title='Year',
                      yaxis_title='Market Size')

    return fig

def generate_summary(historical_df, forecast_df, input1, input2):
    prompt = f"""Based on the following historical and forecasted market size data for {input1} in the {input2} industry, 
    provide a brief summary of the market trends and future outlook. 
    Historical data: {historical_df.to_dict()}
    Forecast data: {forecast_df[['ds', 'yhat']].to_dict()}
    Limit your response to 3-4 sentences."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides market analysis summaries."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Streamlit app
def main():
    st.title("Market Size Prediction App")

    input1 = st.text_input("Enter the product/service:", "robotics")
    input2 = st.text_input("Enter the industry:", "construction")

    if st.button("Predict Market Size"):
        historical_df = get_market_size_data(input1, input2)
        
        if historical_df is not None:
            forecast_df = forecast_market_size(historical_df)
            
            st.subheader("Market Size Prediction Chart")
            fig = plot_market_size(historical_df, forecast_df)
            st.plotly_chart(fig)
            
            summary = generate_summary(historical_df, forecast_df, input1, input2)
            st.subheader("Summary")
            st.write(summary)

            st.subheader("Historical Data")
            st.dataframe(historical_df)

            st.subheader("Forecast Data")
            st.dataframe(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

if __name__ == "__main__":
    main()
