import os
import streamlit as st
import openai
import pandas as pd
import timesfm
from datetime import datetime, timedelta
import json

# Set JAX to use CPU
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Set OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

# Initialize TimesFM model
@st.cache_resource
def load_timesfm_model():
    tfm = timesfm.TimesFm(
        context_len=512,
        horizon_len=5,  # 5 future predictions
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend='cpu',
    )
    tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
    return tfm

tfm = load_timesfm_model()

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

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides market size data in JSON format."},
            {"role": "user", "content": prompt}
        ]
    )

    try:
        data = json.loads(response.choices[0].message.content)
        return pd.DataFrame(list(data.items()), columns=['Year', 'MarketSize'])
    except json.JSONDecodeError:
        st.error("Error: Could not parse the response from OpenAI. Please try again.")
        return None

def forecast_market_size(df):
    df['ds'] = pd.to_datetime(df['Year'], format='%Y')
    df['y'] = df['MarketSize']
    df['unique_id'] = ['market_size'] * len(df)

    forecast_df = tfm.forecast_on_df(
        inputs=df,
        freq="Y",
        value_name="y",
        num_jobs=-1,
    )

    return forecast_df

def plot_market_size(historical_df, forecast_df):
    # Combine historical and forecast data
    plot_df = pd.concat([
        historical_df.rename(columns={'Year': 'date', 'MarketSize': 'Historical'}),
        forecast_df[['ds', 'timesfm', 'timesfm-q-0.1', 'timesfm-q-0.9']].rename(columns={
            'ds': 'date', 'timesfm': 'Forecast', 'timesfm-q-0.1': 'Lower Bound', 'timesfm-q-0.9': 'Upper Bound'
        })
    ]).melt(id_vars=['date'], var_name='Series', value_name='Value')

    # Create the chart
    chart = st.line_chart(plot_df.set_index('date'), y='Value')

    return chart

def generate_summary(historical_df, forecast_df, input1, input2):
    prompt = f"""Based on the following historical and forecasted market size data for {input1} in the {input2} industry, 
    provide a brief summary of the market trends and future outlook. 
    Historical data: {historical_df.to_dict()}
    Forecast data: {forecast_df[['ds', 'timesfm']].to_dict()}
    Limit your response to 3-4 sentences."""

    response = openai.ChatCompletion.create(
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
            plot_market_size(historical_df, forecast_df)
            
            summary = generate_summary(historical_df, forecast_df, input1, input2)
            st.subheader("Summary")
            st.write(summary)

            st.subheader("Historical Data")
            st.dataframe(historical_df)

            st.subheader("Forecast Data")
            st.dataframe(forecast_df[['ds', 'timesfm', 'timesfm-q-0.1', 'timesfm-q-0.9']])

if __name__ == "__main__":
    main()
