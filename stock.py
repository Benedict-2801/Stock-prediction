import streamlit as st
import pandas as pd
import time
import math
from datetime import datetime
import numpy as np
import plotly.express as px
import yfinance as yf
import google.generativeai as genai

import login as log
import contact_us
import dash
import neccessity as ns
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout

graph_dict = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

def logout():
    # Clear session state variables
    for key in st.session_state.keys():
        del st.session_state[key]
    # Redirect to login page
    st.experimental_set_query_params(page="home")
    st.experimental_rerun()

def main():
    query_params = st.experimental_get_query_params()
    page = query_params.get("page", ["home"])[0]

    if page == "home":
        main_content()
    elif page=="Logout":
        logout()
    elif page == "contact":
        contact_us.content()

def main_content():
    genai.configure(api_key="AIzaSyAogD00Y22cHkhSEL4IzsHVAKNdzHg3NcI")
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    _, navbar = st.columns(2)
    with _:
        st.markdown("Welcome")
    with navbar:
        sign_up_col, contact_col = st.columns(2)
        with sign_up_col:
            if st.button("Log out"):
                st.experimental_set_query_params(page="Logout")
                st.experimental_rerun()
        with contact_col:
            if st.button("Contact Us"):
                st.experimental_set_query_params(page="contact")
                st.experimental_rerun()

    try:
        def toggle_sidebar():
            if 'sidebar_visible' not in st.session_state:
                st.session_state.sidebar_visible = True
            st.session_state.sidebar_visible = not st.session_state.sidebar_visible

        if 'sidebar_visible' not in st.session_state or st.session_state.sidebar_visible:
            with st.sidebar:
                initialize_session_state()

                st.session_state.navbar = st.selectbox("Select", ["dashboard", "data analysis and prediction"])
                
                chat_bot, base = st.columns(2)
                with chat_bot:
                    if st.button("Chat"):
                        if  not st.session_state.chatbot_active:
                           st.session_state.chatbot_active = True
                        else:
                           st.session_state.chatbot_active = False
                        
                if st.session_state.chatbot_active:
                    ns.chat_bot(model)

                if not st.session_state.chatbot_active:
                    handle_data_input()

        if st.session_state.navbar == "dashboard":
            dash.dashboard_content()
        else:
            handle_analysis_and_prediction(model)
    except Exception as e:
        st.write(e)

def initialize_session_state():
    if 'chatbot_active' not in st.session_state:
        st.session_state.chatbot_active = False

    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'stock_symbol' not in st.session_state:
        st.session_state.stock_symbol = ""
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

def handle_data_input():
    selection = st.selectbox("Select the data input type:", ["download from internet", "upload files"])
    try:
        if selection == "download from internet":
            st.session_state.stock_symbol = st.text_input("Give a stock symbol:", st.session_state.stock_symbol)
            if st.session_state.stock_symbol:
                end_date = datetime.today().date()
                data = yf.download(st.session_state.stock_symbol, start="1900-01-01", end=end_date)
        else:
            st.session_state.uploaded_file = st.file_uploader("Upload a file", type=["csv"], key='file_uploader')
            if st.session_state.uploaded_file is not None:
                data = pd.read_csv(st.session_state.uploaded_file)
    except ValueError as e:
        st.write("No files uploaded yet")
    try:
        if st.session_state.data is None or not st.session_state.data.equals(data):
            st.session_state.data = pd.DataFrame(data)
        if st.session_state.data is not None:
            st.write("Original Data:")
            st.write(st.session_state.data.head(2000))
            st.session_state.data = ns.preprocess(st.session_state.data)
    except Exception as e:
        print()

def handle_analysis_and_prediction(model):
    data = st.session_state.data
    stock_symbol = st.session_state.stock_symbol
    try:
        if 'data' in locals():
            prompt = ns.get_prompt(data.columns)
            response = model.generate_content(prompt)
            response_text = response.text

            content = parse_response(response_text)

            if 'Date' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
                data['Date'] = data.index

            predicted_data = content.get('predicted_data', '').replace('_', ' ')
            graphs = {key: value for key, value in content.items() if key != 'predicted_data'}
            
            x_axis, y_axis = select_axes(content)

            fig, graph_type = ns.plot_graph(data, graphs, x_axis, y_axis)
            st.plotly_chart(fig)
            graph_dict[graph_type] = fig      
            st.markdown(f"Data size: {data.shape}")
            st.write(data.head(2000))
            train_and_predict(data, predicted_data)
    except Exception as e:
        st.spinner("please wait")
   
def parse_response(response_text):
    content = {}
    cnt = 0
    entries = response_text.strip().split("\\")
    for entry in entries:
        elements = entry.strip().split(",")
        if len(elements) == 1:
            content["predicted_data"] = elements[0].strip()
        else:
            content[f"graph{cnt}"] = [e.strip() for e in elements]
            cnt += 1
    return content

def select_axes(content):
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("x_axis", set([i[0] for i in content.values() if i != 'predicted_data' and len(i[0]) > 1]))
    with col2:
        y_axis = st.selectbox("y_axis", set([i[1] for i in content.values() if i != 'predicted_data' and len(i[1]) > 1]))
    return x_axis, y_axis

def train_and_predict(data, predicted_data):
    
    
    # Set epochs and batch size
    epochs = st.slider("epochs", 0, 100, 10)
    batch_size = st.slider("batch size", 0, 128, 2)

    # Filter and prepare data
    df = data.filter([predicted_data])
    dataset = df.values

    # Define the length of training data
    training_data_len = math.ceil(len(dataset) * 0.8)

    # Check dataset shape
    if data.shape[0] != len(data):
        st.error("The dataset has an incorrect shape for processing.")
    else:
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Prepare training data
        train_data = scaled_data[:training_data_len]
        x_train, y_train = [], []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        st.write("x_train shape:", x_train.shape)

        # Build the LSTM model
        if 'model' not in st.session_state or not st.session_state.data.equals(data):
            model = Sequential()
            model.add(Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1))))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
            st.session_state.data = data
            st.write("Model trained and cached with the new dataset.")
        else:
            st.write("Model loaded from cache.")

        # Prepare testing data
    test_data = scaled_data[training_data_len - 60:]
    x_test, y_test = [], dataset[training_data_len:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    st.write("Testing...")

    # Get the model's predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate RMSE
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    st.write("RMSE:", rmse)

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    st.line_chart({
        'Train': train[predicted_data],
        'Validation': valid[predicted_data],
        'Predictions': valid['Predictions']
    })


        
if __name__ == "__main__":
    if 'authentication_status' not in st.session_state:
        st.session_state.authentication_status = False
        
    user = log.login_signup()
    if user is not None and st.session_state.authentication_status:
        st.session_state["username"] = user
        main()
    else:
        st.write("Invalid credentials")
