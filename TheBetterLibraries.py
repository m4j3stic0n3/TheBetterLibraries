import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import tempfile
import os
import json
from datetime import datetime, timedelta

# ------------------------------------------------
# 1. FMP & Gemini Configuration
# ------------------------------------------------
FMP_API_KEY = "" 

# For Gemini
GOOGLE_API_KEY = "" 
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = 'gemini-2.0-flash'
gen_model = genai.GenerativeModel(MODEL_NAME)

# ------------------------------------------------
# 2. Streamlit Page Config
# ------------------------------------------------
st.set_page_config(layout="wide")
st.title("Stock Analysis Tool")
st.sidebar.header("Settings")

# ------------------------------------------------
# 3. Sidebar Inputs
# ------------------------------------------------
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated):", "AAPL,MSFT,GOOG")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

# Technical indicators
st.sidebar.subheader("Technical Indicators")
indicators = st.sidebar.multiselect(
    "Select Indicators:",
    ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
    default=["20-Day SMA"]
)

# ------------------------------------------------
# 4. Fetch Data from FMP
# ------------------------------------------------
def fetch_fmp_data(ticker, start_dt, end_dt, apikey=FMP_API_KEY):
    """
    Fetch daily historical data from FMP for a given ticker between start_dt and end_dt.
    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume (sorted ascending).
    """
    # Format dates as YYYY-MM-DD
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    # FMP endpoint for full historical data
    # Example: https://financialmodelingprep.com/api/v3/historical-price-full/AAPL?from=2022-01-01&to=2023-01-01&apikey=demo
    url = (
        f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
        f"?from={start_str}&to={end_str}&apikey={apikey}"
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data_json = resp.json()
    except Exception as e:
        st.warning(f"Error fetching data from FMP for {ticker}: {e}")
        return pd.DataFrame()

    # The JSON typically has: { "symbol": "AAPL", "historical": [ {date, open, high, low, close, volume}, ...] }
    historical = data_json.get("historical", [])
    if not historical:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(historical)
    # Standardize column names to match typical OHLC
    df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }, inplace=True)

    # Convert to datetime, numeric
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort ascending by date
    df.sort_values("Date", inplace=True)

    # Reindex by Date for convenience
    df.set_index("Date", inplace=True)

    return df

# ------------------------------------------------
# 5. Data Fetch Button
# ------------------------------------------------
if st.sidebar.button("Fetch Data"):
    stock_data = {}
    for ticker in tickers:
        df = fetch_fmp_data(ticker, start_date, end_date)
        if not df.empty:
            stock_data[ticker] = df
        else:
            st.warning(f"No data found for {ticker} (FMP returned empty).")

    st.session_state["stock_data"] = stock_data
    if stock_data:
        st.success("Stock data loaded successfully for: " + ", ".join(stock_data.keys()))
    else:
        st.warning("No stock data loaded. Check tickers, date range, or FMP usage.")

# ------------------------------------------------
# 6. Analysis + Chart
# ------------------------------------------------
if "stock_data" in st.session_state and st.session_state["stock_data"]:
    def analyze_ticker(ticker, data):
        # Build candlestick chart
        fig = go.Figure([
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Candlestick"
            )
        ])

        # Apply local indicators
        def add_indicator(ind):
            if ind == "20-Day SMA":
                sma = data["Close"].rolling(window=20).mean()
                fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
            elif ind == "20-Day EMA":
                ema = data["Close"].ewm(span=20).mean()
                fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
            elif ind == "20-Day Bollinger Bands":
                sma = data["Close"].rolling(window=20).mean()
                std = data["Close"].rolling(window=20).std()
                bb_upper = sma + 2 * std
                bb_lower = sma - 2 * std
                fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
                fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
            elif ind == "VWAP":
                # Basic VWAP calculation
                # cumulative (close*volume) / cumulative volume
                data["VWAP"] = (data["Close"] * data["Volume"]).cumsum() / data["Volume"].cumsum()
                fig.add_trace(go.Scatter(x=data.index, y=data["VWAP"], mode='lines', name='VWAP'))

        for indicator in indicators:
            add_indicator(indicator)

        fig.update_layout(xaxis_rangeslider_visible=False)

        # Convert the chart to an image (PNG) for Gemini
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.write_image(tmpfile.name)
            tmpfile_path = tmpfile.name

        with open(tmpfile_path, "rb") as f:
            image_bytes = f.read()
        os.remove(tmpfile_path)

        # Prepare Gemini prompt
        image_part = {"data": image_bytes, "mime_type": "image/png"}
        analysis_prompt = (
            f"You are a Stock Trader specializing in Technical Analysis at a top financial institution. "
            f"Analyze the stock chart for {ticker} based on its candlestick chart and the displayed technical indicators. "
            f"Provide a detailed justification of your analysis, explaining what patterns, signals, and trends you observe. "
            f"Then, based solely on the chart, provide a recommendation from the following options: "
            f"'Strong Buy', 'Buy', 'Weak Buy', 'Hold', 'Weak Sell', 'Sell', or 'Strong Sell'. "
            f"Return your output as a JSON object with two keys: 'action' and 'justification'."
        )

        # Call Gemini
        contents = [
            {"role": "user", "parts": [analysis_prompt]},
            {"role": "user", "parts": [image_part]}
        ]
        response = gen_model.generate_content(contents=contents)

        # Parse JSON response
        try:
            result_text = response.text
            json_start_index = result_text.find('{')
            json_end_index = result_text.rfind('}') + 1
            if json_start_index != -1 and json_end_index > json_start_index:
                json_string = result_text[json_start_index:json_end_index]
                result = json.loads(json_string)
            else:
                raise ValueError("No valid JSON object found in the response")
        except json.JSONDecodeError as e:
            result = {"action": "Error", "justification": f"JSON Parsing error: {e}. Raw text: {response.text}"}
        except ValueError as ve:
            result = {"action": "Error", "justification": f"Value Error: {ve}. Raw text: {response.text}"}
        except Exception as e:
            result = {"action": "Error", "justification": f"General Error: {e}. Raw text: {response.text}"}

        return fig, result

    # Create tabs
    tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)

    overall_results = []

    # For each ticker, do analysis
    for i, ticker in enumerate(st.session_state["stock_data"]):
        data = st.session_state["stock_data"][ticker]
        fig, result = analyze_ticker(ticker, data)
        overall_results.append({"Stock": ticker, "Recommendation": result.get("action", "N/A")})

        with tabs[i + 1]:
            st.subheader(f"Analysis for {ticker}")
            st.plotly_chart(fig)
            st.write("**Detailed Justification:**")
            st.write(result.get("justification", "No justification provided."))

    # Overall summary tab
    with tabs[0]:
        st.subheader("Overall Structured Recommendations")
        df_summary = pd.DataFrame(overall_results)
        st.table(df_summary)

else:
    st.info("Please fetch stock data using the sidebar.")
