import json
import asyncio
import websockets
import pandas as pd
import http.client
from datetime import datetime
from tzlocal import get_localzone
from data_fetcher import create_features

# Initialize DataFrame for list of data since starting
data_list_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Initialize Dataframe for current data
data_current_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

async def send_ping(websocket):
    while True:
        await asyncio.sleep(30)  # Wait for 30 seconds
        await websocket.ping()  # Send PING frame to the server

async def input_listener(stop_event): # functions to get user input for stopping live update
    while True:
        user_input = await asyncio.to_thread(input, "\nType 'x' to stop\n")
        if user_input == "x":
            stop_event.set()
            return
        
def get_last_23_candles(interval, symbol, window):

    conn = http.client.HTTPSConnection("api.datasource.cybotrade.rs")

    headers = { 'X-API-Key': "UxgWSqmUi1oxUCeOKnGt3xz02AjbZoHrJ65tMtV4baAXTS0p" }

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit" : window
    }

    topic = "/bybit-spot/candle?" + "&".join([f"{k}={v}" for k, v in params.items()])

    conn.request("GET", topic, headers=headers)

    res = conn.getresponse()
    data = res.read().decode("utf-8")

    # parce JSON
    json_data = json.loads(data)

    if 'data' in json_data:
        candles = json_data['data']

        # Convert to Pandas DataFrame
        df = pd.DataFrame(candles)

        # Rename 'start_time' to 'timestamp'
        df.rename(columns={'start_time': 'timestamp'}, inplace=True)

        # Convert timestamp to human-readable format
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Convert timestamp to datetime (UTC) and make it timezone-aware
        local_timezone = get_localzone()  # Detect system's timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(local_timezone).dt.tz_localize(None)

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        return df  # Return the DataFrame

async def live_1m_data(uri, stop_event, symbol):
    global data_current_df
    # connect to websocket
    async with websockets.connect(uri) as websocket:
    
        # Start the background task to send PING
        asyncio.create_task(send_ping(websocket))

        topic = "bybit-linear|candle?symbol=" + symbol + "&interval=1m"

        # send subcription request
        await websocket.send(json.dumps({
            "topics": [topic],
            "api_key": "UxgWSqmUi1oxUCeOKnGt3xz02AjbZoHrJ65tMtV4baAXTS0p"
        }))

        # receive and print live data from the server
        while not stop_event.is_set():
            try:
                message = await websocket.recv()
                data = json.loads(message) # parse JSON data

                # Check if 'data' key exists and contains a list
                if 'data' in data and isinstance(data['data'], list):

                    for entry in data['data']:  # Loop through each data entry
                        ts = datetime.fromtimestamp(entry.get('start_time', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S')
                        row = {
                            'timestamp': ts,
                            'open': entry.get('open', None),
                            'high': entry.get('high', None),
                            'low': entry.get('low', None),
                            'close': entry.get('close', None),
                            'volume': entry.get('volume', None)
                        }
                    
                    data_current_df = pd.DataFrame([row]).set_index("timestamp")

                    print(data_current_df)
            
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed.")
                break   

async def live_data_update(uri, stop_event, symbol, interval, window):
    global data_list_df
    
    # initialize the previous window data
    data_list_df = get_last_23_candles(interval=interval,symbol=symbol,window=(window - 1))
    
    # connect to websocket
    async with websockets.connect(uri) as websocket:
        # Start the background task to send PING
        asyncio.create_task(send_ping(websocket))

        topic = "bybit-linear|candle?symbol=" + symbol + "&interval=" + interval

        # send subcription request
        await websocket.send(json.dumps({
            "topics": [topic],
            "api_key": "UxgWSqmUi1oxUCeOKnGt3xz02AjbZoHrJ65tMtV4baAXTS0p"
        }))

        # receive and print live data from the server
        while not stop_event.is_set():
            try:
                message = await websocket.recv()
                data = json.loads(message) # parse JSON data

                # Check if 'data' key exists and contains a list
                if 'data' in data and isinstance(data['data'], list):
                    new_rows = [] # Store multiple rows if available

                    for entry in data['data']:  # Loop through each data entry
                        ts = datetime.fromtimestamp(entry.get('start_time', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S')
                        row = {
                            'timestamp': ts,
                            'open': entry.get('open', None),
                            'high': entry.get('high', None),
                            'low': entry.get('low', None),
                            'close': entry.get('close', None),
                            'volume': entry.get('volume', None)
                        }
                        new_rows.append(row)
                    
                    new_row_df = pd.DataFrame(new_rows).set_index("timestamp")
                    data_list_df = pd.concat([data_list_df, new_row_df]).tail(window)

                    #data_list_df = data_list_df[['open', 'high', 'low', 'close', 'volume']]

                    # Print the updated DataFrame
                    print(data_list_df)
            
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed.")
                break

async def main_collect_live_data(symbol, interval, window):
    global data_list_df, data_current_df
    uri = "wss://stream.flow.balaenaquant.com"  # WebSocket server URL
    stop_event = asyncio.Event()  # Define a stop event

    # run live data update as background task
    loop_live_data_update = asyncio.create_task(live_data_update(uri, stop_event, symbol, interval, window))
    loop_live_1mdata_update = asyncio.create_task(live_1m_data(uri, stop_event, symbol))

    await input_listener(stop_event)

    loop_live_data_update.cancel()
    loop_live_1mdata_update.cancel()
    try:
        await loop_live_data_update
    except asyncio.CancelledError:
        print("Live update stopped.")

# Run the main function
if __name__ == "__main__":

    asyncio.run(main_collect_live_data(symbol = "BTCUSDT", interval = "1m", window = 24))
    print("\nFinished.....\n")
    print(data_list_df)
    print(data_current_df)