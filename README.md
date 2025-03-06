import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Custom indicator functions
def calculate_moving_averages(data, short_window, long_window):
    data['SMA'] = data['Close'].rolling(window=short_window).mean()
    data['LMA'] = data['Close'].rolling(window=long_window).mean()
    return data

def calculate_rsi(data, period=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['MACD'] = data['Close'].ewm(span=short_window, adjust=False).mean() - data['Close'].ewm(span=long_window, adjust=False).mean()
    data['Signal Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

def calculate_atr(data, period=14):
    data['TR'] = np.maximum((data['High'] - data['Low']), 
                            np.maximum(abs(data['High'] - data['Close'].shift(1)), 
                                      abs(data['Low'] - data['Close'].shift(1))))
    data['ATR'] = data['TR'].rolling(window=period).mean()
    return data

def custom_indicator(data):
    data = calculate_moving_averages(data, short_window=50, long_window=200)
    data = calculate_rsi(data)
    data = calculate_macd(data)
    data = calculate_atr(data)
    
    # NaN değerleri temizle ve kopya oluştur
    data = data.dropna().copy()
    
    # Uzun pozisyon sinyalleri
    data['Buy Signal'] = (
        (data['SMA'] > data['LMA']) & 
        ((data['RSI'] < 50) | (data['MACD'] > data['Signal Line']))
    )
    data['Sell Signal'] = (
        (data['SMA'] < data['LMA']) & 
        ((data['RSI'] > 40) | (data['MACD'] < data['Signal Line']))
    )
    
    # Kısa pozisyon sinyalleri
    data['Short Buy Signal'] = (
        (data['SMA'] < data['LMA']) & 
        ((data['RSI'] > 60) | (data['MACD'] < data['Signal Line']))
    )
    data['Short Sell Signal'] = (
        (data['SMA'] > data['LMA']) & 
        ((data['RSI'] < 40) | (data['MACD'] > data['Signal Line']))
    )
    
    # Sinyal üretimini kontrol et
    print(f"Toplam veri satırı: {len(data)}")
    print(f"Uzun Alım sinyali sayısı: {data['Buy Signal'].sum()}")
    print(f"Uzun Satım sinyali sayısı: {data['Sell Signal'].sum()}")
    print(f"Kısa Alım sinyali sayısı: {data['Short Buy Signal'].sum()}")
    print(f"Kısa Satım sinyali sayısı: {data['Short Sell Signal'].sum()}")
    
    return data

# Download historical data for BTC-USD
data = yf.download('BTC-USD', start='2020-03-01', end='2025-03-01')
data.columns = [col[0] for col in data.columns]

# Apply custom indicator
data = custom_indicator(data)

# Backtest the strategy with advanced logic
initial_balance = 10000
balance = initial_balance
positions = []  # Tüm işlemleri sakla
open_long_positions = []  # Uzun pozisyonlar
open_short_positions = []  # Kısa pozisyonlar
transaction_fee = 0.998  # %0.2 işlem ücreti

for i in range(len(data)):
    current_price = data['Close'].iloc[i]
    atr = data['ATR'].iloc[i]
    
    # Pozisyon büyüklüğü: Bakiyenin %10’u
    position_size = min(balance * 0.1 / current_price, 0.5)  # Maksimum 0.5 BTC limiti
    
    # Uzun pozisyon açma
    if (pd.notna(data['Buy Signal'].iloc[i]) and 
        data['Buy Signal'].iloc[i] and 
        balance > 0 and 
        len(open_long_positions) < 10):  # Maksimum 10 pozisyon
        btc_amount = position_size * transaction_fee
        balance -= btc_amount * current_price
        stop_loss = current_price - (2 * atr)
        take_profit = current_price + (4 * atr)
        open_long_positions.append({
            'entry_price': current_price,
            'btc_amount': btc_amount,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': stop_loss,
            'date': data.index[i]
        })
        positions.append(('Buy', data.index[i], current_price, btc_amount))

    # Kısa pozisyon açma
    if (pd.notna(data['Short Buy Signal'].iloc[i]) and 
        data['Short Buy Signal'].iloc[i] and 
        balance > 0 and 
        len(open_short_positions) < 10):
        btc_amount = position_size * transaction_fee
        balance -= btc_amount * current_price
        stop_loss = current_price + (2 * atr)
        take_profit = current_price - (4 * atr)
        open_short_positions.append({
            'entry_price': current_price,
            'btc_amount': btc_amount,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': stop_loss,
            'date': data.index[i]
        })
        positions.append(('Short Buy', data.index[i], current_price, btc_amount))

    # Uzun pozisyon kapama ve trailing stop
    for pos in open_long_positions[:]:
        entry_price = pos['entry_price']
        btc_amount = pos['btc_amount']
        profit_pct = (current_price - entry_price) / entry_price
        
        # Trailing stop güncellemesi
        if profit_pct > 0.2:  # %20 kârda stop-loss’u giriş fiyatına çek
            pos['trailing_stop'] = max(pos['trailing_stop'], entry_price)
        if profit_pct > 0.4:  # %40 kârda stop-loss’u %20 kâr seviyesine çek
            pos['trailing_stop'] = max(pos['trailing_stop'], entry_price * 1.2)

        if (current_price <= pos['trailing_stop'] or 
            current_price >= pos['take_profit'] or 
            (pd.notna(data['Sell Signal'].iloc[i]) and data['Sell Signal'].iloc[i])):
            proceeds = btc_amount * current_price * transaction_fee
            balance += proceeds
            open_long_positions.remove(pos)
            positions.append(('Sell (SL/TP)' if not data['Sell Signal'].iloc[i] else 'Sell', 
                            data.index[i], current_price, proceeds))

    # Kısa pozisyon kapama ve trailing stop
    for pos in open_short_positions[:]:
        entry_price = pos['entry_price']
        btc_amount = pos['btc_amount']
        profit_pct = (entry_price - current_price) / entry_price
        
        # Trailing stop güncellemesi
        if profit_pct > 0.2:
            pos['trailing_stop'] = min(pos['trailing_stop'], entry_price)
        if profit_pct > 0.4:
            pos['trailing_stop'] = min(pos['trailing_stop'], entry_price * 0.8)

        if (current_price >= pos['trailing_stop'] or 
            current_price <= pos['take_profit'] or 
            (pd.notna(data['Short Sell Signal'].iloc[i]) and data['Short Sell Signal'].iloc[i])):
            proceeds = btc_amount * (entry_price * 2 - current_price) * transaction_fee  # Kısa pozisyon kârı
            balance += proceeds
            open_short_positions.remove(pos)
            positions.append(('Short Sell (SL/TP)' if not data['Short Sell Signal'].iloc[i] else 'Short Sell', 
                            data.index[i], current_price, proceeds))

# Final balance calculation
for pos in open_long_positions:
    proceeds = pos['btc_amount'] * data['Close'].iloc[-1] * transaction_fee
    balance += proceeds
    positions.append(('Sell (Final)', data.index[-1], data['Close'].iloc[-1], proceeds))
for pos in open_short_positions:
    proceeds = pos['btc_amount'] * (pos['entry_price'] * 2 - data['Close'].iloc[-1]) * transaction_fee
    balance += proceeds
    positions.append(('Short Sell (Final)', data.index[-1], data['Close'].iloc[-1], proceeds))

final_balance = balance
profit = final_balance - initial_balance
performance = (profit / initial_balance) * 100

# Output the results
print(f"Initial Balance: ${initial_balance:.2f}")
print(f"Final Balance: ${final_balance:.2f}")
print(f"Profit: ${profit:.2f}")
print(f"Performance: {performance:.2f}%")
print(f"Toplam işlem sayısı: {len(positions) // 2}")

# İşlem detaylarını yazdır
print("\nİşlem Detayları:")
for pos in positions:
    print(pos)

# Plot the buy/sell signals on the price chart
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='BTC-USD')
plt.scatter(data[data['Buy Signal']].index, data[data['Buy Signal']]['Close'], label='Buy Signal', marker='^', color='g', alpha=1)
plt.scatter(data[data['Sell Signal']].index, data[data['Sell Signal']]['Close'], label='Sell Signal', marker='v', color='r', alpha=1)
plt.scatter(data[data['Short Buy Signal']].index, data[data['Short Buy Signal']]['Close'], label='Short Buy Signal', marker='^', color='b', alpha=1)
plt.scatter(data[data['Short Sell Signal']].index, data[data['Short Sell Signal']]['Close'], label='Short Sell Signal', marker='v', color='y', alpha=1)
plt.title('BTC-USD Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
