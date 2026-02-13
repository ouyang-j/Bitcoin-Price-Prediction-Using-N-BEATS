from openbb import obb  # new OpenBB SDK import 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

class CryptoBacktester: 
    def __init__(self, assets, start='2022-01-01'):
        self.assets = assets 
        self.data = pd.concat([obb.crypto.load(symbol=coin, start_date=start)['Close'].rename(coin) for coin in assets], axis=1) 
        self.returns = self.data.pct_change().dropna() 
        self.portfolio = pd.DataFrame(index=self.data.index)
        
    def run_strategy(self, signal_fn): 
        weights = signal_fn(self.returns) 
        self.portfolio['strategy_return'] = (weights.shift(1) * self.returns).sum(axis=1) 
        self.portfolio['cumulative_return'] = (1 + self.portfolio['strategy_return']).cumprod()
        
    def evaluate(self): 
        perf = self.portfolio['strategy_return'] 
        sharpe = perf.mean() / perf.std() * np.sqrt(252) 
        print(f"Sharpe: {sharpe:.2f}, Total Return:{self.portfolio['cumulative_return'].iloc[-1] - 1:.2%}")
        
    def plot(self): 
        self.portfolio['cumulative_return'].plot(title="Cumulative Strategy Return") 
        plt.grid(True)
        plt.show()
