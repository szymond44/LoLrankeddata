import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

class LoLPredictor:
    def __init__(self, filename, history_length):
        # Store file path and how many past games to analyze in Markov states
        self.filename = filename
        self.history_length = history_length
    
    def loading_and_cleaning(self):
        # Open and read the JSON file
        with open(self.filename) as f:
            content = f.read().strip()
            all_items = []
            decoder = json.JSONDecoder()
            pos = 0
            
            # Parse concatenated JSON objects manually
            while pos < len(content):
                while pos < len(content) and content[pos].isspace():
                    pos += 1
                if pos >= len(content):
                    break   
                obj, pos = decoder.raw_decode(content, pos)
                all_items.extend(obj.get('items', []))
            
        clean_data = []
        for match in all_items:
            outcome = match.get('result')
            ts = match.get('startedAt')
            
            # Robust extraction of LP rank
            try:
                rank = match['lp']['after']['value']
            except Exception: 
                rank = 0
            clean_data.append({'timestamp': ts, 'outcome': outcome, 'rank': rank})

        # Create DataFrame and convert types
        df = pd.DataFrame(clean_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Map WON/LOST strings to binary integers
        df['outcome'] = df['outcome'].map({'WON': 1, 'LOST': 0})
        
        # Filter out matches with invalid rank and sort chronologically
        df = df[df['rank'] != 0]
        df = df.sort_values(by='timestamp')
        
        # Extract date for grouping
        df['date'] = df['timestamp'].dt.date
        
        self.df = df
        return self.df
    
    def prepare_data(self):
        # Calculate daily aggregate stats
        daily_groups = self.df.groupby('date')
        daily_stats = daily_groups['outcome'].agg(['mean', 'count'])

        # Calculate game sequence number within each day
        self.df['game_nr'] = self.df.groupby('date').cumcount() + 1
        
        # Calculate stats based on game number (1st game of day, 2nd game, etc.)
        seq_groups = self.df.groupby(['game_nr'])
        seq_stats = seq_groups['outcome'].agg(['mean', 'count'])
        
        # Map daily totals back to main dataframe
        self.df['day_count'] = self.df['date'].map(daily_stats['count'])
        self.df['lp_diff'] = self.df['rank'].diff()
        
        # Create lag columns dynamically based on history_length
        lag_cols = []
        for idx in range(1, self.history_length + 1):   # It must start at 1 and not at 0
            self.df[f'outcome_lag_{idx}'] = self.df['outcome'].shift(idx)
            lag_cols.append(f'outcome_lag_{idx}')
        
        # Group by the sequence of past results to find win probabilities
        states_groups = self.df.groupby(lag_cols)
        win_probs = states_groups['outcome'].mean()   
        
        # Reshape for heatmap: Level 0 (Most recent lag) becomes columns
        if self.history_length > 1:
            matrix = win_probs.unstack(level=0)
        else:
            # Handle single-game history (Series -> DataFrame)
            matrix = pd.DataFrame(win_probs).T

        # Calculate win rate grouped by total volume of games in a day
        volume_groups = daily_stats.groupby('count')
        self.volume_stats = volume_groups['mean'].mean()

        # Save processed data to self
        self.seq_stats = seq_stats
        self.daily_stats = daily_stats
        self.win_probs = win_probs
        self.matrix = matrix

    def get_daily_stats(self, target_date=None):
        daily_groups = self.df.groupby('date')
        self.daily_stats = daily_groups['outcome'].agg(['mean', 'count'])
        
        # Map daily totals back to main dataframe (needed for volume stats)
        self.df['day_count'] = self.df['date'].map(self.daily_stats['count'])
        
        # Formatting for display
        display_df = self.daily_stats.copy()
        display_df['Win Rate %'] = (display_df['mean'] * 100).round(2)
        
        print("\n" + "="*40)
        if target_date:
            print(f"1. DAILY WIN RATE STATS (Date: {target_date})")
            print("="*40)
            try:
                query_date = pd.to_datetime(target_date).date()
                specific_day = display_df.loc[[query_date]]
                print(specific_day[['count', 'Win Rate %']])
            except (KeyError, ValueError):
                print(f"(!) No data found for date: {target_date}")
        else:
            print("1. DAILY WIN RATE STATS (All Time)")
            print("="*40)
            print(display_df[['count', 'Win Rate %']])
            
        return self.daily_stats

    def get_sequence_stats(self):
        self.df['game_nr'] = self.df.groupby('date').cumcount() + 1
        seq_groups = self.df.groupby(['game_nr'])
        self.seq_stats = seq_groups['outcome'].agg(['mean', 'count'])
        
        display_df = self.seq_stats.copy()
        display_df['Win Rate %'] = (display_df['mean'] * 100).round(2)
        
        print("\n" + "="*40)
        print("2. STATS BY GAME SEQUENCE NUMBER")
        print("="*40)
        print(display_df[['count', 'Win Rate %']])
        
        return self.seq_stats

    def get_heatmap_stats(self):
        lag_cols = []
        for idx in range(1, self.history_length + 1):
            self.df[f'outcome_lag_{idx}'] = self.df['outcome'].shift(idx)
            lag_cols.append(f'outcome_lag_{idx}')
        
        states_groups = self.df.groupby(lag_cols)
        win_probs = states_groups['outcome'].mean()   
        
        if self.history_length > 1:
            self.matrix = win_probs.unstack(level=0)
        else:
            self.matrix = pd.DataFrame(win_probs).T
            
        print("\n" + "="*40)
        print(f"3. SEQUENCE PROBABILITIES (History Length: {self.history_length})")
        print("="*40)
        print((self.matrix * 100).round(2).astype(str) + '%')
        
        return self.matrix

    def get_volume_stats(self):
        # Ensure daily stats are calculated first to get 'day_count'
        if 'day_count' not in self.df.columns:
            self.get_daily_stats(target_date=None) 
            
        daily_groups = self.df.groupby('date')
        daily_agg = daily_groups['outcome'].agg(['mean', 'count'])
        
        volume_groups = daily_agg.groupby('count')
        self.volume_stats = volume_groups['mean'].mean()
        
        vol_display = self.volume_stats.to_frame(name='Win Rate')
        vol_display['Win Rate %'] = (vol_display['Win Rate'] * 100).round(2)
        vol_display.index.name = "Total Games/Day"
        
        print("\n" + "="*40)
        print("4. WIN RATE BY DAILY VOLUME")
        print("="*40)
        print(vol_display[['Win Rate %']])
        
        return self.volume_stats
        
    def Overall_rank_plot(self):   
        # Figure 1: Rank history over time 
        plt.figure(1, figsize=(14, 7))
        # Plot main rank line
        plt.plot(self.df['timestamp'], self.df['rank'], color='#2c3e50', linewidth=2, label='Ranking LP', zorder=1)
        
        # Overlay scatter points for wins (green) and losses (red)
        wins = self.df[self.df['outcome'] == 1]
        losses = self.df[self.df['outcome'] == 0]
        plt.scatter(wins['timestamp'], wins['rank'], c='#27ae60', s=25, zorder=2, label='Win')
        plt.scatter(losses['timestamp'], losses['rank'], c='#c0392b', s=25, zorder=2, label='Loss')
        
        plt.title('LP rank over time', fontsize=14)
        plt.ylabel('Total LP')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.gcf().autofmt_xdate()

        plt.show()
    
    def daily_win_rate(self):    
        # Figure 2: Daily win rate with game count labels
        plt.figure(2, figsize=(14, 7))
        x_positions = range(len(self.daily_stats))
        
        # Plot bar chart
        bars = plt.bar(x_positions, self.daily_stats['mean'] * 100, 
                       color='#e67e22', alpha=0.8)
        
        plt.title('Daily win rate', fontsize=14)
        plt.ylabel('Win rate (%)')
        plt.xlabel('Date')
        plt.grid(axis='y', alpha=0.3)
        plt.axhline(50, color='gray', linestyle='--')
        
        # Force integer ticks on Y axis
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add text label (game count) above each bar
        for bar, count in zip(bars, self.daily_stats['count']):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 2, int(count), 
                    ha='center', fontsize=10, fontweight='bold')
        
        # Reduce density of x-axis date labels
        plt.xticks(x_positions[::max(1, len(self.daily_stats)//20)], 
                  [str(date) for date in self.daily_stats.index][::max(1, len(self.daily_stats)//20)], 
                  rotation=45, ha='right')
        plt.tight_layout()

        plt.show()

    def win_rate_by_game_number(self):
        # Figure 3: Win rate by game number in session
        plt.figure(3, figsize=(14, 7))
        bars = plt.bar(self.seq_stats.index, self.seq_stats['mean'] * 100, 
                       color='#3498db', alpha=0.8)
        
        plt.axhline(50, color='gray', linestyle='--')
        plt.title('Win rate by game number in day', fontsize=14)
        plt.xlabel('Game number in session')
        plt.ylabel('Win rate (%)')
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage labels above bars
        for x, y in zip(self.seq_stats.index, self.seq_stats['mean'] * 100):
            plt.text(x, y + 2, f"{y:.0f}%", ha='center', fontsize=10, fontweight='bold')
        
        # Add sample size (n=X) inside bottom of bars
        for bar, count in zip(bars, self.seq_stats['count']):
            plt.text(bar.get_x() + bar.get_width()/2, 2, f"n={int(count)}", 
                    ha='center', fontsize=9, color='black')
        plt.show()

    def sequence_heatmap(self):
        # Figure 4: Sequence heatmap
        plt.figure(figsize=(12, 8))
        
        plot_matrix = self.matrix.copy()
        
        # Rename columns for readability (newest game result)
        plot_matrix.columns = ['Recent: Loss', 'Recent: Win']
        
        # Rename rows to represent sequence history as string strings (L-W-L)
        new_index = []
        for index_value in plot_matrix.index:
            if self.history_length == 1:
                label = "No prior history"
            elif isinstance(index_value, tuple):
                label = "-".join(['W' if x==1.0 else 'L' for x in index_value])
            else:
                label = 'W' if index_value==1.0 else 'L'
            new_index.append(label)
            
        plot_matrix.index = new_index
        
        sns.heatmap(plot_matrix, annot=True, cmap='RdYlGn', fmt=".2f", linewidths=.5)
        
        plt.title(f"Win probability (x-axis = most recent game)")
        plt.xlabel("Result of 1 game ago")
        plt.ylabel(f"History before that (2 games ago ... {self.history_length} games ago)")
        plt.yticks(rotation=0)

        plt.show()

    def win_rate_by_daily_volume(self):    
        # Figure 5: Win rate by daily volume
        plt.figure(figsize=(8, 5))
        
        # Plot average win rate vs total games played that day
        plt.bar(self.volume_stats.index, self.volume_stats, color='purple', alpha=0.7)
        
        plt.title("Do i play better when i play less?")
        plt.xlabel("Total games played in one day")
        plt.ylabel("Average win rate")
        plt.axhline(0.5, color='red', linestyle='--', label='50% win rate') 
        plt.legend()
        
        plt.show()

    def visualising_data(self):
        # Wrapper to execute all plotting functions
        self.Overall_rank_plot()
        self.Daily_Win_rate()
        self.Win_rate_by_game_number()
        self.Sequence_heatmap()

# Execution Block
filename = 'C:/Users/szymo/Desktop/folder/zadaniapython/riwia.json'
player = LoLPredictor(filename, 1)        
player.loading_and_cleaning()
player.prepare_data()
player.get_heatmap_stats()
player.get_sequence_stats()
player.get_volume_stats()
player.win_rate_by_daily_volume()
