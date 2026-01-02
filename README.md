# League of legends performance analyzer

This is a simple tool to help you analyze your league of legends match history. It reads a file containing your past games and turns that data into clear graphs and statistics. The goal is to help you spot patterns in your gameplay, like whether you play worse when you are tired or how your rank changes over time.

## What it does

* Tracks your rank: plots a timeline of your lp gains and losses so you can see your progress.
* Checks for fatigue: shows your win rate based on how many games you have already played that day (e.g., do you lose more on your 5th game?).
* Analyzes daily volume: calculates if you have a higher win rate on days where you play fewer games total.
* Predicts streaks: looks at your recent wins and losses to see if you are likely to win the next one based on momentum, based on Markov states.
* Cleans data: automatically fixes and organizes messy data files for you.
* It also can visualise all of the functionalities.
