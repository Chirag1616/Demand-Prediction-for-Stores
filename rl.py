import pandas as pd
import numpy as np
import pandas as pd
import joblib
from itertools import product

# -------------------------------
# Load trained RF model
# -------------------------------
rf_model = joblib.load("best_rf_model.pkl")

# -------------------------------
# Define actions
# -------------------------------
actions = [0, 1, 2]  # decrease, same, increase

# -------------------------------
# Discretize state features
# -------------------------------
sales_bins = np.linspace(0, 20000, 20)  # sales_lag_1
week_bins = np.linspace(1, 52, 13)      # quarter of year
holiday_bins = [0, 1]
spike_bins = [0, 1]

# Create a mapping from multi-feature tuple -> Q-table index
state_space = list(product(range(len(sales_bins)),
                           holiday_bins,
                           spike_bins, spike_bins))  # sales_lag_bin, holiday, spike1, spike2
q_table = np.zeros((len(state_space), len(actions)))

# Function to map multi-feature state to index
def get_state_idx(state):
    sales_idx = np.digitize(state['sales_lag_1'], sales_bins) - 1
    holiday_idx = state['Holiday_Flag']
    spike1_idx = state['Spike1_Flag']
    spike2_idx = state['Spike2_Flag']
    return state_space.index((sales_idx, holiday_idx, spike1_idx, spike2_idx))

# -------------------------------
# Step function
# -------------------------------
def step(state, action):
    # Compute derived features
    state['sales_roll_mean_3'] = np.mean([state['sales_lag_3'], state['sales_lag_2'], state['sales_lag_1']])

    # Prepare features for RF
    rf_features = ['sales_lag_1', 'sales_roll_mean_3', 'week_sin', 'week_cos',
                   'month_sin', 'month_cos', 'Yearly_Trend', 'Holiday_Flag',
                   'Spike1_Flag', 'Spike2_Flag']
    df_week = pd.DataFrame([{k: state[k] for k in rf_features}])

    predicted_demand = rf_model.predict(df_week)[0]

    # Reward function
    if action == 2:
        reward = predicted_demand - state['sales_lag_1']
    elif action == 0:
        reward = state['sales_lag_1'] - predicted_demand
    else:
        reward = -abs(predicted_demand - state['sales_lag_1'])
    reward = np.clip(reward, -20000, 20000)

    # Update next state lags
    next_state = state.copy()
    next_state['sales_lag_3'] = state['sales_lag_2']
    next_state['sales_lag_2'] = state['sales_lag_1']
    next_state['sales_lag_1'] = predicted_demand

    return next_state, reward, predicted_demand

# -------------------------------
# Training loop
# -------------------------------
alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 2000

# Example initial state
state = {
    'sales_lag_1': 10000,
    'sales_lag_2': 9500,
    'sales_lag_3': 9000,
    'week_sin': np.sin(2*np.pi*37/52),
    'week_cos': np.cos(2*np.pi*37/52),
    'month_sin': np.sin(2*np.pi*9/12),
    'month_cos': np.cos(2*np.pi*9/12),
    'Yearly_Trend': 0.5,
    'Holiday_Flag': 0,
    'Spike1_Flag': 0,
    'Spike2_Flag': 0
}

# Store evaluation results
eval_results = []

for ep in range(episodes):
    current_state = state.copy()
    state_idx = get_state_idx(current_state)

    # Epsilon-greedy
    if np.random.rand() < epsilon:
        action_idx = np.random.choice(len(actions))
    else:
        action_idx = np.argmax(q_table[state_idx])

    # Take step
    next_state, reward, predicted_demand = step(current_state, actions[action_idx])

    # Update Q-table
    next_state_idx = get_state_idx(next_state)
    q_table[state_idx, action_idx] += alpha * (
        reward + gamma * np.max(q_table[next_state_idx]) - q_table[state_idx, action_idx]
    )

    # Store evaluation info
    eval_results.append({
        'episode': ep,
        'state_sales_lag_1': current_state['sales_lag_1'],
        'action': actions[action_idx],
        'predicted_demand': predicted_demand
    })

    current_state = next_state.copy()

# -------------------------------
# Use trained policy
# -------------------------------
state_idx = get_state_idx(state)
best_action_idx = np.argmax(q_table[state_idx])
best_action = actions[best_action_idx]

if best_action == 2:
    print("✅ Recommended action: Increase inventory — demand likely to rise")
elif best_action == 0:
    print("⚠️ Recommended action: Decrease inventory — demand may drop")
else:
    print("ℹ️ Recommended action: Keep inventory same — demand stable")
