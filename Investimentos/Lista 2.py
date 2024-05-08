import pandas as  pd

# Q1 - Single Index model

# Input data
data = [
    {'Security': 'A', 'Beta': 0.8, 'Expected Excess Return': 0.1, 'sigma e': 0.25},
    {'Security': 'B', 'Beta': 1.0, 'Expected Excess Return': 0.12, 'sigma e': 0.1},
    {'Security': 'C', 'Beta': 1.2, 'Expected Excess Return': 0.14, 'sigma e': 0.2}
]

sigma_e_m = 0.2

# Calculate security return variance
for d in data:
    d['sigma'] = round(d['Beta']**2 * sigma_e_m**2 + d['sigma e'], 6)

# Create DataFrame to display results
df = pd.DataFrame(data)
df.set_index('Security', inplace=True)
print(df)