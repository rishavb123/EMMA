import seaborn as sns
import pandas as pd
import json
import matplotlib.pyplot as plt


# Sample JSON data (replace with your actual file data)
filename = (
    "/Users/rbhagat/projects/EMMA/plotting/outputs/2024-05-11/14-06-30/results.json"
)

metric_name_to_plot = "convergence_efficiency_1"

run_id_mapping = {
    "disagreement_diayn_key_prediction_door_key_change_10_1715020025": "10 Skill POI Diayn",
    "disagreement_diayn_key_prediction_door_key_change_1714548168": "5 Skill POI Diayn",
    "disagreement_ir_emb_key_prediction_door_key_change_1714165501": "IR POI emb",
    "key_prediction_door_key_change_1714165830": "PPO Baseline",
    "warmstart_both_disagreement_diayn_10_skills_key_prediction_door_key_change_1715352928": "100k Warmstart POI Diayn",
}

aggregator = "iqm"

# Parse JSON data
with open(filename) as f:
    data_dict = json.load(f)

# Filter out runs with less than 4 data points
filtered_data = {}
for metric_name, metric_data in data_dict.items():
    filtered_data[metric_name] = {
        run_id_mapping[run_id]: value
        for run_id, value in metric_data.items()
        if data_dict[f"{metric_name_to_plot}_len"][run_id] >= 5
        and run_id in run_id_mapping
    }

# Convert data to pandas DataFrame
df = pd.DataFrame.from_dict(
    filtered_data[f"{metric_name_to_plot}_{aggregator}"],
    orient="index",
    columns=[f"{aggregator}"],
)

# Create bar chart using seaborn
sns.barplot(x=df.index, y=df[f"{aggregator}"])
sns.despine()  # Remove extra grid lines for cleaner visualization

# Customize the plot (optional)
plt.title(f"{aggregator.title()} {metric_name_to_plot.replace('_', ' ').title()}")
plt.xlabel("Run ID")
plt.ylabel(f"{aggregator.title()} {metric_name_to_plot.replace('_', ' ').title()}")
plt.show()  # To display the plot, uncomment this line

print(df)  # Print the DataFrame containing the means for valid runs
