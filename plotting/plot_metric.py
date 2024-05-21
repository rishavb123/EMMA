import seaborn as sns
import pandas as pd
import json
import matplotlib.pyplot as plt


# Sample JSON data (replace with your actual file data)
# filename = "/Users/rbhagat/projects/EMMA/plotting/outputs/kp_orig/results.json"
# run_id_mapping = {
#     "disagreement_diayn_key_prediction_door_key_change_10_1715020025": "POI Diayn",
#     # "disagreement_diayn_key_prediction_door_key_change_1714548168": "5 Skill POI Diayn",
#     "disagreement_ir_emb_key_prediction_door_key_change_1714165501": "POI IR w/ POI emb",
#     "disagreement_diayn_key_prediction_door_key_change_1714165692": "Diayn Baseline",
#     # "warmstart_both_disagreement_diayn_10_skills_key_prediction_door_key_change_1715352928": "100k Warmstart POI Diayn",
#     "key_prediction_door_key_change_1714165830": "PPO Baseline",
# }

# filename = "/Users/rbhagat/projects/EMMA/plotting/outputs/kp_repeat/results.json"
# run_id_mapping = {
#     "disagreement_diayn_10_skills_key_prediction_door_key_change_1715354304": "POI Diayn",
#     # "disagreement_diayn_10_30_uniform_skills_key_prediction_door_key_change_1715440148": "10 Skill POI Diayn, 0.3 uniform",
#     "disagreement_ir_emb_key_prediction_door_key_change_1715354338": "POI IR w/ POI emb",
#     "disagreement_diayn_10_100_uniform_skills_key_prediction_door_key_change_1715440497": "Diayn Baseline",
#     "key_prediction_door_key_change_1715353432": "PPO Baseline",
# }

filename = "/Users/rbhagat/projects/EMMA/plotting/outputs/kp_repeat8/results.json"
run_id_mapping = {
    "disagreement_diayn_10_skills_key_prediction_door_key_change_1715735913": "POI Diayn",
    "disagreement_ir_emb_key_prediction_door_key_change_1715735695": "POI IR w/ POI emb",
    "traditional_diayn_10_skills_key_prediction_door_key_change_1715886603": "Diayn Baseline",
    "key_prediction_door_key_change_1715735670": "PPO Baseline",
}

metric_name_to_plot = "train_convergence_efficiency_1"
# metric_name_to_plot = "eval_convergence_efficiency_1"
# metric_name_to_plot = "train_asymptotic_performance_1"
# metric_name_to_plot = "eval_asymptotic_performance_1"

y_val_label = None
# y_val_label = "PPO Normalized Eval Convergence Efficiency"
# y_val_label = "PPO Normalized Eval Adaptive Efficiency"
# y_val_label = "PPO Normalized Train Adaptive Performance"

aggregator = "iqm"

if y_val_label is None:
    y_val_label = (
        f"{aggregator.title()} {metric_name_to_plot.replace('_', ' ').title()}"
    )

# Parse JSON data
with open(filename) as f:
    data_dict = json.load(f)

# Filter out runs with less than 4 data points
filtered_data = {}
for metric_name, metric_data in data_dict.items():
    filtered_data[metric_name] = {
        run_id_mapping[run_id]: value
        for run_id, value in metric_data.items()
        if data_dict[f"{metric_name_to_plot}_len"][run_id] >= 3
        and run_id in run_id_mapping
    }

# Convert data to pandas DataFrame
df = pd.DataFrame.from_dict(
    filtered_data[f"{metric_name_to_plot}_{aggregator}"],
    orient="index",
    columns=[f"{aggregator}"],
)

df /= df.loc["PPO Baseline", aggregator]

# Create bar chart using seaborn
df = df.reindex([v for v in run_id_mapping.values() if v in df.index])
sns.barplot(x=df.index, y=df[f"{aggregator}"])
sns.despine()  # Remove extra grid lines for cleaner visualization

# Customize the plot
plt.title(y_val_label)
plt.xlabel("Run ID")
plt.ylabel(y_val_label)
plt.show()  # To display the plot, uncomment this line

plt.close()

df.columns = [y_val_label]

print(df)  # Print the DataFrame containing the means for valid runs
