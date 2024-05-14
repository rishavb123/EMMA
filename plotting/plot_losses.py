import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt

filename = (
    "/Users/rbhagat/projects/EMMA/plotting/kp_orig/1/plot_results.pkl"
)

run_id_mapping = {
    "disagreement_diayn_key_prediction_door_key_change_10_1715020025": "10 Skill POI Diayn",
    "disagreement_diayn_key_prediction_door_key_change_1714548168": "5 Skill POI Diayn",
    "disagreement_ir_emb_key_prediction_door_key_change_1714165501": "IR POI emb",
    "key_prediction_door_key_change_1714165830": "PPO Baseline",
    "warmstart_both_disagreement_diayn_10_skills_key_prediction_door_key_change_1715352928": "100k Warmstart POI Diayn",
}

with open(filename, "rb") as f:
    plot_dfs = pkl.load(f)

for plot_col, plot_df in plot_dfs.items():

    plot_df = plot_df[[c for c in plot_df.columns if c in run_id_mapping]]
    plot_df.columns = [run_id_mapping[c] for c in plot_df.columns]

    train = "train" in plot_col

    sns.lineplot(data=plot_df)

    plt.legend()

    plt.xlabel("Steps")
    plt.ylabel(f"{'Train' if train else 'Eval'} External Model Loss")
    plt.title(f"{'Train' if train else 'Eval'} External Model Loss over steps")

    plt.show()
    plt.close()
