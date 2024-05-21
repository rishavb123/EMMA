import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt

filename = "/Users/rbhagat/projects/EMMA/plotting/outputs/kp_orig/plot_results.pkl"
run_id_mapping = {
    "disagreement_diayn_key_prediction_door_key_change_10_1715020025": "POI Diayn",
    # "disagreement_diayn_key_prediction_door_key_change_1714548168": "5 Skill POI Diayn",
    "disagreement_ir_emb_key_prediction_door_key_change_1714165501": "POI IR w/ POI emb",
    "key_prediction_door_key_change_1714165830": "PPO Baseline",
    "disagreement_diayn_key_prediction_door_key_change_1714165692": "Diayn Baseline",
    # "warmstart_both_disagreement_diayn_10_skills_key_prediction_door_key_change_1715352928": "100k Warmstart POI Diayn",
}

# filename = "/Users/rbhagat/projects/EMMA/plotting/outputs/kp_repeat/plot_results.pkl"
# run_id_mapping = {
#     "disagreement_diayn_10_skills_key_prediction_door_key_change_1715354304": "POI Diayn",
#     # "disagreement_diayn_10_30_uniform_skills_key_prediction_door_key_change_1715440148": "10 Skill POI Diayn, 0.3 uniform",
#     "disagreement_ir_emb_key_prediction_door_key_change_1715354338": "POI IR w/ POI emb",
#     "key_prediction_door_key_change_1715353432": "PPO Baseline",
#     "disagreement_diayn_10_100_uniform_skills_key_prediction_door_key_change_1715440497": "Diayn Baseline",
# }

# filename = "/Users/rbhagat/projects/EMMA/plotting/outputs/kp_repeat8/plot_results.pkl"
# run_id_mapping = {
#     "disagreement_diayn_10_skills_key_prediction_door_key_change_1715735913": "POI Diayn",
#     "disagreement_ir_emb_key_prediction_door_key_change_1715735695": "POI IR w/ POI emb",
#     "key_prediction_door_key_change_1715735670": "PPO Baseline",
#     "traditional_diayn_10_skills_key_prediction_door_key_change_1715886603": "Diayn Baseline",
# }

with open(filename, "rb") as f:
    plot_dfs = pkl.load(f)

for plot_col, plot_df in plot_dfs.items():

    plot_df = plot_df[[c for c in plot_df.columns if c in run_id_mapping]]
    plot_df.columns = [run_id_mapping[c] for c in plot_df.columns]

    plot_df = plot_df[run_id_mapping.values()]

    train = "train" in plot_col
    rewards = "rew" in plot_col

    plt.figure(figsize=(8, 5))

    if not rewards:
        if train:
            conv_thresh = 0.5
        else:
            conv_thresh = 2
        plt.axhline(
            y=conv_thresh,
            color="black",
            alpha=0.5,
            linestyle="dashed",
            label="Convergence Threshold",
        )

    sns.lineplot(data=plot_df)

    plt.legend()

    plt.xlabel("Steps")
    if not rewards:
        plt.ylabel(f"{'On Policy' if train else 'Random Agent'} External Model Loss")
        plt.title(
            f"{'On Policy' if train else 'Random Agent'} External Model Loss over steps"
        )
    else:
        plt.ylabel("Task Rewards")
        plt.title("Task Rewards over steps")

    plt.show()
    plt.close()
