import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# === Configuration ===
export_dir = "GradCAM_ResNet50_Model"
csv_path = os.path.join(export_dir, "activation_features.csv")
summary_path = os.path.join(export_dir, "family_to_category_summary.txt")
plot_path = os.path.join(export_dir, "family_category_distribution.png")

# === Category Mapping ===
category_map = {
    "TrojanDownloader": ["actions/activities", "internet connection"],
    "Ransomware": ["storage & settings", "actions/activities", "sensitive data collection", "C&C", "hardware", "antivirus", "internet connection", "media"],
    "Spyware": ["sensitive data collection"],
    "Adware": ["actions/activities", "internet connection"],
    "Rootkit": ["actions/activities", "internet connection"],
    "Worm": ["internet connection", "actions/activities"],
    "Backdoor": ["C&C", "hardware", "actions/activities", "internet connection"],
    "Keylogger": ["sensitive data collection"],
    "BankingTrojan": ["sensitive data collection", "internet connection", "C&C", "actions/activities"],
    "Dropper": ["actions/activities"],
    "FileInfector": ["media"],
    "PUA": ["sensitive data collection"],
    "TrojanSMS": ["actions/activities", "internet connection"],
    "TrojanSpy": ["sensitive data collection"],
    "ZeroDay": ["Unknown vulnerabilities"],
    "No Category": ["require further investigation"],
}

# === Load Data ===
df = pd.read_csv(csv_path)
assert "Label" in df.columns, "❌ 'Label' column not found in CSV."

# === Map malware families to behavioral categories ===
df["Categories"] = df["Label"].apply(lambda label: ", ".join(category_map.get(label, ["unknown"])))

# === Count categories per family ===
family_category_counts = defaultdict(lambda: defaultdict(int))
for _, row in df.iterrows():
    family = row["Label"]
    for category in category_map.get(family, ["unknown"]):
        family_category_counts[family][category] += 1

# === Save textual summary ===
summary_lines = ["Malware Family to Behavior Category Mapping:\n"]
for family, cats in family_category_counts.items():
    summary_lines.append(f"\n🔹 {family}:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        summary_lines.append(f"   - {cat}: {count} samples")

with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))
print(f"📄 Summary report saved to: {summary_path}")

# === Prepare data for visualization ===
plot_data = []
for family, cat_dict in family_category_counts.items():
    for cat, count in cat_dict.items():
        plot_data.append({
            "Malware Family": family,
            "Behavioral Category": cat,
            "Count": count
        })

plot_df = pd.DataFrame(plot_data)

# === Plotting ===
plt.figure(figsize=(16, 7))
sns.barplot(data=plot_df, x="Malware Family", y="Count", hue="Behavioral Category")
plt.xticks(rotation=45, ha='right')
plt.title("Behavioral Categories per Malware Family")
plt.xlabel("Malware Family")
plt.ylabel("Number of Samples")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
print(f"📊 Bar chart saved to: {plot_path}")

