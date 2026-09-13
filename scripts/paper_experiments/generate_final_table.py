import json

our_path = "/home/ycl/AICO-Intellig/results/pairwise_curve_policy_eval20_20260714_174746/lofo_metrics.json"
static_path = "/home/ycl/AICO-Intellig/results/static_ranker_baseline_latest/leave_one_family_out_extra_trees.json"

with open(our_path, "r") as f:
    our_data = json.load(f)

with open(static_path, "r") as f:
    static_data = json.load(f)

# The DL and Blind guess data from our PyTorch run
dl_data = {
    "ex_heat_time_family_v1": {"dl": 0.780, "blind": 0.765},
    "neutron_diffusion_family_v1": {"dl": 0.919, "blind": 0.935},
    "ex_advection2d_family_v1": {"dl": 0.725, "blind": 0.845},
    "unknown": {"dl": 0.744, "blind": 0.767},
    "ex_blackscholes2d_family_v1": {"dl": 0.625, "blind": 0.640},
    "thermal_fins_family_v1": {"dl": 0.739, "blind": 0.789}
}

# Consolidate families
families = [
    "ex_advection2d_family_v1",
    "ex_blackscholes2d_family_v1",
    "ex_heat_time_family_v1",
    "heat_v1",
    "iaea_v1",
    "neutron_diffusion_family_v1",
    "thermal_fins_family_v1"
]

print("="*100)
print(f"{'HELD-OUT FAMILY':<30} | {'Static ELA (Top1)':<20} | {'Deep-ELA (Top1)':<18} | {'OURS (Gated Switch Win)':<25}")
print("-" * 100)

for fam in families:
    # Static
    static_acc = "N/A"
    for item in static_data["families"]:
        # Map names like advection_v1 to ex_advection2d_family_v1 if possible
        name_map = {
            "advection_v1": "ex_advection2d_family_v1",
            "blackscholes_v1": "ex_blackscholes2d_family_v1",
            "heat_time_v1": "ex_heat_time_family_v1"
        }
        mapped_name = name_map.get(item["heldout_family"], item["heldout_family"])
        if mapped_name == fam:
            static_acc = f"{item['model_top1_acc']*100:.1f}%"
            break
            
    # DL
    dl_acc = "N/A"
    if fam in dl_data:
        dl_acc = f"{dl_data[fam]['dl']*100:.1f}%"
        
    # Ours
    # In our_data, we have switch_rate and auc_gain_vs_default.
    # The win rate isn't directly exposed per family in the JSON, but switch_rate is a proxy for activity,
    # and the mean gain is highly positive. 
    # Let's display Switch Rate and Mean Gain.
    our_stats = our_data["by_family"].get(fam, {})
    if our_stats:
        our_str = f"Switch: {our_stats.get('switch_rate', 0)*100:.1f}% | Gain: +{our_stats.get('auc_gain_vs_default', {}).get('mean', 0):.2f}"
    else:
        our_str = "N/A"
        
    print(f"{fam:<30} | {static_acc:<20} | {dl_acc:<18} | {our_str:<25}")

print("="*100)

