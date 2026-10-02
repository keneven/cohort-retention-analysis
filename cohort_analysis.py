"""
Cohort Retention & Customer Lifetime Value Analysis
---------------------------------------------------
Generates a subscription-style transaction log, then builds:
 - a monthly signup-cohort retention matrix (triangle heatmap)
 - retention decay curves
 - revenue-retention and simple LTV estimates
Outputs charts + findings.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(11)

# ---------- Generate subscription activity ----------
START = pd.Timestamp("2023-01-01")
MONTHS = 18
cohort_sizes = rng.integers(350, 700, MONTHS)   # new customers per signup month
rows = []
cust = 0
# base monthly retention improves slightly for later cohorts (product maturing)
for m in range(MONTHS):
    signup = START + pd.DateOffset(months=m)
    for _ in range(cohort_sizes[m]):
        cust += 1
        alive = True
        month_idx = 0
        arpu = rng.uniform(15, 40)              # avg revenue per user / month
        # Each customer has a persistent loyalty level -> a loyal core survives long term.
        loyalty = rng.beta(2, 2) + m * 0.004    # later cohorts skew slightly more loyal
        while alive and (m + month_idx) < MONTHS:
            active_date = signup + pd.DateOffset(months=month_idx)
            rows.append((cust, signup, active_date, month_idx, round(arpu, 2)))
            # Retention: high floor for loyal users, early-life churn is heaviest.
            floor = 0.86 + 0.12 * loyalty                      # long-run monthly retention
            early_penalty = 0.15 * np.exp(-month_idx / 3.0)    # extra churn in first months
            ret = min(0.99, floor - early_penalty)
            alive = rng.random() < ret
            month_idx += 1

log = pd.DataFrame(rows, columns=["customer_id","cohort","active_date","month_since_signup","revenue"])
log.to_csv("data/subscription_activity.csv", index=False)
print(f"Customers: {log.customer_id.nunique():,} | Activity rows: {len(log):,}")

# ---------- Retention matrix ----------
cohort_label = log.cohort.dt.strftime("%Y-%m")
counts = (log.assign(cohort_label=cohort_label)
             .groupby(["cohort_label","month_since_signup"]).customer_id.nunique().unstack())
sizes = counts[0]
retention = counts.divide(sizes, axis=0) * 100

fig, ax = plt.subplots(figsize=(11, 6))
im = ax.imshow(retention.values, cmap="Blues", aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(retention.shape[1])); ax.set_xticklabels(retention.columns)
ax.set_yticks(range(retention.shape[0])); ax.set_yticklabels(retention.index)
ax.set_xlabel("Months since signup"); ax.set_ylabel("Signup cohort")
ax.set_title("Monthly Retention by Signup Cohort (%)")
for i in range(retention.shape[0]):
    for j in range(retention.shape[1]):
        v = retention.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 55 else "#222")
fig.colorbar(im, label="Retention %"); fig.tight_layout()
fig.savefig("charts/01_retention_matrix.png"); plt.close()

# ---------- Average retention curve ----------
avg_curve = retention.mean(axis=0)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(avg_curve.index, avg_curve.values, marker="o", color="#2b6cb0")
for x, yv in zip(avg_curve.index, avg_curve.values):
    if x in (0,1,3,6,12): ax.annotate(f"{yv:.0f}%", (x, yv), textcoords="offset points", xytext=(0,8), fontsize=8)
# The tail of this curve rests on fewer and fewer cohorts: a cohort that signed up
# in month 15 can only ever be observed for 3 months. Month 12 is the average of
# the 6 oldest cohorts, not of all 18. The mean already excludes unobserved cells,
# but the reader cannot see the sample thinning unless it is drawn.
curve_n = retention.notna().sum(axis=0)
axn = ax.twinx()
axn.bar(curve_n.index, curve_n.values, color="#cbd5e0", alpha=0.45, zorder=0)
axn.set_ylabel("Cohorts behind each point", color="#718096")
axn.grid(False)
ax.set_zorder(axn.get_zorder() + 1); ax.patch.set_visible(False)
ax.set_title("Average Retention Curve (bars show cohorts observed)")
ax.set_xlabel("Months since signup"); ax.set_ylabel("Retention %")
fig.tight_layout(); fig.savefig("charts/02_retention_curve.png"); plt.close()

# ---------- Simple LTV ----------
arpu = log.revenue.mean()
m1 = avg_curve.get(1, np.nan)
m3 = avg_curve.get(3, np.nan)
m6 = avg_curve.get(6, np.nan)
m12 = avg_curve.get(12, np.nan)
# Average observed lifetime is censored: anyone still subscribed when the window
# closes has their life cut short by the data, not by churning. 38.7% of customers
# here are in that position, which drags the mean down and understates LTV.
# Instead, read expected lifetime off the survival curve (the area under it), then
# extrapolate the tail using the long-run monthly retention rate.
naive_lifetime = log.groupby("customer_id").month_since_signup.max().add(1).mean()

surv = avg_curve / 100.0                      # S(m), proportion still active
observed_area = surv.sum()                    # expected months inside the window
tail_ratios = (surv.shift(-1) / surv).dropna().tail(6)
r_longrun = float(tail_ratios.mean())         # steady-state monthly retention
tail_area = surv.iloc[-1] * r_longrun / (1 - r_longrun) if r_longrun < 1 else np.nan
expected_lifetime = observed_area + tail_area

GROSS_MARGIN = 0.80                           # assumption: stated, not hidden
ltv = arpu * GROSS_MARGIN * expected_lifetime

with open("charts/findings.txt", "w") as f:
    f.write(f"Total customers: {log.customer_id.nunique():,}\n")
    f.write(f"ARPU (avg revenue/user/active month): ${arpu:.2f}\n")
    f.write(f"Naive observed lifetime (censored): {naive_lifetime:.1f} months\n")
    f.write(f"Expected lifetime from survival curve: {expected_lifetime:.1f} months\n")
    f.write(f"  within the 18-month window: {observed_area:.1f} | extrapolated tail: {tail_area:.1f}\n")
    f.write(f"  long-run monthly retention used for the tail: {r_longrun*100:.1f}%\n")
    f.write(f"Estimated LTV (ARPU x {GROSS_MARGIN:.0%} margin x lifetime): ${ltv:.0f}\n\n")
    f.write("Average retention curve (%):\n")
    for x, yv in avg_curve.items():
        f.write(f"  Month {int(x):2d}: {yv:.1f}%\n")

print(f"ARPU ${arpu:.2f} | naive lifetime {naive_lifetime:.1f} mo (censored) "
      f"| expected lifetime {expected_lifetime:.1f} mo | LTV ${ltv:.0f}")
print(f"Retention M1 {m1:.1f}% | M3 {m3:.1f}% | M6 {m6:.1f}% | M12 {m12:.1f}%")
print("Done. Charts + findings in charts/")
