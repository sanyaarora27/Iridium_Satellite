import numpy as np

total = {}
for seg in range(5):
    sats = np.load(f"data/ra_sat_{seg:03d}.npy")
    for s in [51, 85, 87, 92, 109]:
        total[s] = total.get(s, 0) + int(np.sum(sats == s))

for s, n in sorted(total.items()):
    print(f"Satellite {s}: {n} messages across all 5 segments")
print(f"Total: {sum(total.values())}")
