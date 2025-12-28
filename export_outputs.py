import numpy as np
import pandas as pd

# Z is (T, 3): [arousal, valence, dominance]
Z = latent_z  # from your existing code

df = pd.DataFrame(Z, columns=["arousal", "valence", "dominance"])
df["time_index"] = np.arange(len(Z))

df.to_csv("trajectory.csv", index=False)
print("Saved trajectory.csv")
