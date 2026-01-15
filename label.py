import pandas as pd
import glob
import os

folder = "/Users/stella/Downloads/Data Training/data"   # 🔥 đường dẫn đúng đến folder chứa file thu

files = glob.glob(folder + "/*.csv")

for file in files:
    base = os.path.basename(file)
    label = base.split("_")[0]        # SIGN là phần trước dấu _
    
    df = pd.read_csv(file)
    df["SIGN"] = label                # Thêm cột cuối
    df.to_csv(file, index=False)

    print(f"✔ SIGN='{label}' added to {base}")

print("\n🎉 DONE! Check CSV again.")
