import os

folder = f"C:/Users/natep/Documents/GitHub/CAPSTONE2022-2023/haar/temp/n/"

for i, fn in enumerate(os.listdir(folder)):
    src = f"{folder}/{fn}"
    dest = f"C:/Users/natep/Documents/GitHub/CAPSTONE2022-2023/haar/temp/n/{i+1}.jpg"
    os.rename(src, dest)