import os

folder = "data/topo"

for filename in os.listdir(folder):
    # 只处理 .npy 文件
    if not filename.endswith(".npy"):
        continue

    # 跳过已经处理过的
    if filename.endswith("_topo.npy"):
        continue

    old_path = os.path.join(folder, filename)

    # 去掉 .npy，加上 _topo.npy
    new_name = filename[:-4] + "_topo.npy"
    new_path = os.path.join(folder, new_name)

    # 防止覆盖
    if os.path.exists(new_path):
        print(f"[跳过] 已存在: {new_name}")
        continue

    os.rename(old_path, new_path)
    print(f"[重命名] {filename} -> {new_name}")

print("处理完成！")
