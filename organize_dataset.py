import os, shutil

# Source folders
train_src = "data/cats-dogs/training_set/training_set"
test_src = "data/cats-dogs/test_set/test_set"

# Destination folders
train_dst = "data/train"
test_dst = "data/test"

# Create destination folders
for split_dst in [train_dst, test_dst]:
    for cls in ["cats", "dogs"]:
        os.makedirs(os.path.join(split_dst, cls), exist_ok=True)

def copy_images(src_root, dst_root, split_name):
    for cls in ["cats", "dogs"]:
        src_path = os.path.join(src_root, cls)
        dst_path = os.path.join(dst_root, cls)
        count = 0
        for img in os.listdir(src_path):
            if img.lower().endswith(".jpg"):
                full_src = os.path.join(src_path, img)
                full_dst = os.path.join(dst_path, img)
                if os.path.isfile(full_src):
                    try:
                        shutil.copy(full_src, full_dst)
                        count += 1
                    except Exception as e:
                        print(f"❌ Failed to copy {img}: {e}")
        print(f"✅ Copied {count} {cls} images to {split_name}/{cls}")

copy_images(train_src, train_dst, "train")
copy_images(test_src, test_dst, "test")