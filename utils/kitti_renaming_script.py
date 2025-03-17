import os

directory = "./datasets/kitti_holdout/gt_image_2/"

for filename in os.listdir(directory):
    if "road" in filename:
        new_filename = filename.replace("road_", "")  # Remove "road"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_filename}')

print("Renaming completed!")
