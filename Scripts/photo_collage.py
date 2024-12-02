from PIL import Image
import os
import math
import random
import shutil

def create_photo_collage(folder_path, collage_width=800, thumbnail_size=(100, 100), output_file="collage.jpg"):
    """
    Creates a photo collage from images in a folder.

    Args:
        folder_path (str): Path to the folder containing images.
        collage_width (int): Width of the collage in pixels.
        thumbnail_size (tuple): Size of each thumbnail in the collage (width, height).
        output_file (str): Name of the output collage file.
    """
    # List all image files in the folder
    images = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
    ]

    if not images:
        print("No images found in the specified folder.")
        return

    # Calculate grid dimensions
    thumbnails_per_row = collage_width // thumbnail_size[0]
    rows = math.ceil(len(images) / thumbnails_per_row)
    collage_height = rows * thumbnail_size[1]

    # Create a blank canvas for the collage
    collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))

    # Add each image as a thumbnail to the collage
    x_offset, y_offset = 0, 0
    for image_path in images:
        with Image.open(image_path) as img:
            img.thumbnail(thumbnail_size)
            collage.paste(img, (x_offset, y_offset))

            # Update offsets for the next image
            x_offset += thumbnail_size[0]
            if x_offset >= collage_width:
                x_offset = 0
                y_offset += thumbnail_size[1]

    # Save and display the collage
    collage.save(output_file)
    collage.show()

    print(f"Collage saved as {output_file}.")


def select_random_photos_from_folders(source_dir, output_dir):
    """
    Selects one random photo from each folder in a directory structure and copies it to a new directory.

    Args:
        source_dir (str): The path to the directory containing subfolders with photos.
        output_dir (str): The path to the directory where selected photos will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each subdirectory in the source directory
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        # Skip if it's not a directory
        if not os.path.isdir(folder_path):
            continue

        # Collect all image file paths in the current folder
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
        image_files = [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.lower().endswith(image_extensions)
        ]

        # If no images found, skip this folder
        if not image_files:
            print(f"No images found in folder: {folder_path}")
            continue

        # Pick a random image from the current folder
        random_photo = random.choice(image_files)

        # Copy the random photo to the output directory
        output_file_path = os.path.join(output_dir, os.path.basename(random_photo))
        shutil.copy(random_photo, output_file_path)

        print(f"Random photo selected: {random_photo}")
        print(f"Copied to: {output_file_path}")

# Usage
def main():
    source = "C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/local/clipped_frames"
    output = "C:/Users/kowen/OneDrive - University of Alberta/Projects/GoalGuru/local/selected_photos" 
    for folder in os.listdir(source):
        folder_path = os.path.join(source, folder)
        select_random_photos_from_folders(folder_path, output)
    create_photo_collage(output)

if __name__ == "__main__":
    main()