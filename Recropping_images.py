from PIL import Image
import os

# definerer funktion
def resize_and_crop(input_folder, output_folder):
    # Gennemser hvert billede i input folderen
    for filename in os.listdir(input_folder):
        # JPG eller JPEG
        if filename.lower().endswith((".jpg", ".jpeg")):
            # Load billede
            img_path = os.path.join(input_folder, filename)
            original_image = Image.open(img_path)

            # Finder mindste side
            min_dimension = min(original_image.width, original_image.height)

            target_width = int(original_image.width * (300 / min_dimension))
            target_height = int(original_image.height * (300 / min_dimension))

            resized_image = original_image.resize((target_width, target_height), resample=Image.LANCZOS)

            center_x = resized_image.width // 2
            center_y = resized_image.height // 2

            crop_box = (
                max(0, center_x - 150),  # left
                max(0, center_y - 150),  # upper
                min(resized_image.width, center_x + 150),  # right
                min(resized_image.height, center_y + 150)  # lower
            )

            # Cropper
            cropped_image = resized_image.crop(crop_box)

            # Gemmer billeder
            output_path = os.path.join(output_folder, f"cropped_{filename}")
            cropped_image.save(output_path)

if __name__ == "__main__":
    # Definerer input- og output-mapper
    input_folder = r"C:\Users\Bruger\OneDrive\Skrivebord\ITIS project\ISIC2020 tr\train"
    output_folder = r"C:\Users\Bruger\OneDrive\Skrivebord\ITIS project\ISIC2020 formateret"

    # Kører funktionen
    resize_and_crop(input_folder, output_folder)
