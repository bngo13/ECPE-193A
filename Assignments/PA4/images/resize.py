import cv2
import sys
import os

def main():
    if len(sys.argv) != 4:
        print("Usage: python resize.py <input_image> <height> <width>")
        sys.exit(1)

    input_image = sys.argv[1]
    height = int(sys.argv[2])
    width = int(sys.argv[3])

    # Read the image
    img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not open or find the image '{input_image}'")
        sys.exit(1)

    # Resize the image
    resized_img = cv2.resize(img, (width, height))

    # Generate output filename based on the input filename and new size
    base, ext = os.path.splitext(input_image)
    output_image = f"{height}x{width}_{base}{ext}"

    # Save the resized image
    cv2.imwrite(output_image, resized_img)
    print(f"Resized image saved as {output_image}")

if __name__ == "__main__":
    main()
