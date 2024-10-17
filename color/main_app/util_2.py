import zipfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Upload image and convert to grayscale
def load_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale format
    return img, gray


# Color clustering by K-means
def cluster_colors(image, num_clusters):
    pixels = image.reshape(-1, 3)  # Convert pixel data to 2D array (RGB format)

    # K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixels)

    # Cluster centers (colors) and labels (labels)
    labels = kmeans.labels_  # Cluster number of each pixel

    centers = kmeans.cluster_centers_.astype(int)  # Cluster centers (RGB colors)
    return labels.reshape(image.shape[:2]), centers


# Create a mask to find cluster contours
def find_contours(labels, num_clusters):
    h, w = labels.shape
    contour_image = np.zeros((h, w), dtype=np.uint8)

    # Finding contours for each cluster
    for cluster_idx in range(num_clusters):
        mask = np.uint8(labels == cluster_idx)  # Create a mask for each cluster
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours with black lines
        cv2.drawContours(
            contour_image, contours, -1, (255, 255, 255), 1
        )  # Outlines in white
    return contour_image


# Add numbers to each cluster
def label_image(image, labels, num_clusters):
    h, w = labels.shape
    labeled_image = np.ones((h, w), dtype=np.uint8) * 255  # Image with white background

    # Add cluster numbers for each unique cluster
    for cluster_idx in range(num_clusters):
        mask = (labels == cluster_idx).astype(np.uint8)
        moments = cv2.moments(mask)

        # Calculate center of mass for the cluster
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])  # Convert to integer
            cY = int(moments["m01"] / moments["m00"])  # Convert to integer

            # Add the number of the cluster at its center
            cv2.putText(
                labeled_image,
                str(cluster_idx),
                (cX, cY),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

    return labeled_image


# Combine the image with the outline
def combine_with_edges(labeled_image, contours):
    combined_image = np.copy(labeled_image)
    combined_image[contours == 255] = 0  # Draw the contours with black
    return combined_image


# Recolor the image based on the clusters
def recolor_image(labels, centers):
    h, w = labels.shape
    recolored_image = np.zeros(
        (h, w, 3), dtype=np.uint8
    )  # Create a blank image with the same dimensions

    # Replace each pixel with its cluster's center color
    for i in range(h):
        for j in range(w):
            recolored_image[i, j] = centers[
                labels[i, j]
            ]  # Set the pixel to the cluster center's color

    return recolored_image


# Save the white picture
def save_white_image(image, output_path):
    cv2.imwrite(
        output_path.replace(".png", "_white.png"), image
    )  # Save the result to a file
    plt.savefig(output_path.replace(".png", "_white.svg"), format="svg")  # Save as SVG
    # plt.close()


# Save the recolored image
def save_recolored_image(image, output_path):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
    cv2.imwrite(output_path.replace(".png", "_with_colors.png"), image_bgr)
    plt.savefig(
        output_path.replace(".png", "_with_colors.svg"), format="svg"
    )  # Save as SVG


# Save colors to a text file
def save_colors_to_text(centers, output_path):
    with open(output_path.replace("png", "txt"), "w") as f:
        for idx, center in enumerate(centers):
            rgb_color = f"Cluster {idx}: RGB({center[0]}, {center[1]}, {center[2]})\n"
            f.write(rgb_color)


# Create a PNG with colors and their RGB values
def create_color_image(centers, output_path):
    # Create an image to display the colors and their RGB codes
    bar_height = 200
    bar_width = 500
    padding = 10
    height = (bar_height + padding) * len(centers)
    width = bar_width
    color_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # Draw each color with its RGB code
    for idx, center in enumerate(centers):
        color_block_start_y = idx * (bar_height + padding)
        color_block_end_y = color_block_start_y + bar_height

        # Draw color rectangle
        color_image[color_block_start_y:color_block_end_y, :] = center

        # Add RGB text to the image
        rgb_text = f"Color {idx} - RGB: {center[0]}, {center[1]}, {center[2]}"
        cv2.putText(
            color_image,
            rgb_text,
            (10, color_block_end_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

    # Save the image as PNG
    cv2.imwrite(output_path.replace(".png", "_colors.png"), color_image)


def zip_files(file_paths, output_zip):
    with zipfile.ZipFile(output_zip, "w") as zipf:
        for file in file_paths:
            zipf.write(file, os.path.basename(file))


# Main function
def main(image_path, output_path, num_clusters):
    # 1. Upload image and convert to grayscale
    original_image, gray_image = load_image(image_path)

    # 2. Color separation with K-means (clusters)
    labels, centers = cluster_colors(original_image, num_clusters)

    # 3. Find the contours of each cluster
    contours = find_contours(labels, num_clusters)

    # 4. Create a numbered black and white image
    labeled_image = label_image(original_image, labels, num_clusters)

    # 5. Recolor the image based on clusters
    recolored_image = recolor_image(labels, centers)

    # 6. Combine contours and numbers
    final_image = combine_with_edges(labeled_image, contours)

    # 7. Save the whitw picture
    save_white_image(final_image, output_path)

    # 8. Save the recolored image
    save_recolored_image(recolored_image, output_path)

    # 9. Save colors to text file
    save_colors_to_text(centers, output_path)

    # 10. Create a PNG with colors and their RGB values
    create_color_image(centers, output_path)
    
    files_to_zip = [
        output_path.replace(".png", "_white.png"),
        output_path.replace(".png", "_white.svg"),
        output_path.replace(".png", "_with_colors.png"),
        output_path.replace(".png", "_with_colors.svg"),
        output_path.replace("png", "txt"),
        output_path.replace(".png", "_colors.png"),
    ]
    
    # 11. Save zip file
    zip_files(files_to_zip, output_path.replace("png", "zip"))
