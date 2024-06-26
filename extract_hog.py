from skimage import io, color
from skimage.feature import hog


def extract_hog_features(image_path):
    # Load an image
    image = io.imread(image_path)
    gray_image = color.rgb2gray(image)  # Convert to grayscale

    # Parameters for HOG
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    block_norm = 'L2-Hys'

    # Extract HOG features and HOG image
    features, hog_image = hog(gray_image, orientations=orientations,
                              pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block,
                              block_norm=block_norm,
                              visualize=True)

    return features, hog_image, gray_image



