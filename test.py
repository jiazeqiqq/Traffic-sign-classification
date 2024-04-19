import Feature_extraction as fe
import matplotlib.pyplot as plt
path = './dataset/new_imageset/003_1_0001_1_j.png'
features, hog_image, gray_image = fe.extract_hog_features(path)
print("features:", features.shape)

#Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(gray_image, cmap=plt.cm.gray)
ax1.set_title('Input Image')
ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()