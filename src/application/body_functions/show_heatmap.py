import cv2
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt


def show_heatmap(probability_map, img):
    probability_map = cv2.resize(probability_map, (img.shape[1], img.shape[0]))
    plt.figure(figsize=[14, 10])
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(probability_map, alpha=0.6)
    plt.colorbar()
    plt.axis("on")
    plt.show()
