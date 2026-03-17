from torchvision.datasets import CocoCaptions
from PIL import Image
import matplotlib.pyplot as plt

# Paths relative to where you run this script
root = "data/coco/val2017"
ann_file = "data/coco/annotations/captions_val2017.json"

# 1) Load dataset
coco = CocoCaptions(root=root, annFile=ann_file, transform=None)

print("Number of images:", len(coco))

# 2) Get a sample
img, captions = coco[0]  # img is a PIL.Image, captions is a list of strings

# 3) Show the image
plt.imshow(img)
plt.axis("off")
plt.title("COCO sample image")
plt.savefig("imag1")
# 4) Print its captions
print("Captions for this image:")
for i, c in enumerate(captions):
    print(f"{i+1}. {c}")
