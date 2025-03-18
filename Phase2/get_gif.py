import glob
import cv2
import imageio

imgs = glob.glob("image/*.png")

# Sort the images by filename
imgs.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

print(imgs)
# Read the images and store them in a list
frames = []
for img in imgs:
    frames.append(cv2.imread(img))
    
# Save the frames as a GIF
with imageio.get_writer('image/output.gif', mode='I', fps=2) as writer:
    for image in frames:
        writer.append_data(image)
writer.close()