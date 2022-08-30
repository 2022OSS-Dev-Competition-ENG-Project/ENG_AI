from PIL import Image
import glob



files = glob.glob("./nomal/*.jpeg")


for f in files :
    img = Image.open(f)
    img_resize = img.resize((255,255))
    img_resize.save(f,"JPEG")

