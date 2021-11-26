from nudenet import Detector
from PIL import Image

if __name__ == '__main__':
    img = Image.open("rga6845.jpg")
    x = Detector()
    print(x.detect(img))
