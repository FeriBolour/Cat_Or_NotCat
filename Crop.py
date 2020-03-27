from PIL import Image
import glob


def center_crop(im,new_width,new_height):
    width, height = im.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    im.show()


#folder='Y:/Github/Cat_Or_NotCat/Training set' # All jpegs are in this folder
folder = 'C:/Users/FarshadHome/Desktop'
imList=glob.glob(folder+'/*.jpg') # Reading all images with .jpg

for img in imList: # Loop
    im = Image.open(img)
    new_width = 500
    new_height= 900
    center_crop(im,new_width,new_height)
    new_width = 250
    new_height= 900
    center_crop(im,new_width,new_height)