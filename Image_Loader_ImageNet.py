"""
Method to extract images from ImageNet in Google Collab and create two different datasets for training and Validation
"""


from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2
import PIL.Image
import urllib
import matplotlib.pyplot as plt
import matplotlib.image as mimg



page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04037443")
print(page.content)

#beautiful soup library used to itemize the images from the HTML content of the link"""
soup = BeautifulSoup(page.content, 'html.parser')
str_soup = str(soup)

split_urls = str_soup.split('\r\n')
print(len(split_urls))

# resp = urllib.request.urlopen(split_urls[0])
# image = np.asarray(bytearray(resp.read()), dtype="uint8")
# image = cv2.imdecode(image, cv2.IMREAD_COLOR)
# print(resp,image)
# plt.imshow(image)

def url_to_image(url):

	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)


	return image


!mkdir /content/train 
!mkdir /content/train/cars 
!mkdir /content/validation
!mkdir /content/validation/cars 

# """Training DATA"""

n_of_training_images=100    #the number of training images to use
for progress in range(n_of_training_images):              #store all the images on a directory
    # Print out progress whenever progress is a multiple of 20 so we can follow the
    # (relatively slow) progress
    if(progress%20==0):
        print(progress)
    if not split_urls[progress] == None:
      try:
        I = url_to_image(split_urls[progress])
        if (len(I.shape))==3: #check if the image has width, length and channels
          save_path = '/content/train/cars/img'+str(progress)+'.jpg'#create a name of each image
          cv2.imwrite(save_path,I)

      except:
        None

# """Validation DATA"""



for progress in range(100):#store all the images on a directory
    # Print out progress whenever progress is a multiple of 20 so we can follow the
    # (relatively slow) progress
    if(progress%20==0):
        print(progress)
    if not split_urls[progress] == None:
      try:
        I = url_to_image(split_urls[n_of_training_images+progress])#get images that are different from the ones used for training
        if (len(I.shape))==3: #check if the image has width, length and channels
          save_path = '/content/validation/cars/img'+str(progress)+'.jpg'#create a name of each image
          cv2.imwrite(save_path,I)

      except:
        None

!ls /content/validation/cars/
img = mimg.imread('/content/validation/cars/img10.jpg')
plt.imshow(img)
