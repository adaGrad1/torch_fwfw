from util import datapath, return_and_create_fpath


import os
import shutil


print("Downloading archive...")
!wget https://pjreddie.com/media/files/imagenet64.tar
print("Copying to local runtime...")
shutil.move('imagenet64.tar', datapath('imagenet64.tar'))
print("Uncompressing...")
os.chdir(datapath(''))
!tar -xf imagenet64.tar
print("Data ready!")