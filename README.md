# MRI-Image-Convolutional-Neural-Network-in-Python

Hey!

I've been super interested in Pytorch recently, and this project was the perfect excuse to get started!

This is a super basic convolutional neural net that analyzes and sorts 3 classes of MRI photos (lung cancer, brain cancer, and non-cancerous cerebral abcesses)
I intend to try to make a diagnostic neural net and add on to this project, but it shall take a while to get the appropriate control dataset together, as 
medical databases do not often include benign photos. 

Additionally, some issues may present themselves regarding the plane of view each photo was taken in (axial, sagittal, etc.). We may get around this by supplying each 
view type as its own class.

A special thanks to NIH MedPix, who supplied all the photos for the dataset.

Additionally, I have included the feature map code and some photos that give some insight into which patterns the trained net deems most important in training. 
It's interesting to see the net highlight the marked contrasts on the boundaries of abcesses and glioblastomas, as well as to see the different ways image planarity affects 
the lung class judgements.

I've included:The main script to create the net,the compiled data set, as well as the script to run a feature map on this type of net (one may simply torch.load(r"....",weights_only=False) to check their pretrained net)


Feel free to add on and improve the project as you like!






![image](https://github.com/user-attachments/assets/1b01e5d4-2618-430c-92e4-7e0547679910)
![image](https://github.com/user-attachments/assets/361c4e10-641e-4b3d-89d5-bf9a94790687)
![image](https://github.com/user-attachments/assets/a739e5b7-5212-49ab-9204-749baebff9dd)
![image](https://github.com/user-attachments/assets/a1b8e7cd-4a4e-45a7-ae03-29050e670374)
![image](https://github.com/user-attachments/assets/684b8c06-2441-4b6d-a14a-d348eb7230eb)
![image](https://github.com/user-attachments/assets/0e8db53e-ce1a-4caa-9c64-c7f24caa4428)
![image](https://github.com/user-attachments/assets/c8e2f5e6-3f44-4ea2-beeb-a69f261cde58)
![image](https://github.com/user-attachments/assets/d4b4fc0d-6107-4a19-b0ad-375133579981)
![image](https://github.com/user-attachments/assets/372fc4ba-ff20-40a6-a1a6-24d0d1dc78cb)


