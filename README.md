**6DoF Pose estimation from keypoints**

This code is the python adaptation of the matlab post processing you can find [here](https://github.com/wbenbihi/hourglasstensorlfow)
I also used the their neural network in an other repo.
I just translated the matlab code into python code, I did not invented anything
Feel free to use this code if you need it.
call main.py to create a figure with the cad projection on it. There is an exemple of the data you need in the folder images_test
The code is pretty simple to adapt, if you have question ask me.
The video maker is not very optimized, it saves all the frame, you can use it almost the same way than main.py


**C++ code**

I made a c++ version of the code. It is faster and suitable for real time application. The code currently print the rotation of the translation matrix.
The code need GSL and OpenCV to work. When you installed this libraries, the *make* should works and give you a executable.

The c++ main take three arguments :

    1. the path the image (example : ./../images_test/val_01_00_000000.bmp)
    2. the path to the keypoints (example : ./../images_test/val_01_00_000000_00.bmp)
    3. verbosity, give verbose for a verbose output, anythink else for a no verbose output
    
the other parameters like the number of joints, the data related to the cad model, the dimention, the name of the joints...
should be modified in the ***utils_struct.h*** part of the code.  