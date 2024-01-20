# SelfieVideoGenerator
creates video from selfie images

folder structure:  
&emsp;*code : scripts  
&emsp;*data : selfie images  
&emsp;*model : face landmark detection model  
  
1- put selfie images into data folder  
2- install dependencies. requirements.txt contains necessary libraries  
3- download face landmark model from https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat into model folder  
4- edit parameters in videoCreate script - OPTIONAL  
&emsp;set blenderC value. images blend for ***blenderC*** frames between images  
&emsp;set freezerC value. images freeze for ***freezerC*** frames  
5- run videoCreate script  
6- selfie video will be stored in code folder, convert to other video formats for a smaller video size  
  
* Recommendation: selfie images should contain one face
