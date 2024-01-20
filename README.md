# SelfieVideoGenerator
creates video from selfie images

folder structure:
  > code : scripts
  > data : selfie images
  > model : face landmark detection model

1- put selfie images into data folder
2- install dependencies. requirements.txt contains necessary libraries
3- edit parameters in videoCreate script - OPTIONAL
  set blenderC value. images blend for <blenderC> frames between images
  set freezerC value. images freeze for <freezerC> frames 
4- run videoCreate script
5- selfie video will be stored in code folder, convert to other video formats for a smaller video size

* Recommendation: selfie images should contain one face
