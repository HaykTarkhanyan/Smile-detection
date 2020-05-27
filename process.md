**NOTE:** I was training models in google colab so all of my code is there,  all data files
and experiments are in my drive, so I'm gonna export codes here and push them
to GitHub(without writing commit messages:) but I'm leaving some
of hardcoded values because your not going to change file directory etc. So perhaps
it would be fair not to judge my code cleannes by only this repo :)



I found dataset called LFW(labeled faces in the wild), then I found text files
where smilings images filenames where stored which was quite small subset something
like 600 from 13000. Then i found very same dataset but with faces cropped and
I decided to use that one instead of end to end approach because I didn't have much
data to detect smile from big resoultion image with different backgrounds, but
there were lots of pretrained models to detect face, and it would be easier just
to classify if there is smile in the image of the face.



One more decision I had to make was whether to use grayscale images or RGB.
I decided to use grayscale because it would be faster to train a model, and I
think model will be able to extract correct feature map just from grayscales.
Though gut feeling isn'y acceptable, I will try RGBs if i got time

I choosed **keras** because I have the most expericance of working with it.

For model choice I decided to use one of the classic architechures because it
would take me long time to converg to optimal custom architechure.

my input shape was (64,64) so perhaps the obvious choice was **LeNet 5**(old but gold)
but I made some modifications like using MaxPooling instead of average, and ReLu
activation instead of tanh to avoid wanisihing gradients.

About checking if eyes are open,
After lots of googling i decided to try eye detection models in openCV, and if
they can't detect eyes then I assume that eyes are closed, though sometimes model
is able to detect eyes if they are just slightky closed, hence expect a little bit of
True Negatives


For extracting bounding boxes of the face I use this model offered by OpenCV.
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
And for detecting eyes this one
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml


After experiments and hyperparameter tuning classification model I converged to
a model that is doing fine. I'll push two models, one is working badly when there is to much light
I guess because when converting to gracyscale it becomes indestiguishable from teeth. One model 
does badly when is too dark in the room. And both struggle with classifing smiles too fat from camera


