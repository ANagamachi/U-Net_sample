# U-Net_sample
Learning CFD data using U-Net and predict conformation stress from velocity.

## How to use
### Learning
~~~
$ python unet_y4stuicxx.py

Input 4 (height from wall)

Input 0 (index number)
~~~
Then U-Net_sample/infertanh/unet_y4stuicij/unet_y4stuicxx/unet_y4stuicxx_model.h5 etc. will be written.

Then do same thing changing index number from 1 to 5.

### Predict for test data
~~~
$ cd infertanh
$ python infer_unet_y4stuicij.py

Input 4

Input 884

Input 10000
~~~
Then U-Net_sample/infertanh/unet_y4stuicij/884_10000/ will be written.
