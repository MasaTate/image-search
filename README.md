# image-search
This is similar image retrieval system by AKAZE and BoVW.
If you create BoVW of all images in database, you can easily find similar image as input.

## requirements
[Numpy](https://numpy.org/)  
[Pillow](https://pillow.readthedocs.io/en/stable/#)  
[OpenCv](https://github.com/opencv/opencv-python)  
[scikit-learn (optional)](https://scikit-learn.org/stable/)

## Usage
### Create code book
In order to create BoVW, we need to import all features of all images and do clustering.
The center of each cluster will be code word, which construct a code book.  
To create codebook of database, run
```
create_codebook.py --root_path ROOT_PATH --result_path RESULT_PATH

optional arguments:
  --root_path ROOT_PATH
                        input image root dir path
  --result_path RESULT_PATH
                        output dir path
```

### Create bovw
After creating code book, create bovw of all images in database using code book.  
Make sure that you choose right code book file, and run
```
create_bovw.py --root_path ROOT_PATH --result_path RESULT_PATH --codebook_path CODEBOOK_PATH

optional arguments:
  --root_path ROOT_PATH
                        input image root dir path
  --result_path RESULT_PATH
                        output dir path
  --codebook_path CODEBOOK_PATH
                        codebook path
```

### Retrieve images
If you've done all preparation, then retrieve images that you want.  
Bring an sample image that you want to retrieve and run
```
find.py --result_path RESULT_PATH --codebook_path CODEBOOK_PATH --bovw_path BOVW_PATH --query_path QUERY_PATH [--num NUM] [--start_date START_DATE] [--end_date END_DATE]

optional arguments:
  --result_path RESULT_PATH
                        output dir path
  --codebook_path CODEBOOK_PATH
                        codebook path
  --bovw_path BOVW_PATH
                        bovw path
  --query_path QUERY_PATH
                        query image path
  --num NUM             number of images to retrieve
  --start_date START_DATE
                        Input start date like yyyy:mm:dd
  --end_date END_DATE   Input end date like yyyy:mm:dd
```