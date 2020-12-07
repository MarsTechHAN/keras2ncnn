# keras2ncnn

### Now availabel on pypy!
### If you failed to convert a model, welcome to open an issue and attach the h5 file.

---
## Usage:
```
# Install keras2ncnn (only h5py and numpy is required)
python3 -mpip install --upgrade keras2ncnn

# If you only want to convert the model
python3 -m keras2ncnn -i SOME_H5DF_FILE.h5 -o ./  

# You can see the structure of the converted model and the original model(after optimization)
python3 -m keras2ncnn -i SOME_H5DF_FILE.h5 -o DIR_TO_SAVE_NCNN_PARAM --plot_graph/-p
```
---
## Supported Op
- InputLayer
- Conv2D 
- Conv2DTranspose 
- DepthwiseConv2D
- Add
- Multiply
- ZeroPadding2D
- ReLU
- LeakyReLU
- UpSampling2D
- Concatenate
- GlobalAveragePooling2D
- MaxAveragePooling2D
- AveragePooling2D
- MaxPooling2D
- BatchNormalization
- Dense
- Activation 

