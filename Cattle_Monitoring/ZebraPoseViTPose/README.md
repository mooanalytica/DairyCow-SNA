## ZebraPose fork of ViTPose

The [project website](https://zebrapose.is.tue.mpg.de/) contains all the data and additional information.

You can find all the modified files in ZebraPose folder.
Those include:
1. New validation files for the ap10k dataset, which is the main configuration used in the ZebraPose work. Those will now print PCK at three different threshold and the parts pck as well.
2. New config files for the synthetic and mixed datasets. Those are in the config folder and are available for testing, trian from scratch and train with a pretrained backbone. Please note that you will have to change paths and jsons in those files according to your naming/saving
3. Edits in the `tools/` files, including a minor change in `model_split.py`
4. Some additional scripts to automatically get and process the results, and get a pandas dataframe in output


Please copy all of them in the main folder.

To install ViTPose follow the README_ViT.md instructions. 

### License
By using this code you agree to the [license terms](https://zebrapose.is.tue.mpg.de/license.html)

### Citation
If you use this software, please cite 