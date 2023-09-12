# SAM: House Segmentation:
## Segmenting remote sensing imagery with text prompts using the Segment Anything Model (SAM)

[![image](https://colab.research.google.com/assets/colab-badge.svg)]()

This notebook is used to generate House object masks of specific classes from text prompts with the Segment Anything Model (SAM).

Make sure you use GPU runtime for this notebook. For Google Colab, go to `Runtime` -> `Change runtime type` and select `GPU` as the hardware accelerator.



## Table of Contents

- [SAM: House Segmentation](#sam-house-segmentation)
- [Table of Contents](#table-of-contents)
- [samgeo](#samgeo)
- [Geo-referenced Image](#geo-referenced-image)
- [Output](#output)
- [References](#references)



## samgeo

* A Python package for segmenting geospatial data with the Segment Anything Model (SAM)
* The segment-geospatial package draws its inspiration from segment-anything-eo repository authored by Aliaksandr Hancharenka. To facilitate the use of the Segment Anything Model (SAM) for geospatial data, I have developed the segment-anything-py and segment-geospatial Python packages, which are now available on PyPI and conda-forge. My primary objective is to simplify the process of leveraging SAM for geospatial data analysis by enabling users to achieve this with minimal coding effort. I have adapted the source code of segment-geospatial from the segment-anything-eo repository, and credit for its original version goes to Aliaksandr Hancharenka.

Interactive Segmentation using Text Prompts
![interactive samgeo](./assets/samgeo.gif)

🆓 Free software: MIT license

📖 Documentation: https://samgeo.gishub.org



## Geo-referenced Image:

Original Image                                  | Geo-referenced Image
:----------------------------------------------:|:---------------------------------------------------:
![original image](./assets/Original_Image.png)  | ![Geo-referenced Image](./assets/Georeferenced.png)



## Output:

Original Input Image                      |  House Segmentation
:----------------------------------------:|:-----------------------------------------:
![input image](./assets/Input_Image.png)  |  ![house segmentation on input image](./assets/House_Segmentation.png)

Original Input Image                      |  House Segmentation Mask
:----------------------------------------:|:----------------------------------------:
![input image](./assets/Input_Image.png)  |  ![house segmentation mask](./assets/House_Segmentation_Mask.png)



### References
* [segment-geospatial](https://samgeo.gishub.org/)
* [Segmenting remote sensing imagery with text prompts and the Segment Anything Model (SAM)](https://samgeo.gishub.org/examples/text_prompts/) [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/segment-geospatial/blob/main/docs/examples/text_prompts.ipynb)