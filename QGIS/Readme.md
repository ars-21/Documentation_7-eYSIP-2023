# QGIS and Python API Code Repository

This repository contains QGIS codes and Python API scripts for [Computer-vision-Remote-Sensing-and-GIS-application-for-autonmous-ground-vehicle]().

## Prerequisites

Before running the code in this repository, make sure you have the following prerequisites installed:

### QGIS Software

To download QGIS software, follow these steps:

1. Visit the QGIS download page at [https://www.qgis.org](https://www.qgis.org).
2. Select your operating system (Windows, macOS, Linux) and click on the appropriate download link.
3. Follow the installation instructions provided on the QGIS website for your operating system.

### Python

To download Python, follow these steps:

1. Visit the official Python website at [https://www.python.org](https://www.python.org).
2. Click on the "Downloads" tab in the top menu.
3. Choose the version of Python you want to download (e.g., Python 3.9).
4. Select the installer appropriate for your operating system (Windows, macOS, Linux).
5. Run the installer and follow the installation instructions provided.

After installing QGIS and Python, you will be ready to run the QGIS codes and Python API scripts in this repository.

# Georeferenced Image using QGIS Georeferencer Plugin

This repository contains a georeferenced image that has been aligned to real-world geographic coordinates using the QGIS Georeferencer plugin.

## Description

The georeferenced image in this repository has undergone the georeferencing process using the QGIS Georeferencer plugin. This plugin provides a user-friendly interface within QGIS to accurately align non-spatial images with known geographic locations.

## Georeferencing Details

To georeference the image using the QGIS Georeferencer plugin, follow these steps:

1. Launch QGIS and open the Georeferencer plugin from the "Raster" menu. It may depends on the version you are following, In previous versions it is available in "Layers" menu.

2. Load the non-spatial image into the Georeferencer plugin by clicking the "Open Raster" button and selecting the image file.

3. Identify control points on the non-spatial image that correspond to known locations on a reference dataset. Add control points by clicking on the corresponding locations in both the non-spatial image and the reference dataset.

4. Select an appropriate transformation method based on the characteristics of your data, such as polynomial or affine transformation.

5. Adjust any necessary transformation settings to fine-tune the alignment of the georeferenced image.

6. Evaluate the accuracy of the georeferenced image by comparing it to the reference dataset. Add additional control points or adjust the transformation as needed.

7. Once satisfied with the alignment, save the georeferenced image using the "File" menu in the Georeferencer plugin. Choose the appropriate file format and ensure that spatial referencing information is preserved.

Please note that the specific steps and options for georeferencing may vary depending on the GIS software you choose. It's important to refer to the documentation or tutorials provided by the software for detailed instructions specific to that software.

## Examples and Resources

If you need visual examples or further guidance on georeferencing, refer to the [examples]() folder in this repository. Additionally, the documentation and tutorials provided by the GIS software you are using can be valuable resources for understanding the georeferencing process.

# Geoprocessing using GDAL library
The ## Geospatial Data Abstraction Library (GDAL) is a library for reading and writing raster and vector geospatial data formats.
Various geospatial software(like QGIS, ARCGIS etc.) uses GDAL library in its back-end to do geoprocessing tasks.

Note: There is no Python specific reference documentation, but the GDAL API Tutorial 96 includes Python examples. So we will be using python api as well as GDAL commands in python script. You can refer here 82 for GDAL official documentation.
Some of the important commands and snippets that could be useful are:

1. For finding information of satellite image or aerial image.
!gdalinfo satellite.tif

After running this command in your terminal you will see all the details of the georeferenced image like Coordinate system, projection system used, number of bands, size of image, corner longitude and latitude co-ordinates.

2. Changing Co-ordinate Reference System

gdalwarp test.tif crs_updated.tif -t_srs "+proj=longlat +ellps=WGS84"

The above command is used to change the co-ordinate reference system and save the updated image file.

You can run this same command in python script using python subprocess module.


3. For finding latitude and longitude of every pixel

You can find the longitude and latitude values of all the pixels of the georeferenced satellite image using the above snippet.

4. Georeferencing image

The basic idea behind georeferencing an image is to define the relationship between the X and Y coordinates (essentially pixels) of the image, and latitude and longitude coordinates of where those pixels. Each of these matches is called a ground control point (GCP).
In task1B you used georeferencer plugin in QGIS for having the image georeferenced. The georeferencer plugin used GDAL library in its background to do the task. You can also generate this command from QGIS georeferencer GUI.

In QGIS after opening georeferencer GUI and matching the features of aerial and base image, Go to File>Generate GDAL Script, you will see the below command.

Command looks like:
gdal_translate
-gcp pixelx1 pixely1 longitude1 latitude1
-gcp pixelx2 pixely2 longitude2 latitude2
-gcp pixelx3 pixely3 longitude3 latitude3
-gcp pixelx4 pixely4 longitude4 latitude4
-of GTiff
map-original.jpg
map-with-gcps.tif

here pixelx1 is your x co-ordinate and pixely1 is the y co-ordinate of the image which is not georeferenced(aerial image in our case).
longitude and latitude are the geolocation from base image which is georeferenced.

Example command:

```
gdal_translate -gcp 4203.7 2347.0 4.2946 52.0825 -gcp 3830.5 1673.9 4.2744 52.0888 -gcp 5122.7
1611.8 4.3054 52.1025 -gcp 5522.5 2981.5 4.3371 52.0862 -gcp 3593.9 2691.5 4.2849 52.0710 -gcp
6560.4 4341.7 4.3844 52.0761 -gcp 4432.0 4929.8 4.3406 52.0464 -gcp 486.8 2458.7 4.2042 52.0447
-gcp 1171.9 4210.5 4.2483 52.0252 -gcp 6704.4 907.2 4.3332 52.1289 -gcp 1880.0 1993.3 4.2314
52.0650 -of GTiff map-original.jpg map-with-gcps.tif
```
The above command is used to assign longitude and latitude values to the image which is not georeferenced.

You can use the subprocess python module to run this command in python script.

Once georeferncing is done the image will not have the proper co-ordinate system. You need to change the co-ordinate reference system using the step 2.

## References:

[Georeferencing and digitizing old maps with GDAL](https://kokoalberti.com/articles/georeferencing-and-digitizing-old-maps-with-gdal/)

[Georeferencing Satellite Images](https://tilemill-project.github.io/tilemill/docs/guides/georeferencing-satellite-images/)

# Challenges Faced

### choosing the right coordinate system: 
Selecting the correct coordinate system for georeferencing is crucial. Ensure that you use the appropriate coordinate system that matches the spatial reference of the original data. If you are unsure, consult with GIS experts or refer to official data sources.

### GCP selection and accuracy: 
Choosing Ground Control Points (GCPs) that accurately represent real-world locations is essential. Ensure that the GCPs are well-distributed across the image and correspond to known, accurate coordinates. The more GCPs you use, the better the georeferencing accuracy.

### Residual errors and RMS values: 
After georeferencing, check the residual errors and root mean square (RMS) values. High RMS values indicate poor georeferencing accuracy. You may need to adjust GCPs or use additional GCPs to improve the results.

### Dealing with distortions: 
Sometimes, images may have distortions, such as barrel distortion or warping. You might need to use specialized transformation methods to handle these distortions effectively.

### Handling different image types: 
Different image types (raster, aerial imagery, scanned maps, etc.) may require specific georeferencing techniques. Be familiar with the appropriate methods for each image type.

### Changing landscape features: 
If the landscape features in the image have changed over time, georeferencing may be challenging. In such cases, historical maps or other data sources can be used to aid the process.

### Using appropriate software: 
Choose a reliable GIS software that provides georeferencing tools and supports the image format you are working with. Common software choices include QGIS, ArcGIS, and GDAL.

### Documentation and versioning: 
Keep detailed records of the georeferencing process, including the GCPs used, the transformation method, and any adjustments made. This documentation is helpful for validation and future reference.

### Quality assessment: 
Always visually inspect the georeferenced image over the base map or other data layers to ensure it aligns correctly and maintains its spatial integrity.

### Patience and practice: 
Georeferencing can be a trial-and-error process, especially when dealing with complex datasets. Practice and patience are key to improving your skills and overcoming challenges.
