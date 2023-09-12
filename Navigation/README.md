# Navigation Scripts and ArUco Detection

This repository contains scripts and code for navigation, ArUco marker detection, communication and driver functionality.

## Introduction

This project aims to provide a comprehensive solution for navigation using ArUco markers and controlling a robot or vehicle. The code in this repository enables you to perform ArUco marker detection, calculate robot navigation paths, and control the robot using driver code.

## Features

- ArUco marker detection
- Navigation path planning
- Robot control
- udp communication

## File Structure

The repository has the following structure:

- `aruco.py`: Python script for detecting ArUco markers in images or video streams.
- `controller.py`: Script for calculating navigation paths based on detected ArUco markers.
- `motor_driver.py`: Code for controlling the robot or vehicle based on the calculated navigation paths.
- `udp.py`: UDP communication between a client and server using socket programming


### ArUco Marker Detection
This repository contains scripts and code for ArUco marker detection using computer vision techniques. ArUco markers are 2D barcodes that are widely used in augmented reality and robotics applications for localization, tracking, and pose estimation.

## Introduction
This project aims to provide a comprehensive solution for detecting ArUco markers in images or video streams. The code in this repository utilizes OpenCV, a popular computer vision library, to perform marker detection and extraction.

## Features
Detection of ArUco markers in images or video streams.
Extraction of marker IDs and pose estimation.
Visualization of detected markers with bounding boxes.
Support for custom dictionary and marker configurations.


[aruco_detection.py](): Python script for detecting ArUco markers in images or video streams.

Open the [aruco_detection.py]() script and modify the configuration parameters as needed. 

You can specify the input image or video source, dictionary type, marker size, and other settings.

Run the [aruco_detection.py]() script to perform ArUco marker detection. 
The script will analyze the input and detect any visible markers.

The detected markers will be visualized with bounding boxes and pose estimation information. 



### UDP Communication
This repository contains code for UDP communication between a client and server using socket programming.

## Introduction
UDP (User Datagram Protocol) is a lightweight, connectionless protocol that allows for fast and efficient communication between network devices. This project demonstrates how to establish UDP communication between a client and server using socket programming in Python.

## Features
Establishing a UDP server to listen for incoming messages.
Creating a UDP client to send messages to the server.
Support for bi-directional communication between the client and server.
Handling multiple client connections simultaneously.

## Example file
[UDP.py]()

## Navigation Path Planning

Refer : [Path Planning]()

## Bot Control

# Motor Control

[controller.py]() contains code for controlling a motor using a motor driver and a controller

# Point-to-Point Cone Motor Control

This [controller.py]() file contains code for controlling a motor to move in a point-to-point fashion. The motor is controlled using a motor driver and a controller, and the movements are based on desired coordinates provided as goals.

## Setup

1. Ensure that the motor and the motor driver are properly connected and configured.
2. Make sure that the IP address and port for the UDP communication are set correctly in the code (`UDP_IP` and `UDP_PORT` variables).
3. Install the necessary dependencies for running the code.

## Usage

1. Run the code by executing the Python script.
2. The code will start listening to the `/aruco` topic for pose information, which includes the current position and orientation of the bot.
3. Specify the desired goal coordinates by modifying the `move` function with the desired `x_goal` and `y_goal` values.
4. The code will calculate the necessary motor control signals based on the current position and the goal coordinates.
5. The motor control signals will be sent via UDP communication to the motor driver to move the cone motor accordingly.
6. The bot will continue moving towards the goal until it reaches the specified goal coordinates (within a small tolerance).
7. Once the goal is reached, the code will exit gracefully.

## Dependencies

- ROS (Robot Operating System)
- Python 3
- `numpy` package

Driver code : [motor_driver.py]()
