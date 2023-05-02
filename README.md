# Automatic_O2_Titration
This is a code to simulate automatic oxygen titration for medical applications.

## Video Tutorial
Watch this [final project video](https://youtu.be/RnVUVX2VLiw) for understanding the context for implementation of the code. 

## PID Controller
A PID (proportional integral derivative) controller is a mechanism that allows a process to be controlled by continuously measuring the error between a desired output and the actual value of a process variable. The controller can then adjust the inputs to continuously minimize the error and bring the output closer to a desired set point. This dynamic feedback allows for a rapid response to changes in the target variable.

The "proportional" aspect of the controller calculates the error between the setpoint and the actual process value and applies a correction that is proportional to the error. 

The "integral" aspect takes into account the past errors and applies a correction proportional to the accumulated error over time. 

The "derivative" aspect considers the rate of change of the error and applies a correction proportional to the rate of change.

## Code:

O2_PID_Controller.py- PID controller

simulated_HR.csv- CSV file of simulated "heart rate disturbance"





