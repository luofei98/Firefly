//
// main.cpp
// Firefly [v1]
//
// Copyright (c) 2015 Mihir Garimella.
//

#include <algorithm>
#include <limits.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <netinet/in.h>
#include <pthread.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

// Include UncannyCV, a vision library that includes optimizations for ARMv7-A processors.
extern "C" {
    #include <image_utils.h>
    #include <uv_canny.h>
    #include <uv_colorConversion.h>
    #include <uv_convolution_8u_3x3.h>
    #include <uv_dilate3x3.h>
    #include <uv_free_canny_buf.h>
    #include <uv_free_connectedcomps_buf.h>
    #include <uv_get_connectedcomps.h>
    #include <uv_init_canny_buf.h>
    #include <uv_init_connectedcomps_buf.h>
    #include <uv_lens_correction.h>
    #include <uv_lens_correction_ref.h>
    #include <uv_optical_flow.h>
    #include <uv_optical_flow_ref.h>
    #include <uv_vision.h>
}

// Include ROS libraries to communicate with the ARDrone.
#include <ardrone_autonomy/Navdata.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Twist.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <std_srvs/Empty.h>

// Include a few OpenCV modules for vision tasks that aren't implemented by UncannyCV.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// Include WiringPi, a wrapper for UART communication through the GPIO pins.
#include <wiringSerial.h>

using namespace cv;
using namespace std;

#define PI                              3.14159
#define CONVERT_TO_RAD(x)               ((x) * PI / 180)

// #define DEBUG_MAPPING
// #define DEBUG_PREPROCESSING
// #define DEBUG_NAVIGATION

#define TCP_PORT                        3000

#define FORWARD_VELOCITY                1.5 // (some arbitrary units set by ardrone_autonomy)
#define VELOCITY_TEMPERATURE_MULTIPLIER 0.45
#define FORWARD_VELOCITY_FLOOR          0.1 // (some arbitrary units set by ardrone_autonomy)
#define FORWARD_THRESHOLD               0.5 // m
#define SACCADE_VELOCITY                1.0 // // (some arbitrary units set by ardrone_autonomy)
#define ALIGNING_PROPORTIONAL_GAIN      0.0032 // (rad/s)/px

#define IMAGE_WIDTH                     640 // px
#define IMAGE_HEIGHT                    360 // px
#define ALPHA_X                         0.05
#define ALPHA_Y                         0.12

#define MAP_RESOLUTION                  4 // cm/item
#define MAP_SIZE                        (396 / MAP_RESOLUTION) // cm/(cm/item) = items
#define MINIMUM_GAP_WIDTH               (100 / MAP_RESOLUTION) // cm/(cm/item) = items

#define LOWER_H                         0
#define LOWER_S                         120
#define LOWER_V                         170
#define UPPER_H                         20
#define UPPER_S                         255
#define UPPER_V                         255
#define BLOB_TOLERANCE                  30 // px
#define MINIMUM_BLOB_AREA               150 // px^2
#define BLOB_NOT_FOUND_BEFORE_GIVE_UP   5 // frames

#define TEMPLATE_HEIGHT                 6 // px
#define SEED_SPACING                    20 // px
#define CANNY_THRESHOLD                 10
#define CANNY_MULTIPLIER                0.4
#define MORPHOLOGICAL_SIZE              3 // px
#define LK_WINDOW_SIZE                  15 // px
#define MAXIMUM_TRACK_SIZE              (2 * (IMAGE_WIDTH / SEED_SPACING) * (IMAGE_HEIGHT / SEED_SPACING)) // items
#define MINIMUM_EXPANSION               4 // px

#define MINIMUM_DISTANCE                0.25 // m
#define MAXIMUM_DISTANCE                10 // m
#define EMPTY_DISTANCE                  -1 // (units don't matter)
#define COMBINE_ESTIMATES_WHEN          5 // estimates
#define FIND_NEW_POINTS_AFTER           20 // frames
#define RANSAC_BASELINE                 0.2
#define RANSAC_MULTIPLIER               0.1
#define MAXIMUM_VERTICAL_SEPARATION     40 // px
#define OBSTACLE_AVOID_DISTANCE         0.5 // m

#define UART_NUMBER_OF_BITS             11 // bits

#define TEMPERATURE_FOUND_THRESHOLD     500 // * 0.1 degrees C
#define TEMPERATURE_SAFE_THRESHOLD      400 // * 0.1 degrees C
#define TEMPERATURE_DANGER_THRESHOLD    750 // * 0.1 degrees C

#define INCREASING_BEFORE_MAXIMUM       3 // sets of readings
#define DECREASING_AFTER_MAXIMUM        1 // sets of readings
#define CONSTANT_BEFORE_RESET           3 // sets of readings
#define DECREASING_BEFORE_SACCADE       6 // sets of readings
#define BLOB_DISPARITY_GOOD_ENOUGH      15 // px
#define BLOB_DISPARITY_MAXIMUM          150 // px
#define MINIMUM_TURN_ANGLE              45 // degrees

// Define types that we're going to use later on.
typedef struct {
    bool found;
    int left;
    int right;
    int width;
} Feature;

typedef enum {
    STATE_LANDED,
    STATE_HOVERING,
    STATE_ALIGNING,
    STATE_FORWARD,
    STATE_LEFT,
    STATE_RIGHT,
    STATE_STOPPED
} State;

typedef enum { 
    SENSOR_TEMPERATURE_FRONT,
    SENSOR_TEMPERATURE_BOTTOM,
    SENSOR_TEMPERATURE_GRADIENT,
    SENSOR_CONCENTRATION_LEFT,
    SENSOR_CONCENTRATION_RIGHT,
    SENSOR_CONCENTRATION_GRADIENT,
    SENSOR_RELATIVE_CONCENTRATION
} Sensor;

// Define mutex-protected global variables. Note that each mutex protects the variables directly above it.
State state = STATE_LANDED;
pthread_mutex_t stateMutex = PTHREAD_MUTEX_INITIALIZER;

bool newReadings = false;
int temperatureFront = 0;
int temperatureBottom = 0;
int increasing = 0;
int constant = 0;
int decreasing = 0;
int highestTemperature = 0;
int largeScaleDecreasing = 0;
bool higherConcentrationSide = false; // (false = left, true = right)
int concentrationLeft = 0;
int concentrationRight = 0;
pthread_mutex_t sensorMutex = PTHREAD_MUTEX_INITIALIZER;

int blobNotFoundFront = 1;
int integratedDisparity = 0;
int blobDisparity = 0;
pthread_mutex_t blobMutex = PTHREAD_MUTEX_INITIALIZER;

float vx = 0;
float dx = 0;
float dy = 0;
float yaw = 0;
long previousOdometryTime = 0;
float baselineYaw = 0;
float readableYaw = 0;
pthread_mutex_t navdataMutex = PTHREAD_MUTEX_INITIALIZER;

float baseMap[MAP_SIZE];
float latestMap[MAP_SIZE];
pthread_mutex_t mapMutex = PTHREAD_MUTEX_INITIALIZER;

bool newImage = false;
uv_image latestImage;
uv_image *latestPyramid;
char *latestPyramidBuffer;
int imageCounter = 0;
pthread_mutex_t imageMutex = PTHREAD_MUTEX_INITIALIZER;

ofstream debuggingFile;
int imageNumber = 0;
pthread_mutex_t debuggingMutex = PTHREAD_MUTEX_INITIALIZER;

// Define global variables that will only ever be used on one thread at a time.
uv_handle handle;
int *lookupTable;
unsigned short int *fractionTable;
int *indexTable;
uv_size windowSize;
char gaussian[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};

float maximumTurnAngle;
default_random_engine yawGenerator;
normal_distribution<double> normalDistribution(180.0, SIGMA_OBSTACLE_AVOIDANCE);

std_msgs::Empty emptyMessage;
std_srvs::Empty emptyRequest;
ros::Publisher hoverPublisher;
ros::Publisher resetPublisher;
ros::Publisher movePublisher;
ros::Publisher landPublisher;
ros::ServiceClient flatTrimClient;

//
// Main and helper functions for the communication thread (thread 0).
//

void *navigation(void *arg);
void *mapping(void *arg);

long time_usec();
long time_msec();
void log(string message);
State changeState(State state_new);
State copyState();
const char *printState(State state_to_print);
void receivedImageFront(const sensor_msgs::ImageConstPtr& message);
int findFire(Mat image);
void receivedNavdata(const ardrone_autonomy::Navdata &message);
void uv_convertFromMat(Mat image, uv_image *converted);
Mat uv_convertToMat(uv_image *image);
void uv_deepCopy(uv_image *dst, uv_image *src);
void moveDrone(float vx, float vy, float vz, float ax, float ay, float az);

int main(int argc, char **argv)
{
    debuggingFile.open("debugging.txt");

    // Initialize UncannyCV.
    uv_initialize(&handle, (unsigned char *)("licence.lic"), (unsigned char *)("licence.sig"), 0, NULL);
    uv_image randomImage;
    uv_RandomImage(&randomImage, IMAGE_WIDTH, IMAGE_HEIGHT, 1, 1);
    lookupTable = (int *)(malloc(sizeof(int) * IMAGE_WIDTH * IMAGE_HEIGHT));
    fractionTable = (unsigned short int *)(malloc(sizeof(unsigned short int) * IMAGE_WIDTH * IMAGE_HEIGHT));
    indexTable = (int *)(malloc(sizeof(int) * IMAGE_WIDTH * IMAGE_HEIGHT));
    createLookupTables(&randomImage, ALPHA_X, ALPHA_Y, lookupTable, indexTable, fractionTable, IMAGE_WIDTH * IMAGE_HEIGHT);
    uv_release_image(&randomImage);
    windowSize = uvSize(LK_WINDOW_SIZE, LK_WINDOW_SIZE);

    // Initialize this ROS node.
    ros::init(argc, argv, "sensor_platform");
    ros::NodeHandle node;
    resetPublisher = node.advertise<std_msgs::Empty>("ardrone/reset", 1);
    hoverPublisher = node.advertise<std_msgs::Empty>("ardrone/takeoff", 1);
    movePublisher = node.advertise<geometry_msgs::Twist>("cmd_vel", 1);
    landPublisher = node.advertise<std_msgs::Empty>("ardrone/land", 1);
    flatTrimClient = node.serviceClient<std_srvs::Empty>("ardrone/flattrim");
    ros::Subscriber navdataSubscription = node.subscribe("ardrone/navdata", 1, receivedNavdata);
    image_transport::ImageTransport imageStream(node);
    image_transport::Subscriber imageSubscriptionFront = imageStream.subscribe("ardrone/front/image_raw", 1, receivedImageFront);

    // Seed a random number generator that we can use later on.
    srand(time(0));

    // Start a TCP socket for high level commands and wait for the client to connect.
    sockaddr_in server_address, client_address;
    int socketFileDescriptor = socket(AF_INET, SOCK_STREAM, 0);
    int reuseAddress = 1;
    setsockopt(socketFileDescriptor, SOL_SOCKET, SO_REUSEADDR, &reuseAddress, sizeof(int)); // NEW
    if (socketFileDescriptor < 0) {
        perror("Couldn't open socket");
        return -1;
    } else {
        printf("Opened socket.\n");
    }
    memset((char *)&server_address, 0, sizeof(server_address));
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = INADDR_ANY;
    server_address.sin_port = htons(TCP_PORT);
    if (::bind(socketFileDescriptor, (sockaddr *)&server_address, sizeof(server_address)) < 0) {
        perror("Couldn't bind to socket");
        return -1;
    } else {
        printf("Bound to socket.\nWaiting for connections.\n");
    }
    listen(socketFileDescriptor, 1);
    socklen_t client_length = sizeof(client_address);
    int serverFileDescriptor = accept(socketFileDescriptor, (sockaddr *)&client_address, &client_length);
    if (serverFileDescriptor < 0) {
        perror("Couldn't accept on socket");
        return -1;
    }
    char command_buffer[1];

    // Start serial communication with the sensor module.
    int uart = serialOpen("/dev/ttyS2", 115200);

    // Define a local variable that we're going to use later on.
    long previousLoopTime = time_usec();

    // Create threads for the navigation and mapping tasks.
    pthread_t thread1, thread2;
    pthread_create(&thread1, NULL, navigation, NULL);
    pthread_create(&thread2, NULL, mapping, NULL);

    printf("Ready.\n\n");

    // Run the communication tasks on thread 0.
    while (true) {
        // Clear the ROS message queue, processing any messages with the appropriate callbacks if necessary.
        ros::spinOnce();

        // Read commands from the TCP socket.
        bool newCommand = false;
        fd_set readFileDescriptors;
        FD_ZERO(&readFileDescriptors);
        FD_SET(serverFileDescriptor, &readFileDescriptors);
        timeval pollTimer;
        pollTimer.tv_sec = 0;
        pollTimer.tv_usec = 0;
        int select_status = select(serverFileDescriptor + 1, &readFileDescriptors, NULL, NULL, &pollTimer);
        if (select_status > 0) {
            int read_status = read(serverFileDescriptor, command_buffer, 1);
            if (read_status > 0) {
                newCommand = true;
            } else if (read_status == 0) {
                printf("Disconnected from client.\n");
                changeState(STATE_STOPPED);
                break;
            } else {
                perror("Couldn't read from socket");
            }
        }

        // Read sensor readings over serial.
        while (serialDataAvail(uart) > UART_NUMBER_OF_BITS) { serialGetchar(uart); }
        if (serialDataAvail(uart) == UART_NUMBER_OF_BITS) {
            int temperatureFront_previous;
            int temperatureBottom_previous;
            for (Sensor sensor = SENSOR_TEMPERATURE_FRONT; sensor <= SENSOR_RELATIVE_CONCENTRATION; sensor = (Sensor)((int)(sensor) + 1)) {
                int value;
                if (sensor == SENSOR_TEMPERATURE_FRONT || sensor == SENSOR_TEMPERATURE_BOTTOM || sensor == SENSOR_CONCENTRATION_LEFT || sensor == SENSOR_CONCENTRATION_RIGHT) {
                    int upper = serialGetchar(uart);
                    int lower = serialGetchar(uart);
                    if (upper >= 0 && lower >= 0) {
                        value = ((uint8_t)(upper) << 8) | (uint8_t)(lower);
                    } else {
                        break;
                    }
                } else {
                    value = serialGetchar(uart);
                    if (value < 0) break;
                }
                pthread_mutex_lock(&sensorMutex);
                switch(sensor) {
                    case SENSOR_TEMPERATURE_FRONT:
                        temperatureFront_previous = temperatureFront;
                        temperatureFront = value;
                        break;
                    case SENSOR_TEMPERATURE_BOTTOM:
                        temperatureBottom_previous = temperatureBottom;
                        temperatureBottom = value;
                        break;
                    case SENSOR_TEMPERATURE_GRADIENT:
                        // Keep track of the sign of the derivative of temperature readings. Stop the counters once they've reached the threshold that we're going to use to prevent them from overflowing.
                        if (value == 0 /* Derivative is negative. */) {
                            if (decreasing < DECREASING_AFTER_MAXIMUM) {
                                if (++decreasing == 1) {
                                    // If we've reached a maximum in temperature, record the highest temperature from the current and previous sets of sensor readings.
                                    highestTemperature = temperatureFront_previous;
                                    if (temperatureBottom_previous > highestTemperature) highestTemperature = temperatureBottom_previous;
                                    if (temperatureFront > highestTemperature) highestTemperature = temperatureFront;
                                    if (temperatureBottom > highestTemperature) highestTemperature = temperatureBottom;
                                }
                                constant = 0;
                            }
                        } else if (value == 1 /* Derivative is zero. */) {
                            if (++constant == CONSTANT_BEFORE_RESET) {
                                increasing = 0;
                                constant = 0;
                                decreasing = 0;
                            }
                        } else /* Derivative is positive. */ {
                            if (decreasing > 0) {
                                decreasing = 0;
                                increasing = 1;
                            } else if (increasing < INCREASING_BEFORE_MAXIMUM) {
                                increasing++;
                            }
                            constant = 0;
                        }
                        newReadings = true;
                        break;
                    case SENSOR_CONCENTRATION_LEFT:
                        concentrationLeft = value;
                        break;
                    case SENSOR_CONCENTRATION_RIGHT:
                        concentrationRight = value;
                        break;
                    case SENSOR_CONCENTRATION_GRADIENT:
                        // Keep track of how long concentration readings have been decreasing. Stop the counter once it's reached the threshold that we're going to use to prevent it from overflowing.
                        if (value == 0) {
                            if (largeScaleDecreasing < DECREASING_BEFORE_SACCADE) {
                                largeScaleDecreasing++;
                            }
                        } else if (value == 2) {
                            largeScaleDecreasing = 0;
                        }
                        break;
                    case SENSOR_RELATIVE_CONCENTRATION:
                        higherConcentrationSide = (bool)(value);
                        break;
                }
                pthread_mutex_unlock(&sensorMutex);
            }
        }

        State state_copy = copyState();
        if (newCommand) {
            // If we've received any new commands, handle them accordingly.
            if (command_buffer[0] == 'f' && state_copy == STATE_LANDED /* Flat trim. */) {
                flatTrimClient.call(emptyRequest);
            } else if (command_buffer[0] == 'r' && state_copy == STATE_LANDED) {
                resetPublisher.publish(emptyMessage);
            } else if (command_buffer[0] == 't' && state_copy == STATE_LANDED /* Take off. */) {
                state_copy = changeState(STATE_HOVERING);
            } else if (command_buffer[0] == 'm' && state_copy == STATE_HOVERING /* Start to move. */) {
                state_copy = changeState(STATE_ALIGNING);
            } else if (command_buffer[0] == 'l' && state_copy != STATE_LANDED /* Land. */) {
                state_copy = changeState(STATE_LANDED);
            } else if (command_buffer[0] == 's' /* Emergency stop. */) {
                changeState(STATE_STOPPED);
                break;
            }
        }

        // Run this loop at a frequency of 250 Hz.
        int loopTime = time_usec() - previousLoopTime;
        if (loopTime < 4000) {
            usleep(4000 - loopTime);
        }
        previousLoopTime = time_usec();
    }

    // Close all file descriptors gracefully.
    close(serverFileDescriptor);
    serialClose(uart);

    // Wait for the other threads to terminate.
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    uv_deinitialize(handle);

    pthread_mutex_lock(&debuggingMutex);
    debuggingFile.close();
    pthread_mutex_unlock(&debuggingMutex);

    return 0;
}

long time_msec()
{
    // Return the system time in milliseconds.
    timeval time;
    gettimeofday(&time, NULL);
    return (time.tv_sec * 1000) + (time.tv_usec / 1000);
}

long time_usec()
{
    // Return the system time in microseconds.
    timeval time;
    gettimeofday(&time, NULL);
    return (time.tv_sec * 1000000) + (time.tv_usec);
}

State changeState(State state_new)
{
    State state_old = copyState();
    if (state_new != state_old) {
        #ifdef DEBUG_NAVIGATION
        printf("%s -> %s\n", printState(state_old), printState(state_new));
        #endif

        // Send the appropriate command to the ARDrone (or generally take the appropriate action) based on our new state.
        if (state_new == STATE_LANDED || state_new == STATE_STOPPED) {
            landPublisher.publish(emptyMessage);
        } else if (state_new == STATE_HOVERING || state_new == STATE_ALIGNING) {
            hoverPublisher.publish(emptyMessage);
            moveDrone(0, 0, 0, 0, 0, 0);
        } else if (state_new == STATE_FORWARD) {
            hoverPublisher.publish(emptyMessage);
            moveDrone(0, 0, 0, 0, 0, 0);

            // Empty both copies of the map before starting forward flight.
            pthread_mutex_lock(&mapMutex);
            fill_n(baseMap, sizeof(baseMap) / sizeof(baseMap[0]), EMPTY_DISTANCE);
            fill_n(latestMap, sizeof(latestMap) / sizeof(latestMap[0]), EMPTY_DISTANCE);
            pthread_mutex_unlock(&mapMutex);
        } else if (state_new == STATE_LEFT || state_new == STATE_RIGHT) {
            moveDrone(0, 0, 0, 0, 0, (state_new == STATE_RIGHT) ? SACCADE_VELOCITY : -SACCADE_VELOCITY);
            pthread_mutex_lock(&navdataMutex);
            baselineYaw = yaw - 2;
            pthread_mutex_unlock(&navdataMutex);
        } else if (state_new == STATE_RIGHT) {
            moveDrone(0, 0, 0, 0, 0, SACCADE_VELOCITY);
        }

        // Reset the integrated odometry measurements.
        pthread_mutex_lock(&navdataMutex);
        previousOdometryTime = 0;
        dx = 0;
        dy = 0;
        pthread_mutex_unlock(&navdataMutex);

        // Reset counters that we use to keep track of sensor readings.
        pthread_mutex_lock(&sensorMutex);
        newReadings = false;
        increasing = 0;
        constant = 0;
        decreasing = 0;
        largeScaleDecreasing = 0;
        pthread_mutex_unlock(&sensorMutex);

        pthread_mutex_lock(&blobMutex);
        blobNotFoundFront = 1;
        pthread_mutex_unlock(&blobMutex);

        // Change the global state in a thread-safe way.
        pthread_mutex_lock(&stateMutex);
        state = state_new;
        pthread_mutex_unlock(&stateMutex);
    }
    return state_new;
}

State copyState()
{
    // Make a copy of the global state in a thread-safe way.
    State copy;
    pthread_mutex_lock(&stateMutex);
    copy = state;
    pthread_mutex_unlock(&stateMutex);
    return copy;
}

const char *printState(State state_to_print)
{
    // Print the name of the given state.
    switch(state_to_print) {
        case STATE_LANDED:
            return "STATE_LANDED";
        case STATE_HOVERING:
            return "STATE_HOVERING";
        case STATE_ALIGNING:
            return "STATE_ALIGNING";
        case STATE_FORWARD:
            return "STATE_FORWARD";
        case STATE_LEFT:
            return "STATE_LEFT";
        case STATE_RIGHT:
            return "STATE_RIGHT";
        case STATE_STOPPED:
            return "STATE_STOPPED";
        default:
            return "";
    }
}

void receivedImageFront(const sensor_msgs::ImageConstPtr &message)
{
    Mat raw = cv_bridge::toCvShare(message, "bgr8")->image;
    if (!raw.empty() && raw.rows == IMAGE_HEIGHT && raw.cols == IMAGE_WIDTH) {
        #ifdef DEBUG_PREPROCESSING
        long start = time_usec();
        #endif

        // Convert the image to a uv_image.
        uv_image image, gray, undistorted;
        uv_convertFromMat(raw, &image);

        // Convert the image to grayscale.
        uv_create_image(IMAGE_WIDTH, IMAGE_HEIGHT, UV_GRAY, &gray);
        uv_rgb2gray(handle, &image, &gray);
        uv_release_image(&image);

        // Warp the image to remove lens distortion.
        uv_create_image(IMAGE_WIDTH, IMAGE_HEIGHT, UV_GRAY, &undistorted);
        uv_lens_correction(handle, indexTable, fractionTable, &gray, &undistorted);
        uv_release_image(&gray);

        // Blur the image to remove noise.
        uv_convolution_8u_3x3(handle, &undistorted, &undistorted, gaussian, 4);

        // Compute image pyramids for optical flow computation later on.
        char *pyramidBuffer;
        uv_image *pyramid = uv_initialize_opticalflowBuf(&undistorted, windowSize, 3, &pyramidBuffer);
        uv_buildOpticalFlowPyramid(handle, &undistorted, pyramid, windowSize, 3, 1, 0);
        
        // Store the warped and blurred image, along with the original OpenCV image and pointers to the pyramid. 
        pthread_mutex_lock(&imageMutex);
        uv_deepCopy(&latestImage, &undistorted);
        if (newImage) {
            // Deallocate the previous pyramid only if it will never be used.
            free(latestPyramid);
            free(latestPyramidBuffer);
        }
        latestPyramid = pyramid;
        latestPyramidBuffer = pyramidBuffer;
        newImage = true;
        pthread_mutex_unlock(&imageMutex);

        #ifdef DEBUG_PREPROCESSING
        printf("Finished preprocessing steps in %ld us.\n", time_usec() - start);
        #endif

        // Use an appearance-based classifier to search for potential targets in this image.
        int blobPosition = findFire(raw);

        // Save this image, along with sensor readings and other relevant variables, to a debugging file.
        pthread_mutex_lock(&sensorMutex);
        int temperatureBottom_copy = temperatureBottom;
        int temperatureFront_copy = temperatureFront;
        int concentrationLeft_copy = concentrationLeft;
        int concentrationRight_copy = concentrationRight;
        pthread_mutex_unlock(&sensorMutex);

        int state_copy = (int)(copyState());

        pthread_mutex_lock(&navdataMutex);
        float yaw_copy = yaw;
        float vx_copy = vx;
        pthread_mutex_unlock(&navdataMutex);

        pthread_mutex_lock(&debuggingMutex);
        string filename = "/home/odroid/node/image" + to_string(imageNumber) + ".png";
        debuggingFile << temperatureFront_copy << "," << temperatureBottom_copy << "," << concentrationLeft_copy << "," << concentrationRight_copy << "," << yaw_copy << "," << vx_copy << "," << blobPosition << "," << state_copy << "," << imageNumber << endl;
        imageNumber++;
        pthread_mutex_unlock(&debuggingMutex);

        imwrite(filename, raw);
    } else {
        #ifdef DEBUG_PREPROCESSING
        printf("Error on receivedImageFront().\n");
        #endif
    }
    raw.release();
}

int findFire(Mat image)
{
    // Use a simple color-based classifier to search for bright orange blobs that could be the fire.
    Mat hsv, thresholded;
    vector<vector<Point> > blobs;
    cvtColor(image, hsv, CV_BGR2HSV);
    inRange(hsv, Scalar(LOWER_H, LOWER_S, LOWER_V), Scalar(UPPER_H, UPPER_S, UPPER_V), thresholded);
    findContours(thresholded, blobs, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    // Find the index of the largest blob, using Hu moments to calculate the area of each blob that we found.
    float area_max = 0;
    int largestBlobIndex = 0;
    for (int b = 0; b < blobs.size(); b++) {
        float area = contourArea(blobs[b], false);
        if (area > area_max) {
            area_max = area;
            largestBlobIndex = b;
        }
    }
    
    // Find the bounding box of the largest blob.
    int blobPosition = -1;
    if (area_max > MINIMUM_BLOB_AREA) {
        int x_min = numeric_limits<int>::max();
        int x_max = 0;
        int y_min = numeric_limits<int>::max();
        int y_max = 0;
        for (int p = 0; p < blobs[largestBlobIndex].size(); p++) {
            Point point = blobs[largestBlobIndex][p];
            if (point.x < x_min) x_min = point.x;
            if (point.x > x_max) x_max = point.x;
            if (point.y < y_min) y_min = point.y;
            if (point.y > y_max) y_max = point.y;
        }

        blobPosition = (x_min + x_max) / 2;

        // Store the position of the blob.
        pthread_mutex_lock(&blobMutex);
        blobDisparity = (IMAGE_WIDTH - (x_min + x_max)) / 2;
        blobNotFoundFront = 0;
        pthread_mutex_unlock(&blobMutex);
    } else {
        // Note that we haven't found a blob in the latest image.
        pthread_mutex_lock(&blobMutex);
        if (blobNotFoundFront <= BLOB_NOT_FOUND_BEFORE_GIVE_UP) {
            blobNotFoundFront++;
        }
        pthread_mutex_unlock(&blobMutex);
    }

    return blobPosition;
}

void receivedNavdata(const ardrone_autonomy::Navdata &message)
{
    State state_copy = copyState();
    if (state_copy != STATE_LANDED) {
        // Maintain odometry (in meters) by integrating the ARDrone's optical flow estimates.
        pthread_mutex_lock(&navdataMutex);
        if (previousOdometryTime != 0) {
            long elapsedTime = time_usec() - previousOdometryTime;
            dx += (elapsedTime * message.vx) / 1000000000.f;
            dy += (elapsedTime * message.vy) / 1000000000.f;
        }
        previousOdometryTime = time_usec();
        vx = message.vx;
        
        // Store the ARDrone's yaw.
        yaw = message.rotZ;
        readableYaw = yaw - baselineYaw;
        if (readableYaw < 0) {
            readableYaw += 360;
        } else if (readableYaw > 360) {
            readableYaw -= 360;
        }
        pthread_mutex_unlock(&navdataMutex);
    }
}

void uv_convertFromMat(Mat image, uv_image *converted)
{
    // Convert an OpenCV Mat to an UncannyCV uv_image.
    int size = image.rows * image.cols * image.channels();
    converted->height = image.rows;
    converted->width = image.cols;
    converted->channel_data[0].pdata = (unsigned char *)(malloc(size * sizeof(unsigned char)));
    memcpy(converted->)
    converted->channel_data[0].stride = image.step[0];
    if (image.type() == CV_8UC1) {
        converted->colourSpace = UV_GRAY;
    } else if (image.type() == CV_8UC3) {
        converted->colourSpace = UV_RGB24;
    }
}

Mat uv_convertToMat(uv_image *image)
{
    // Convert an UncannyCV uv_image to an OpenCV Mat.
    Mat converted(Size(image->width, image->height), image->colourSpace == UV_GRAY ? CV_8UC1 : CV_8UC3, image->channel_data[0].pdata, Mat::AUTO_STEP);
    return converted;
}

void uv_deepCopy(uv_image *dst, uv_image *src)
{
    // Efficiently make a deep copy of an uv_image.
    uv_release_image(dst);
    int size = src->height * src->width * (src->colourSpace == UV_GRAY ? 1 : 3);
    dst->height = src->height;
    dst->width = src->width;
    dst->channel_data[0].pdata = (unsigned char *)(malloc(size * sizeof(unsigned char)));
    memcpy(dst->channel_data[0].pdata, src->channel_data[0].pdata, size * sizeof(unsigned char));
    dst->channel_data[0].stride = src->channel_data[0].stride;
    dst->colourSpace = src->colourSpace;
    uv_release_image(src);
}

void moveDrone(float vx, float vy, float vz, float ax, float ay, float az)
{
    // Command the ARDrone to move with given linear (vx, vy, vz) and angular (az) velocities.
    geometry_msgs::Twist moveMessage;
    moveMessage.linear.x = vx;
    moveMessage.linear.y = vy;
    moveMessage.linear.z = vz;
    moveMessage.angular.x = ax;
    moveMessage.angular.y = ay;
    moveMessage.angular.z = az;
    movePublisher.publish(moveMessage);
}

//
// Main and helper functions for the navigation task (thread 1).
//

bool shouldAvoidObstacle();
void avoidObstacle();
State beginSaccade();

void *navigation(void *arg)
{
    // Initialize some local variables that we're going to use later on.
    long previousLoopTime = time_usec();
    int verifyCounter = 0;
    bool alignedWithBlob = false;

    while (true) {
        // Make copies of mutex-protected variables that we're going to use frequently in this loop.
        State state_copy = copyState();

        pthread_mutex_lock(&blobMutex);
        int blobNotFoundFront_copy = blobNotFoundFront;
        int blobDisparity_copy = blobDisparity;
        pthread_mutex_unlock(&blobMutex);

        pthread_mutex_lock(&sensorMutex);
        int temperatureFront_copy = temperatureFront;
        int temperatureBottom_copy = temperatureBottom;
        int increasing_copy = increasing;
        int decreasing_copy = decreasing;
        int highestTemperature_copy = highestTemperature;
        int largeScaleDecreasing_copy = largeScaleDecreasing;
        bool newReadings_copy = newReadings;
        newReadings = false;
        pthread_mutex_unlock(&sensorMutex);

        pthread_mutex_lock(&navdataMutex);
        float readableYaw_copy = readableYaw;
        pthread_mutex_unlock(&navdataMutex);

        // Run the core navigation algorithm. See my project logs for more details.
        if (state_copy == STATE_FORWARD) {
            if (newReadings_copy) {
                if (temperatureFront_copy >= TEMPERATURE_DANGER_THRESHOLD) {
                    state_copy = changeState(STATE_LANDED);
                    #ifdef DEBUG_NAVIGATION
                    printf("Temperature reached danger threshold, assuming that we've found the fire.\n");
                    #endif
                } else if (alignedWithBlob && increasing_copy >= INCREASING_BEFORE_MAXIMUM && decreasing_copy >= DECREASING_AFTER_MAXIMUM) {
                    #ifdef DEBUG_NAVIGATION
                    printf("Found potential target with temperature %i\n", highestTemperature_copy);
                    #endif
                    if (highestTemperature_copy >= TEMPERATURE_FOUND_THRESHOLD) {
                        state_copy = changeState(STATE_LANDED);
                    } else {
                        pthread_mutex_lock(&sensorMutex);
                        constant = 0;
                        decreasing = 0;
                        pthread_mutex_unlock(&sensorMutex);
                    }
                } else if (largeScaleDecreasing_copy >= DECREASING_BEFORE_SACCADE) {
                    state_copy = beginSaccade();
                }
            }

            if (state_copy == STATE_FORWARD && !shouldAvoidObstacle()) {
                float forwardVelocity = FORWARD_VELOCITY - (VELOCITY_TEMPERATURE_MULTIPLIER * floor(temperatureFront_copy + temperatureBottom_copy) / 200.0);// / (increasing_copy + 1);
                if (forwardVelocity < FORWARD_VELOCITY_FLOOR) forwardVelocity = FORWARD_VELOCITY_FLOOR;
                if (blobNotFoundFront_copy == 0) {
                    if (abs(blobDisparity_copy) > BLOB_DISPARITY_MAXIMUM) {
                        if (!alignedWithBlob) {
                            changeState(STATE_ALIGNING);
                        }
                    } else {
                        moveDrone(forwardVelocity, 0, 0, 0, 0, ALIGNING_PROPORTIONAL_GAIN * blobDisparity_copy);
                        alignedWithBlob = true;
                    }
                } else {
                    moveDrone(forwardVelocity, 0, 0, 0, 0, 0);
                }
            }
        } else if (state_copy == STATE_LEFT || state_copy == STATE_RIGHT) {
            if (blobNotFoundFront_copy == 0 && readableYaw_copy >= (state_copy == STATE_LEFT ? MINIMUM_TURN_ANGLE : (360 - MINIMUM_TURN_ANGLE))) {
                changeState(STATE_ALIGNING);
            } else if (blobNotFoundFront_copy != 0 && readableYaw_copy >= (state_copy == STATE_LEFT ? maximumTurnAngle : (360 - maximumTurnAngle))) {
                changeState(STATE_FORWARD);
            }
        } else if (state_copy == STATE_ALIGNING) {
            if (blobNotFoundFront_copy == 0 && abs(blobDisparity_copy) > BLOB_DISPARITY_GOOD_ENOUGH) {
                moveDrone(0, 0, 0, 0, 0, ALIGNING_PROPORTIONAL_GAIN * blobDisparity_copy);
            } else if (blobNotFoundFront_copy > BLOB_NOT_FOUND_BEFORE_GIVE_UP) {
                alignedWithBlob = false;
                changeState(STATE_FORWARD);
            } else if (blobNotFoundFront_copy == 0 && abs(blobDisparity_copy) <= BLOB_DISPARITY_GOOD_ENOUGH) {
                alignedWithBlob = true;
                changeState(STATE_FORWARD);
            } else {
                moveDrone(0, 0, 0, 0, 0, 0);
            }
        } else if (state_copy == STATE_STOPPED) {
            break;
        }

        // Run this loop at a frequency of 40 Hz.
        int loopTime = time_usec() - previousLoopTime;
        if (loopTime < 25000) {
            usleep(25000 - loopTime);
        }
        previousLoopTime = time_usec();
    }

    return (void *)0;
}

bool shouldAvoidObstacle()
{
    pthread_mutex_lock(&navdataMutex);
    double dx_copy = dx;
    double dy_copy = dy;
    pthread_mutex_unlock(&navdataMutex);

    // Search the map for obstacles directly in the path of the robot.
    int start = ((MAP_SIZE - 1) / 2) + round(100 * dy_copy / MAP_RESOLUTION) - (MINIMUM_GAP_WIDTH / 2);
    if (start < 0 || start + (MINIMUM_GAP_WIDTH / 2) >= MAP_SIZE /* We've drifted outside the range of the local map. */) {
        // Because we no longer have an accurate map, assume that there's an obstacle and react accordingly.
        avoidObstacle();
        #ifdef DEBUG_NAVIGATION
        printf("%g m horizontal drift\n", dy_copy);
        #endif
        return true;
    } else {
        pthread_mutex_lock(&mapMutex);
        // If there's an obstacle within the search area, avoid it.
        for (int x = start; x < start + (MINIMUM_GAP_WIDTH / 2); x++) {
            if (latestMap[x] < dx_copy + OBSTACLE_AVOID_DISTANCE && latestMap[x] != EMPTY_DISTANCE) {
                avoidObstacle();
                #ifdef DEBUG_NAVIGATION
                printf("Avoiding an obstacle.\n");
                #endif
                return true;
            }
        }
        pthread_mutex_unlock(&mapMutex);
    }

    return false;
}

void avoidObstacle()
{
    // Avoid an obstacle in a random direction.
    if (rand() % 2) {
        changeState(STATE_LEFT);
    } else {
        changeState(STATE_RIGHT);
    }

    // To encourage exploration, choose the turn angle according to a normal distribution.
    maximumTurnAngle = normalDistribution(yawGenerator);
}

State beginSaccade()
{
    // Begin a saccade in the direction of higher concentration.
    pthread_mutex_lock(&sensorMutex);
    bool higherConcentrationSide_copy = higherConcentrationSide;
    pthread_mutex_unlock(&sensorMutex);

    maximumTurnAngle = 90;
    if (higherConcentrationSide_copy) {
        return changeState(STATE_RIGHT);
    } else {
        return changeState(STATE_LEFT);
    }
}

//
// Main and helper functions for the mapping task (thread 2).
//

Feature findFeature(int x_point, vector<vector<Point> > contours);
float widthToDistance(int left, int right);
void insertion_sort(vector<float> &v);

void *mapping(void *arg)
{
    // Define some local variables that we're going to use later on.
    long previousTime;

    char dilation[9] = {0,1,1,0,1,1,0,1,1};

    uv_TermCriteria terminationCriteria;
    terminationCriteria.type = 3;
    terminationCriteria.max_iter = 7;
    terminationCriteria.epsilon = 0.3;

    int frame = 0;
    #ifdef DEBUG_MAPPING
    int saved = 0;
    #endif
    uv_image image_previous;
    uv_image *pyramid_previous;
    char *pyramidBuffer_previous;
    double dx_previous = 0;

    vector<vector<float> > distances;
    UvPoint2D32f initialTrack[MAXIMUM_TRACK_SIZE];
    UvPoint2D32f track[MAXIMUM_TRACK_SIZE];
    vector<int> indices;
    int pointsToTrack;
    bool firstFrame = true;

    while (true) {
        // Run the core mapping algorithm during segments of forward flight. See my project logs for more details.
        State state_copy = copyState();
        if (state_copy == STATE_FORWARD) {
            pthread_mutex_lock(&imageMutex);
            bool newImage_copy = false;
            uv_image image;
            uv_image *pyramid;
            char *pyramidBuffer;
            if (newImage) {
                uv_deepCopy(&image, &latestImage);
                pyramid = latestPyramid;
                pyramidBuffer = latestPyramidBuffer;
                newImage_copy = true;
                newImage = false;
            }
            pthread_mutex_unlock(&imageMutex);
            if (newImage_copy) {
                previousTime = time_usec();

                pthread_mutex_lock(&navdataMutex);
                double dx_copy = dx;
                double dx_latest = dx - dx_previous;
                double dy_copy = dy;
                pthread_mutex_unlock(&navdataMutex);
                if (frame == 0) {
                    dx_previous = dx_copy;
                } else if (frame == 1 && dx_latest > FORWARD_THRESHOLD) {
                    frame++;
                }

                if (frame == 0) {
                    uv_image edges;
                    uv_create_image(IMAGE_WIDTH, IMAGE_HEIGHT, UV_GRAY, &edges);
                    unsigned char *cannyBuffer;
                    uv_init_canny_buf(&image, &cannyBuffer);
                    uv_canny(handle, &image, &edges, CANNY_MULTIPLIER * CANNY_THRESHOLD, CANNY_THRESHOLD, cannyBuffer);
                    uv_dilate3x3(handle, &edges, &edges, dilation);

                    indices.clear();
                    distances.clear();
                    pointsToTrack = 0;

                    Mat convertedEdges = uv_convertToMat(&edges);
                    for (int y = SEED_SPACING / 2; y < IMAGE_HEIGHT; y += SEED_SPACING) {
                        Mat edgeSlice = convertedEdges(Rect(0, y - (TEMPLATE_HEIGHT / 2), IMAGE_WIDTH, TEMPLATE_HEIGHT));
                        vector<vector<Point> > contours;
                        findContours(edgeSlice, contours, noArray(), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

                        for (int x = SEED_SPACING / 2; x < IMAGE_WIDTH; x += SEED_SPACING) {
                            Feature latestFeature = findFeature(x, contours);
                            if (latestFeature.found) {
                                initialTrack[2 * pointsToTrack] = uvPoint2D32f(latestFeature.left, y);
                                initialTrack[2 * pointsToTrack + 1] = uvPoint2D32f(latestFeature.right, y);
                                pointsToTrack++;
                            }
                        }

                        edgeSlice.release();
                    }

                    convertedEdges.release();
                    uv_release_image(&edges);
                    uv_free_canny_buf(cannyBuffer);

                    indices.reserve(pointsToTrack);
                    memcpy(track, initialTrack, sizeof(track));
                    for (int i = 0; i < pointsToTrack; i++) { indices[i] = i; }
                    distances.resize(pointsToTrack);

                    #ifdef DEBUG_MAPPING
                    printf("%i points -> ", pointsToTrack);
                    #endif

                    frame++;
                } else {
                    if (pointsToTrack == 0) {
                        frame = 0;
                    } else {
                        float annotatedMap[MAP_SIZE];
                        if (frame != 1) {
                            pthread_mutex_lock(&mapMutex);
                            memcpy(annotatedMap, baseMap, sizeof(annotatedMap));
                            pthread_mutex_unlock(&mapMutex);
                        }

                        UvPoint2D32f track_new[MAXIMUM_TRACK_SIZE];
                        vector<int> indices_new;
                        indices_new.reserve(pointsToTrack);

                        UvPoint2D32f points[MAXIMUM_TRACK_SIZE];
                        char found[MAXIMUM_TRACK_SIZE];
                        float error[MAXIMUM_TRACK_SIZE];
                        uv_CalcOpticalFlowPyrLK(handle, &image_previous, &image, pyramid_previous, pyramid, track, points, pointsToTrack * 2, 3, found, error, terminationCriteria, 0, .0001);

                        int i = 0;
                        for (int p = 0; p < pointsToTrack; p++) {
                            if (found[2 * p] && found[2 * p + 1]) {
                                track_new[2 * i] = points[2 * p];
                                track_new[2 * i + 1] = points[2 * p + 1];
                                indices_new[i] = indices[p];
                                i++;

                                if (frame != 1) {
                                    int previousLeft = round(initialTrack[2 * indices[p]].x);
                                    int previousRight = round(initialTrack[2 * indices[p] + 1].x);
                                    int latestLeft = round(points[2 * p].x);
                                    int latestRight = round(points[2 * p + 1].x);

                                    if (latestRight - latestLeft >= previousRight - previousLeft + MINIMUM_EXPANSION && abs(points[p].y - points[p + 1].y) <= MAXIMUM_VERTICAL_SEPARATION) {
                                        float previousRatio = widthToDistance(previousLeft, previousRight);
                                        float latestRatio = widthToDistance(latestLeft, latestRight);
                                        float distance = (previousRatio * dx_latest) / (latestRatio - previousRatio);
                                        #ifdef DEBUG_MAPPING
                                        printf("%i px -> %i px => %g m (%g m)\n", previousRight - previousLeft, latestRight - latestLeft, distance, dx_latest);
                                        #endif

                                        if (distance >= MINIMUM_DISTANCE && distance <= MAXIMUM_DISTANCE) {
                                            distances[indices[p]].push_back(distance + dx_copy);

                                            if (distances[indices[p]].size() >= COMBINE_ESTIMATES_WHEN) {
                                                insertion_sort(distances[indices[p]]);
                                                vector<float> distancesForFeature = distances[indices[p]];
                                                int numberOfDistances = 1;
                                                int lowestDistanceIndex;
                                                for (int d = distancesForFeature.size() - 1; d >= 0; d--) {
                                                    distance = distancesForFeature[d];
                                                    int lowestConsistentIndex, highestConsistentIndex;
                                                    float threshold = RANSAC_BASELINE + (distance > dx_copy) * RANSAC_MULTIPLIER * (distance - dx_copy);
                                                    for (int e = d - 1; e >= 0; e--) {
                                                        if (distance - distancesForFeature[e] < threshold) {
                                                            lowestConsistentIndex = e;
                                                        } else {
                                                            break;
                                                        }
                                                    }
                                                    for (int e = d + 1; e < distancesForFeature.size(); e++) {
                                                        if (distancesForFeature[e] - distance < threshold) {
                                                            highestConsistentIndex = e;
                                                        } else {
                                                            break;
                                                        }
                                                    }
                                                    int n = highestConsistentIndex - lowestConsistentIndex + 1;
                                                    if (n > numberOfDistances) {
                                                        numberOfDistances = n;
                                                        lowestDistanceIndex = lowestConsistentIndex;
                                                    }
                                                }

                                                if (numberOfDistances > 1) {
                                                    float average = 0;
                                                    for (int d = 0; d < distancesForFeature.size(); d++) {
                                                        if (d >= lowestDistanceIndex && d < lowestDistanceIndex + numberOfDistances) {
                                                            average += distancesForFeature[d];
                                                        }
                                                    }
                                                    average /= numberOfDistances;

                                                    float width = 100 * latestRatio * (average - dx_copy) / MAP_RESOLUTION;
                                                    if (width > 0) {
                                                        float scale = width / (latestRight - latestLeft);
                                                        int offset = round(scale * (((latestLeft + latestRight) / 2) - (IMAGE_WIDTH / 2)));
                                                        int center = ((MAP_SIZE - 1) / 2) + round(100 * dy_copy / MAP_RESOLUTION) + offset;
                                                        int left = center - round(width / 2);
                                                        int right = center + round(width / 2);
                                                        bool addToMap = left < (MAP_SIZE - 1) && right > 0;

                                                        if (addToMap) {
                                                            #ifdef DEBUG_MAPPING
                                                            printf("%g m depth, %g m width -> adding from %i to %i, centered at %i\n", average, latestRatio * (average - dx_copy), left, right, (latestLeft + latestRight) / 2);
                                                            #endif
                                                            if (left < 0) {
                                                                left = 0;
                                                            } else if (right >= MAP_SIZE) {
                                                                right = MAP_SIZE - 1;
                                                            }   

                                                            if (annotatedMap[left] == annotatedMap[right] && annotatedMap[left] != EMPTY_DISTANCE && annotatedMap[left] > average) {
                                                                float distance_old = annotatedMap[left];
                                                                for (int x = left - 1; x >= 0; x--) {
                                                                    if (annotatedMap[x] != distance_old) break;
                                                                    annotatedMap[x] = EMPTY_DISTANCE;
                                                                }
                                                                for (int x = left; x <= right; x++) {
                                                                    annotatedMap[x] = average;
                                                                }
                                                                for (int x = right + 1; x < MAP_SIZE; x++) {
                                                                    if (annotatedMap[x] != distance_old) break;
                                                                    annotatedMap[x] = EMPTY_DISTANCE;
                                                                }
                                                            } else {
                                                                bool closedObstacle = true;
                                                                for (int x = left + 1; x < right - 1; x++) {
                                                                    if (annotatedMap[x] != annotatedMap[left] && annotatedMap[x] != annotatedMap[right] && annotatedMap[x] != EMPTY_DISTANCE && annotatedMap[x] == annotatedMap[x + 1] && annotatedMap[x] < average) {
                                                                        closedObstacle = false;
                                                                        break;
                                                                    }
                                                                }
                                                                if (closedObstacle) {
                                                                    for (int x = left; x <= right; x++) {
                                                                        if (annotatedMap[x] == EMPTY_DISTANCE || annotatedMap[x] > distance) {
                                                                            annotatedMap[x] = average;
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        memcpy(track, track_new, sizeof(track));
                        indices = move(indices_new);
                        pointsToTrack = i;

                        if (frame != 1) {
                            pthread_mutex_lock(&mapMutex);
                            memcpy(latestMap, annotatedMap, sizeof(latestMap));
                            if (++frame == FIND_NEW_POINTS_AFTER + 2) {
                                frame = 0;
                                memcpy(baseMap, latestMap, sizeof(baseMap));
                            }
                            pthread_mutex_unlock(&mapMutex);
                        }
                    }
                }
    
                #ifdef DEBUG_MAPPING
                string filename = "/home/odroid/node/image" + to_string(saved);
                uv_SaveImage_bmp(const_cast<char*>(filename.c_str()), &image);
                saved++;
                #endif
                uv_deepCopy(&image_previous, &image);
                if (!firstFrame) {
                    free(pyramid_previous);
                    free(pyramidBuffer_previous);
                } else {
                    firstFrame = false;
                }
                pyramid_previous = pyramid;
                pyramidBuffer_previous = pyramidBuffer;

                // Run this tracking loop at a frequency of 80 Hz.
                int loopTime = time_usec() - previousTime;
                if (loopTime < 12500) {
                    usleep(12500 - loopTime);
                }
                #ifdef DEBUG_MAPPING
                printf("%i us to process the latest frame.\n", loopTime);
                #endif
            } else {
                usleep(10000);
            }
        } else if (state_copy == STATE_STOPPED) {
            break;
        } else {
            frame = 0;
            usleep(10000);
        }
    }

    return (void *)0;
}

Feature findFeature(int x_point, vector<vector<Point> > contours)
{
    Feature feature;
    feature.left = 0;
    feature.right = numeric_limits<int>::max();

    bool foundLeft = false;
    bool foundRight = false;
    bool intersected = false;
    bool duplicate = false;
    for (int c = 0; c < contours.size(); c++) {
        // Compute each contour's bounding box.
        int x_min = numeric_limits<int>::max();
        int x_max = 0;
        int y_min = numeric_limits<int>::max();
        int y_max = 0;
        for (int p = 0; p < contours[c].size(); p++) {
            Point point = contours[c][p];
            if (point.x < x_min) x_min = point.x;
            if (point.x > x_max) x_max = point.x;
            if (point.y < y_min) y_min = point.y;
            if (point.y > y_max) y_max = point.y;
        }

        // Find the nearsest countours on each side of the given seed point.
        if (y_min == 1 && y_max == TEMPLATE_HEIGHT - 2) {
            int x = (x_min + x_max) / 2;
            if (x_max < x_point && x > feature.left) {
                // Filter out duplicates in the interest of efficiency.
                if (x_point - x_max > SEED_SPACING) {
                    duplicate = true;
                }
                foundLeft = true;
                feature.left = x;
            } else if (x_min > x_point && x < feature.right) {
                foundRight = true;
                feature.right = x;
            } else if (x_min < x_point && x_max > x_point) {
                intersected = true;
            }
        }
    }

    feature.width = feature.right - feature.left;
    feature.found = foundLeft && foundRight && (!duplicate || intersected);
    return feature;
}

float widthToDistance(int left, int right)
{
    // Return the width-to-distance ratio of an object with given pixel boundaries.
    int center = IMAGE_WIDTH / 2;
    float consideredWidth = (left >= center) ? (right - center) : (center - left);
    return ((right - left) / consideredWidth) * tan((CONVERT_TO_RAD(93) / IMAGE_WIDTH) * consideredWidth);
}

void insertion_sort(vector<float> &v)
{
    // Efficiently sort a nearly-sorted vector.
    for (int i = 1; i < v.size(); i++) {
        float x = v[i];
        int j = i;
        while (j > 0 && v[j - 1] > x) {
            v[j] = v[j - 1];
            j--;
        }
        v[j] = x;
    }
}