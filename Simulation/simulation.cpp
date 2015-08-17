//
// simulation.cpp
// Firefly [v1]
//
// Copyright (c) 2015 Mihir Garimella.
//

#include <iostream>
#include <cmath>
#include <random>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#if __cplusplus <= 199711L
  #error This simulation needs a C++11 compliant compiler.
#endif

// Define flags to indicate what kind of simulation we want to run.
// #define EXPERIMENT
//#define VISUALIZE
#define SQUARE
#define DEBUG

// Define helper macros to convert between degrees and radians.
#define RAD_TO_DEG(X) (fixAngle(X) * 180 / M_PI)
#define DEG_TO_RAD(X) ((X) * M_PI / 180)

// Define a helper macro to optionally log a message to the console.
#ifdef DEBUG
    #define debug(fmt, ...) printf((fmt), __VA_ARGS__)
#else
    #define debug(fmt, ...) (0)
#endif

using namespace std;
using namespace cv;

// Create a struct to model an obstacle within the arena.
typedef struct Obstacle {
    Point2f d1;
    Point2f d2;
    Point2f a;
    float ab[2];
    float ad[2];
} Obstacle;

// Create global variables to keep track of...

// ...the robot's state.
float x, y, yaw;
int temperature_previous, temperature_increasing, temperature_maximum, temperature_decreasing, concentration_previous, concentration_decreasing, iterationMoves, forwardMoves;
bool foundTarget;
Point point_previous, coveragePoint_previous;

// ...the robot's hardware.
const float velocity = 0.1; // m/move
const float gasSensorDisparity = 0.2; // m
const float fieldOfView = DEG_TO_RAD(90); // rad

// ...the robot's configuration.
const int decreasingBeforeSaccade = 5; // moves
const int increasingBeforeFound = 3; // moves
const int decreasingAfterFound = 1; // moves
const int foundTemperatureThreshold = 800; // (some arbitrary units)

// ...the layout of the arena.
int diameter = 20; // m
const int temperatureGradientDiameter = 1; // m
const float avoid = 0.5; // m
Obstacle obstacles[4]; // m/s

// ...simulation parameters.
const int iterations = 20; // iterations
const int timeoutMoves = ((10 * 60 * 1.0) / velocity); // moves
const int minimumMoves = 5; // moves
const int moveBackMoves = 3; // moves
const int sigma = 30; // deg

// ...experiment parameters.
#ifdef EXPERIMENT
const int diameter_step = 10; // m
const int imageSize = 2 * diameter; // px
const int diameter_max = 250; // m

// ...visualization parameters.
#else
const int imageSize = 300; // px
#endif
const int imageOffset = 0; // px

// Declare helper functions that we'll define later on.
float randomInRange(float minimum, float maximum);
bool onSegment(Point p, Point q, Point r);
int orientation(Point p, Point q, Point r);
bool intersect(float x1, float y1, float x2, float y2);
int getConcentration(float x_c, float y_c);
int getTemperature(float x_c, float y_c);
bool shouldAvoid(float size, bool checkObstacles);
float dot(float a[2], float b[2]);
float squared(float a);
Point getVisualizationPointFromCoordinates(float x_c, float y_c);
Point getCoveragePointFromCoordiantes(float x_c, float y_c);
void initialize();
float fixAngle(float a);
bool checkAngleDifference(float a, float b, float maximumDifference);

int main() {
    srand(time(NULL));
    
    #ifdef EXPERIMENT
    for (; diameter <= diameter_max; diameter += diameter_step) {
    #endif
    
    // Create local varaibles to keep track of statistics for this test.
    int failures = 0;
    int totalMoves = 0;
    int totalCoverage = 0;
    
    // Create a random number generator to choose the turn angle (for BEHAVIOR_AVOID) according to a normal distribution.
    default_random_engine generator;
    normal_distribution<float> distribution(M_PI, DEG_TO_RAD(sigma));

    for (int i = 1; i <= iterations; i++) {
        initialize();
        debug("Starting iteration %i at (%g, %g) with yaw %g degrees\n", i, x, y, RAD_TO_DEG(yaw));
        
        // Create a dynamically-sized binary image to measure coverage.
        Mat coverage;
        coverage.create(diameter * 8, diameter * 8, CV_8UC1);
        coverage = Scalar(0);
        
        #ifdef VISUALIZE
        // Create an image of the given size for the visualization.
        Mat visualization;
        visualization.create(imageSize + imageOffset * 2, imageSize + imageOffset * 2, CV_8UC3);
        visualization = Scalar(255, 255, 255);
        
        // Draw the arena.
        #ifdef SQUARE
        rectangle(visualization, Point(imageOffset, imageOffset), Point(imageSize + imageOffset, imageSize + imageOffset), Scalar(60, 60, 60), -1);
        #else
        circle(visualization, Point(imageSize / 2 + imageOffset, imageSize / 2 + imageOffset), imageSize / 2, Scalar(60, 60, 60), -1);
        #endif
        
        // Draw the target.
        circle(visualization, Point(imageSize / 2 + imageOffset, imageSize / 2 + imageOffset), 4, Scalar(0, 255, 255));
        circle(visualization, Point(imageSize / 2 + imageOffset, imageSize / 2 + imageOffset), 8, Scalar(0, 255, 255));
        circle(visualization, Point(imageSize / 2 + imageOffset, imageSize / 2 + imageOffset), 12, Scalar(0, 255, 255));

        // Draw the starting point.
        circle(visualization, point_previous, 2, Scalar(0, 0, 255), 2);
        
        // Draw the obstacles.
        for (int i = 0; i < 4; i++) {
            line(visualization, getVisualizationPointFromCoordinates(obstacles[i].d1.x, obstacles[i].d1.y), getVisualizationPointFromCoordinates(obstacles[i].d2.x, obstacles[i].d2.y), Scalar(0, 255, 255));
        }
        #endif
        
        while (true) {
            // Come up with a new position for the robot.
            float x_new = x + (velocity * sin(yaw));
            float y_new = y + (velocity * cos(yaw));
            
            bool finished = false;
            if (iterationMoves + forwardMoves >= timeoutMoves || shouldAvoid(diameter / 2.f, false) || intersect(x, y, x_new, y_new)) {
                // If we've collided with the arena walls or an obstacle, or if the battery of the robot has been depleted, this iteration has failed.
                failures++;
                debug("Iteration %i failed.\n\n", i);
                finished = true;
            } else if (foundTarget) {
                // If we've located the target, prepare for the next iteration.
                iterationMoves += forwardMoves;
                totalMoves += iterationMoves;
                debug("Maximum at (%g, %g).\n", x, y);
                debug("Finished iteration %i in %i moves.\n\n", i, iterationMoves);
                totalCoverage += countNonZero(coverage);
                finished = true;
            }
            
            if (finished) {
                #ifdef VISUALIZE
                // If this iteration is done, show the resulting visualization.
                char w = waitKey();
                if (w == 0x1b) {
                    return 0;
                } else if (w == 's') {
                    string name = "/Users/mihir/Projects/Sensor Platform/Simulation/" + to_string(i) + ".png";
                    imwrite(name.c_str(), visualization);
                    debug("\b\bSaved image at %s.\n\n", name.c_str());
                }
                #endif
                break;
            }
            
            // Move the robot.
            x = x_new;
            y = y_new;
            forwardMoves++;
            Point point_latest = getVisualizationPointFromCoordinates(x, y);
            Point coveragePoint_latest = getCoveragePointFromCoordiantes(x, y);
            line(coverage, coveragePoint_previous, coveragePoint_latest, Scalar(255), ceil(0.25 * imageSize / diameter));
            #ifdef VISUALIZE
            line(visualization, point_previous, point_latest, Scalar(0, 0, 255));
            #endif
            point_previous = point_latest;
            coveragePoint_previous = coveragePoint_latest;

            // If the origin is visible within our field of view, turn towards it.
            if (!intersect(x, y, 0, 0)) {
                float bearing = atan2(x, y) + M_PI;
                if (!checkAngleDifference(yaw, bearing, DEG_TO_RAD(2)) && checkAngleDifference(yaw, bearing, fieldOfView / 2)) {
                    debug("Turning towards object of interest, changing yaw from %g to %g.\n", RAD_TO_DEG(yaw), RAD_TO_DEG(bearing));
                    yaw = bearing;

                    iterationMoves += 30; // Add three seconds of extra flight time to account for this behavior.
                }
            }
            
            if (forwardMoves > minimumMoves && shouldAvoid((diameter / 2.f) - avoid, true)) {
                // If we're about to hit an obstacle, move back and turn away, roughly in the opposite direction.
                x -= moveBackMoves * velocity * sin(yaw);
                y -= moveBackMoves * velocity * cos(yaw);
                float turn = distribution(generator);
                if (turn > 3 * M_PI / 2) {
                    turn =  3 * M_PI / 2;
                } else if (turn < M_PI / 2) {
                    turn = M_PI / 2;
                }

                // If we'll see the origin during our turn, turn towards it now.
                bool aligned = false;
                float yaw_initial = yaw;
                if (!intersect(x, y, 0, 0)) {
                    float bearing = atan2(x, y) + M_PI;
                    if (!checkAngleDifference(yaw_initial, bearing, DEG_TO_RAD(45))) {
                        for (yaw += fieldOfView / 2; yaw < yaw_initial + turn; yaw += fieldOfView / 2) {
                            if (checkAngleDifference(yaw, bearing, fieldOfView / 2)) {
                                debug("Turning towards object of interest, changing yaw from %g to %g.\n", RAD_TO_DEG(yaw), RAD_TO_DEG(bearing));
                                yaw = bearing;
                                aligned = true;;
                                break;
                            }
                        }
                    }
                }
                if (!aligned) {
                    yaw = yaw_initial + turn;
                    debug("Avoiding collision at (%g, %g), changing yaw from %g to %g.\n", x, y, RAD_TO_DEG(yaw_initial), RAD_TO_DEG(yaw));
                }
                
                Point point_latest = getVisualizationPointFromCoordinates(x, y);
                line(coverage, point_previous, point_latest, Scalar(255), ceil(imageSize / (4.f * diameter)));
                #ifdef VISUALIZE
                line(visualization, point_previous, point_latest, Scalar(0, 0, 255));
                circle(visualization, getVisualizationPointFromCoordinates(x, y), 2, Scalar(255, 100, 120), 2);
                #endif
                point_previous = point_latest;

                temperature_previous = -1;
                temperature_increasing = 0;
                temperature_decreasing = 0;
                concentration_previous = -1;
                concentration_decreasing = 0;
                
                iterationMoves += forwardMoves + (aligned ? 80 : 50); // Add five seconds of extra flight time to account for this behavior, and add an additional three seconds if we turned toward the origin.
                forwardMoves = 0;
            } else {
                // End this iteration successfully if we've reached a "good enough" (i.e., higher than a threshold) maximum in temperature.
                int temperature = getTemperature(x, y);
                if (temperature > temperature_previous && temperature_previous != -1) {
                    if (temperature_decreasing > 0) {
                        temperature_decreasing = 0;
                        temperature_increasing = 1;
                    } else {
                        temperature_increasing++;
                    }
                } else if (temperature < temperature_previous && temperature_previous != -1) {
                    if (++temperature_decreasing == 1) {
                        temperature_maximum = temperature_previous;
                    }
                    if (temperature_decreasing >= decreasingAfterFound && temperature_increasing >= increasingBeforeFound) {
                        if (temperature_maximum > foundTemperatureThreshold) {
                            foundTarget = true;
                        } else {
                            temperature_decreasing = 0;
                        }
                    }
                }
                temperature_previous = temperature;

                if (!foundTarget) {
                    // Begin a saccade if concentration has decreased over the last few movements.
                    int concentration = getConcentration(x, y);
                    if (concentration < concentration_previous && concentration_previous != -1) {
                        if (++concentration_decreasing >= decreasingBeforeSaccade) {
                            // Saccade in the direction of highest concentration.
                            float turn;
                            if (getConcentration(x + gasSensorDisparity * sin(yaw + (M_PI / 2)), y + gasSensorDisparity * cos(yaw + (M_PI / 2))) >= getConcentration(x + gasSensorDisparity * sin(yaw - (M_PI / 2)), y + gasSensorDisparity * cos(yaw - (M_PI / 2)))) {
                                turn = M_PI / 2;
                            } else {
                                turn = -M_PI / 2;
                            }

                            // If we'll see the origin during our turn, turn towards it now.
                            bool aligned = false;
                            float yaw_initial = yaw;
                            if (!intersect(x, y, 0, 0)) {
                                float bearing = atan2(x, y) + M_PI;
                                for (yaw += fieldOfView / 2; yaw < yaw_initial + turn; yaw += fieldOfView / 2) {
                                    if (checkAngleDifference(yaw, bearing, fieldOfView / 2)) {
                                        debug("Turning towards object of interest, changing yaw from %g to %g.\n", RAD_TO_DEG(yaw), RAD_TO_DEG(bearing));
                                        yaw = bearing;
                                        aligned = true;;
                                        break;
                                    }
                                }
                            }
                            if (!aligned) {
                                yaw = yaw_initial + turn;
                                debug((turn == M_PI / 2) ? "Saccade right at (%g, %g), changing yaw from %g to %g.\n" : "Saccade left at (%g, %g), changing yaw from %g to %g.\n", x, y, RAD_TO_DEG(yaw_initial), RAD_TO_DEG(yaw));
                            }

                            #ifdef VISUALIZE
                            circle(visualization, point_previous, 2, Scalar(120, 255, 120), 2);
                            #endif
                            
                            temperature_previous = -1;
                            temperature_increasing = 0;
                            temperature_decreasing = 0;
                            concentration_previous = -1;
                            concentration_decreasing = 0;
                            
                            iterationMoves += forwardMoves + (aligned ? 70 : 40); // Add four seconds of extra flight time to account for this saccade, and add an additional three seconds if we turned toward the origin.
                            forwardMoves = 0;
                        } else {
                            concentration_previous = concentration;
                        }
                    } else {
                        concentration_previous = concentration;
                    }
                }
            }
            
            #ifdef VISUALIZE
            // Show the latest copy of the visualization.
            imshow("Visualization", visualization);
            if (waitKey(1) == 0x1b) {
                return 0;
            }
            #endif
        }
    }
    
    // Log the simulation results.
    #ifdef EXPERIMENT
    printf("%i,%i,%i,%i\n", diameter, totalMoves / (iterations - failures), totalCoverage / (iterations - failures), failures);
    }
    #else
    printf("Average of %i moves (%i seconds at 1 meter per second), failed %i times.\n", totalMoves / (iterations - failures), totalMoves / (10 * iterations - 10 * failures), failures);
    #endif
}

// Helper function to return a random number in a given range.
float randomInRange(float minimum, float maximum) {
    // Based on http://stackoverflow.com/questions/686353/c-random-float-number-generation.
    return minimum + (float)(rand()) / (float)(RAND_MAX / (maximum - minimum));
}

// Helper function to return the concentration at a point according to a two-dimensional Gaussian distribution.
int getConcentration(float x_c, float y_c) {
    return round(1000 * exp(squared(x_c) / (-2 * squared(diameter / 2.f)) + squared(y_c) / (-2 * squared(diameter / 2.f))));
}

// Helper function to return the temperature at a point according to a two-dimensional Gaussian distribution.
int getTemperature(float x_c, float y_c) {
    return round(1000 * exp(squared(x_c) / (-2 * squared(temperatureGradientDiameter / 2.f)) + squared(y_c) / (-2 * squared(temperatureGradientDiameter / 2.f))));
}

// Helper function to check whether the robot is about to collide with an obstacle or the wall.
bool shouldAvoid(float size, bool checkObstacles) {
    if (checkObstacles) {
        // Based on http://math.stackexchange.com/questions/190111/how-to-check-if-a-point-is-inside-a-rectangle.
        for (int i = 0; i < 4; i++) {
            float am[2] = {x - obstacles[i].a.x, y - obstacles[i].a.y};
            float am_ab = dot(am, obstacles[i].ab);
            float am_ad = dot(am, obstacles[i].ad);
            
            if (am_ab > 0 && am_ab < squared(2 * (diameter / 10.f + avoid)) && am_ad > 0 && am_ad < squared(2 * avoid)) {
                return true;
            }
        }
    }
    
    if (size != 0) {
        #ifdef SQUARE
        return abs(x) >= size || abs(y) >= size;
        #else
        return squared(x) + squared(y) >= squared(size);
        #endif
    } else {
        return false;
    }
}

// Helper function to find the dot product of two two-dimensional vectors.
float dot(float a[2], float b[2]) {
    return a[0] * b[0] + a[1] * b[1];
}

// Helper function to square a floating-point value.
float squared(float a) {
    return a * a;
}

// Helper function to transform from the simulation coordinate system to the visualization coordinate system.
Point getVisualizationPointFromCoordinates(float x_c, float y_c) {
    int x_p = round(x_c * (imageSize / (float)(diameter))) + imageSize / 2 + imageOffset;
    int y_p = round(-y_c * (imageSize / (float)(diameter))) + imageSize / 2 + imageOffset;
    return Point(x_p, y_p);
}

// Helper function to transform from the simulation coordinate system to the coverage coordinate system.
Point getCoveragePointFromCoordiantes(float x_c, float y_c) {
    int x_p = x_c * 8 + diameter * 4;
    int y_p = -y_c * 8 + diameter * 4;
    return Point(x_p, y_p);
}

// Helper function to initialize the robot and arena for each iteration.
void initialize() {
    float obstacleWidth = diameter / 10.f;
    float obstacleOffset = diameter / 5.f;
    
    // Create one obstacle with a given length and random angle in each quadrant of the arena.
    for (int o = 0; o < 4; o++) {
        float x_o = (o == 0 || o == 3 ? 1 : -1) * obstacleOffset;
        float y_o = (o == 2 || o == 3 ? 1 : -1) * obstacleOffset;
        
        float theta = randomInRange(0, 2 * M_PI);
        obstacles[o].d1 = Point2f(x_o - obstacleWidth * cos(theta), y_o - obstacleWidth * sin(theta));
        obstacles[o].d2 = Point2f(x_o + obstacleWidth * cos(theta), y_o + obstacleWidth * sin(theta));
        
        obstacles[o].a = Point2f(obstacles[o].d1.x - avoid * (sin(theta) + cos(theta)), obstacles[o].d1.y + avoid * (cos(theta) - sin(theta)));
        obstacles[o].ab[0] = 2 * (obstacleWidth + avoid) * cos(theta);
        obstacles[o].ab[1] = 2 * (obstacleWidth + avoid) * sin(theta);
        obstacles[o].ad[0] = 2 * avoid * sin(theta);
        obstacles[o].ad[1] = -2 * avoid * cos(theta);
    }
    
    // Choose values for x and y that won't immediately cause a collision.
    float smallRadius = diameter / 2.f - avoid;
    do {
        #ifdef SQUARE
        x = randomInRange(-smallRadius, smallRadius);
        y = randomInRange(-smallRadius, smallRadius);
        #else
        float theta = randomInRange(0, 2 * M_PI);
        float radius = sqrt(randomInRange(0, squared(smallRadius)));
        x = radius * sin(theta);
        y = radius * cos(theta);
        #endif
    } while (shouldAvoid(0, true));
    
    // Choose a random yaw.
    yaw = randomInRange(0, 2 * M_PI);
    
    // Reset temporary variables for the next iteration.
    concentration_previous = -1;
    temperature_previous = -1;
    temperature_increasing = 0;
    temperature_decreasing = 0;
    concentration_decreasing = 0;
    iterationMoves = 0;
    forwardMoves = 0;
    foundTarget = false;
    point_previous = getVisualizationPointFromCoordinates(x, y);
    coveragePoint_previous = getCoveragePointFromCoordiantes(x, y);
}

// Helper function to translate an angle in R to an angle in the range [0, 2pi].
float fixAngle(float a) {
    while (a > 2 * M_PI) {
        a -= 2 * M_PI;
    }
    while (a < 0) {
        a += 2 * M_PI;
    }
    return a;
}

// Helper function to see if the difference between two angles is less than a given threshold.
bool checkAngleDifference(float a, float b, float maximumDifference) {
    return cos(a) * cos(b) + sin(a) * sin(b) >= cos(maximumDifference);
}

// Helper function to determine whether a line segment will intersect an obstacle.
bool intersect(float x0, float y0, float x1, float y1) {
    for (int i = 0; i < 4; i++) {
        // Based on http://stackoverflow.com/questions/14176776/find-out-if-2-lines-intersect.
        float x2 = obstacles[i].d1.x;
        float y2 = obstacles[i].d1.y;
        float x3 = obstacles[i].d2.x;
        float y3 = obstacles[i].d2.y;
        if (((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0)) * ((x3 - x0) * (y1 - y0) - (y3 - y0) * (x1 - x0)) < 0 && ((x0 - x2) * (y3 - y2) - (y0 - y2) * (x3 - x2)) * ((x1 - x2) * (y3 - y2) - (y1 - y2) * (x3 - x2)) < 0) {
            return true;
        }
    }
    return false;
}