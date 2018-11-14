/*
 * Copyright (C) 2018  Christian Berger
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

class CustomLine {
   public:
    CustomLine() :
        p1(),
        p2(),
        slope(0) {}

    ~CustomLine() {}

    bool operator<(const CustomLine &other) const {
        return std::max(p1.y, p2.y) > std::max(other.p1.y, other.p2.y);
    }

    bool operator==(const CustomLine &other) const {
        return ((p1.y == other.p1.y) && (p1.x == other.p1.x));
    }

    cv::Point p1, p2;
    float slope;
};

int32_t main(int32_t argc, char **argv) {
    int32_t retCode{1};
    auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
    if ( (0 == commandlineArguments.count("cid")) ||
         (0 == commandlineArguments.count("name")) ||
         (0 == commandlineArguments.count("width")) ||
         (0 == commandlineArguments.count("height")) ) {
        std::cerr << argv[0] << " attaches to a shared memory area containing an ARGB image." << std::endl;
        std::cerr << "Usage:   " << argv[0] << " --cid=<OD4 session> --name=<name of shared memory area> [--verbose]" << std::endl;
        std::cerr << "         --cid:       CID of the OD4Session to send and receive messages" << std::endl;
        std::cerr << "         --name:      name of the shared memory area to attach" << std::endl;
        std::cerr << "         --width:     width of the frame" << std::endl;
        std::cerr << "         --height:    height of the frame" << std::endl;
        std::cerr << "         --threshold: binary threshold value (default: 115)" << std::endl;
        std::cerr << "Example: " << argv[0] << " --cid=253 --name=img.argb --width=640 --height=480 --verbose" << std::endl;
    }
    else {
        const std::string NAME{commandlineArguments["name"]};
        const uint32_t WIDTH{static_cast<uint32_t>(std::stoi(commandlineArguments["width"]))};
        const uint32_t HEIGHT{static_cast<uint32_t>(std::stoi(commandlineArguments["height"]))};
        const bool VERBOSE{commandlineArguments.count("verbose") != 0};

        ////////////////////////////////////////////////////////////////////////
        // Parameters for lane-detector.
        const uint32_t THRESHOLD{commandlineArguments.count("verbose") != 0 ? static_cast<uint32_t>(std::stoi(commandlineArguments["threshold"])) : 115};

        // Attach to the shared memory.
        std::unique_ptr<cluon::SharedMemory> sharedMemory{new cluon::SharedMemory{NAME}};
        if (sharedMemory && sharedMemory->valid()) {
            std::clog << argv[0] << ": Attached to shared memory '" << sharedMemory->name() << " (" << sharedMemory->size() << " bytes)." << std::endl;

            ////////////////////////////////////////////////////////////////////
            struct PolySize {
                int sizeX, sizeY, sizeR;
                cv::Point shortSideMiddle;
                cv::Point longSideMiddle;
            };

            struct Config {
                int XTimesYMin, XTimesYMax, maxY, maxArea;
            } conf;
            conf.XTimesYMin = 2;
            conf.XTimesYMax = 20;
            conf.maxY = 235;
            conf.maxArea = 4;

            auto getLineSlope = [](const cv::Point &p1, const cv::Point &p2) {
                float slope = static_cast<float>(M_PI)/2.0f;
                if ((p1.x - p2.x) != 0) {
                    slope = (p1.y - p2.y) / static_cast<float>(p1.x - p2.x);
                    slope = atanf(slope);
                }
                if (slope < 0) {
                    return 180.0f + (slope * 180.0f / static_cast<float>(M_PI));
                }
                return slope * 180.0f / static_cast<float>(M_PI);
            };

            auto createLineFromRect = [](cv::RotatedRect *rect, int sizeY) {
                cv::Point2f rect_points[4];
                rect->points(rect_points);

                CustomLine l;
                cv::Point pt1, pt2;
                if (rect->angle < 90) {
                    float angle = rect->angle * static_cast<float>(M_PI)/180.0f;
                    float xOffset = cosf(angle) * sizeY / 2.0f;
                    float yOffset = sinf(angle) * sizeY / 2.0f;
                    pt1.y = rect->center.y + yOffset;
                    pt1.x = rect->center.x + xOffset;
                    pt2.y = rect->center.y - yOffset;
                    pt2.x = rect->center.x - xOffset;
                }
                else {
                    rect->angle = rect->angle - 180.0f;
                    float angle = (-rect->angle) * static_cast<float>(M_PI)/180.0f;
                    float xOffset = cosf(angle) * sizeY / 2.0f;
                    float yOffset = sinf(angle) * sizeY / 2.0f;
                    pt1.y = rect->center.y + yOffset;
                    pt1.x = rect->center.x - xOffset;
                    pt2.y = rect->center.y - yOffset;
                    pt2.x = rect->center.x + xOffset;
                }
                l.p1 = pt1;
                l.p2 = pt2;
                l.slope = rect->angle;
                return l;
            };

            ////////////////////////////////////////////////////////////////////
            cv::KalmanFilter KF(4, 2, 0);
            KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
            cv::Mat_<float> measurement(2,1); measurement.setTo(cv::Scalar(0));

            // Initialize Kalman filter.
            KF.statePre.at<float>(0) = WIDTH/2;
            KF.statePre.at<float>(1) = HEIGHT/2;
            KF.statePre.at<float>(2) = 0;
            KF.statePre.at<float>(3) = 0;
            cv::setIdentity(KF.measurementMatrix);
            cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));
            cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(10));
            cv::setIdentity(KF.errorCovPost, cv::Scalar::all(.1));

            ////////////////////////////////////////////////////////////////////
            // Create an OpenCV image header using the data in the shared memory.
            IplImage *iplimage{nullptr};
            if (VERBOSE) {
                CvSize size;
                size.width = WIDTH;
                size.height = HEIGHT;

                iplimage = cvCreateImageHeader(size, IPL_DEPTH_8U, 4 /* four channels: ARGB */);
                sharedMemory->lock();
                {
                    iplimage->imageData = sharedMemory->data();
                    iplimage->imageDataOrigin = iplimage->imageData;
                }
                sharedMemory->unlock();
            }

            // Interface to a running OpenDaVINCI session; here, you can send and receive messages.
            cluon::OD4Session od4{static_cast<uint16_t>(std::stoi(commandlineArguments["cid"]))};

            // Endless loop; end the program by pressing Ctrl-C.
            while (od4.isRunning()) {
                cv::Mat img;
                cv::Mat originalImage;

                // Wait for a notification of a new frame.
                sharedMemory->wait();

                // Lock the shared memory.
                sharedMemory->lock();
                {
                    // Copy image into cvMat structure.
                    // Be aware of that any code between lock/unlock is blocking
                    // the camera to provide the next frame. Thus, any
                    // computationally heavy algorithms should be placed outside
                    // lock/unlock.
                    img = cv::cvarrToMat(iplimage);
                }
                sharedMemory->unlock();

                ////////////////////////////////////////////////////////////////
                // Display image.
//                if (VERBOSE) {
//                    cv::imshow(sharedMemory->name().c_str(), img);
//                    cv::waitKey(1);
//                }

                ////////////////////////////////////////////////////////////////
                {
                    // Copy image.
                    img.copyTo(originalImage);
                }

                ////////////////////////////////////////////////////////////////
                {
                    // Turn image into grayscale.
                    cv::cvtColor(img, img, CV_BGR2GRAY);

//                    if (VERBOSE) {
//                        std::stringstream sstr;
//                        sstr << sharedMemory->name() << "-grayscale";
//                        const std::string windowName = sstr.str();
//                        cv::imshow(windowName.c_str(), img);
//                        cv::waitKey(1);
//                    }
                }

                ////////////////////////////////////////////////////////////////
                int32_t height = img.size().height;
                int32_t width = img.size().width;
                {
                    // Cropping image.
                    img = img(cv::Rect(1, 6 * height / 16 - 1, width - 1, 8 * height / 16 - 1));
                    originalImage = originalImage(cv::Rect(1, 6 * height / 16 - 1, width - 1, 8 * height / 16 - 1));

//                    if (VERBOSE) {
//                        std::stringstream sstr;
//                        sstr << sharedMemory->name() << "-cropped";
//                        const std::string windowName = sstr.str();
//                        cv::imshow(windowName.c_str(), img);
//                        cv::waitKey(1);
//                    }
                }
                width = img.size().width;
                height = img.size().height;

                ////////////////////////////////////////////////////////////////
                {
                    // Apply binary threshold.
                    cv::threshold(img, img, THRESHOLD, 255, CV_THRESH_BINARY);

                    if (VERBOSE) {
                        std::stringstream sstr;
                        sstr << sharedMemory->name() << "-threshold";
                        const std::string windowName = sstr.str();
                        cv::imshow(windowName.c_str(), img);
                        cv::waitKey(1);
                    }
                }

                ////////////////////////////////////////////////////////////////
                std::vector<std::vector<cv::Point> > listOfContours;
                std::vector<cv::Vec4i> hierarchy;
                {
                    // Find contours.
                    cv::findContours(img, listOfContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

//                    if (VERBOSE) {
//                        const cv::Scalar WHITE(255, 255, 255);

//                        cv::Mat imageWithContours(img.size().height, img.size().width, CV_32F);
//                        for(auto i = 0u; i < listOfContours.size(); i++) {
//                            cv::drawContours(imageWithContours, listOfContours, i, WHITE, 1, 8, hierarchy, 0, cv::Point());
//                        }

//                        std::stringstream sstr;
//                        sstr << sharedMemory->name() << "-contours";
//                        const std::string windowName = sstr.str();
//                        cv::imshow(windowName.c_str(), imageWithContours);
//                        cv::waitKey(1);
//                    }
                }

                ////////////////////////////////////////////////////////////////
                std::vector<std::vector<cv::Point> > listOfPolygonalContours;
                listOfPolygonalContours.resize(listOfContours.size());
                {
                    // Find polygonal contours.
                    for(auto i = 0u; i < listOfContours.size(); i++) {
                        cv::approxPolyDP(cv::Mat(listOfContours[i]), listOfPolygonalContours[i], 3 /* epsilon */, true /* is it closed? */);
                    }

//                    if (VERBOSE) {
//                        const cv::Scalar WHITE(255, 255, 255);

//                        cv::Mat imageWithPolygonalContours(img.size().height, img.size().width, CV_32F);
//                        for(auto i = 0u; i < listOfPolygonalContours.size(); i++) {
//                            cv::drawContours(imageWithPolygonalContours, listOfPolygonalContours, i, WHITE, 1, 8, hierarchy, 0, cv::Point());
//                        }

//                        std::stringstream sstr;
//                        sstr << sharedMemory->name() << "-polygonal contours";
//                        const std::string windowName = sstr.str();
//                        cv::imshow(windowName.c_str(), imageWithPolygonalContours);
//                        cv::waitKey(1);
//                    }
                }

                ////////////////////////////////////////////////////////////////
                std::vector<PolySize> line_sizes;
                std::vector<cv::RotatedRect> rects;
                {
                    // Compute bounding boxes.

                    cv::RotatedRect rect;
                    for(auto i = 0u; i < listOfPolygonalContours.size(); i++) {
                        rect = cv::minAreaRect(listOfPolygonalContours[i]);
                        cv::Point2f rect_points[4];
                        rect.points(rect_points);
                        int sizeX = 0, sizeY = 0, sizeR = 0;
                        cv::Point shortSideMiddle;
                        cv::Point longSideMiddle;
                        // Find rect sizes
                        for(int j = 0; j < 4; j++) {
                            sizeR = (int) cv::sqrt(
                                                cv::pow((rect_points[j].x - rect_points[(j + 1) % 4].x), 2)
                                                + cv::pow(
                                                        (rect_points[j].y
                                                         - rect_points[(j + 1) % 4].y), 2));
                            if (sizeX == 0) {
                                sizeX = sizeR;
                                shortSideMiddle.x = (int) ((rect_points[j].x
                                                                     + rect_points[(j + 1) % 4].x) / 2);
                                shortSideMiddle.y = (int) ((rect_points[j].y
                                                                     + rect_points[(j + 1) % 4].y) / 2);
                            }  else if (sizeY == 0 && sizeR != sizeX) {
                                sizeY = sizeR;
                                longSideMiddle.x = (int) ((rect_points[j].x
                                                                    + rect_points[(j + 1) % 4].x) / 2);
                                longSideMiddle.y = (int) ((rect_points[j].y
                                                                    + rect_points[(j + 1) % 4].y) / 2);
                            }
                        }

                        if (sizeX > sizeY) {
                            cv::Point2f temp;
                            sizeR = sizeX;
                            sizeX = sizeY;
                            sizeY = sizeR;
                            temp = longSideMiddle;
                            longSideMiddle = shortSideMiddle;
                            shortSideMiddle = temp;
                        }

                        rects.push_back(rect);
                        PolySize polysize = {sizeX, sizeY, sizeR, shortSideMiddle, longSideMiddle};
                        line_sizes.push_back(polysize);
                    }

//                    if (VERBOSE) {
//                        const cv::Scalar RED(0, 0, 255);

//                        cv::Mat imageWithBoundingBoxes;
//                        originalImage.copyTo(imageWithBoundingBoxes);

//                        for(auto i = 0u; i < listOfPolygonalContours.size(); i++) {
//                            rect = cv::minAreaRect(listOfPolygonalContours[i]);
//                            cv::Point2f rect_points[4];
//                            rect.points(rect_points);

//                            for (int j = 0; j < 4; j++) {
//                                cv::line(imageWithBoundingBoxes, rect_points[j], rect_points[(j + 1) % 4], RED, 2);
//                            }
//                        }

//                        std::stringstream sstr;
//                        sstr << sharedMemory->name() << "-bounding boxes";
//                        const std::string windowName = sstr.str();
//                        cv::imshow(windowName.c_str(), imageWithBoundingBoxes);
//                        cv::waitKey(1);
//                    }
                }

                ////////////////////////////////////////////////////////////////
                std::vector<CustomLine> dashLines(listOfContours.size());
                std::vector<CustomLine> solidLines(listOfContours.size());
                uint32_t cntDash = 0;
                uint32_t cntSolid = 0;
                {
                    // Classify lines.
                    int sizeX;
                    int sizeY;
                    int sizeR;
                    int area;
                    cv::RotatedRect rect;
                    cv::Point2f rect_points[4];
                    cv::Point rectCenter;
                    cv::Point shortSideMiddle;
                    for (auto i = 0u; i < line_sizes.size(); i++) {
                        sizeX = line_sizes[i].sizeX;
                        sizeY = line_sizes[i].sizeY;
                        sizeR = line_sizes[i].sizeR;
                        shortSideMiddle = line_sizes[i].shortSideMiddle;
                        area = sizeX * sizeY;
                        rect = rects[i];
                        rect.points(rect_points);
                        rectCenter.x = static_cast<float>(rect.center.x);
                        rectCenter.y = static_cast<float>(rect.center.y);
                        rect.angle = getLineSlope(shortSideMiddle, rectCenter);
                        if (sizeY > conf.XTimesYMin * sizeX
                            && sizeY < conf.XTimesYMax * sizeX
                            && sizeY < conf.maxY) {
                            dashLines[cntDash] = createLineFromRect(&rect, sizeY);
                            cntDash++;
                        }
                        else if (sizeY > sizeX && sizeY > (conf.maxY / 2)
                                 && area < conf.maxArea * 10000) {
                            solidLines[cntSolid] = createLineFromRect(&rect, sizeY);
                            cntSolid++;
                        }
                    }

                    const float MIN_ANGLE{15.0f};
                    for (int j = 0; j < cntSolid; j++) {
                        float a = tanf(static_cast<float>(M_PI) * solidLines[j].slope / 180.0f);
                        cv::Point center;
                        center.x = (solidLines[j].p1.x + solidLines[j].p2.x) / 2;
                        center.y = (solidLines[j].p1.y + solidLines[j].p2.y) / 2;
                        float b = center.y - center.x * a;
                        if ((solidLines[j].slope > MIN_ANGLE - 5
                             && std::max(solidLines[j].p1.x, solidLines[j].p2.x) > width / 2)
                            || (solidLines[j].slope < (-1) * (MIN_ANGLE - 5)
                                && std::min(solidLines[j].p1.x, solidLines[j].p2.x) < width / 2)) {
                            for (int l = 0; l < cntDash; l++) {
                                cv::Point dashCenter;
                                dashCenter.x = (dashLines[l].p1.x + dashLines[l].p2.x) / 2;
                                dashCenter.y = (dashLines[l].p1.y + dashLines[l].p2.y) / 2;
                                float res = a * dashCenter.x + b;
                                if (res > dashCenter.y) {
                                    dashLines[l] = dashLines[cntDash - 1];
                                    cntDash--;
                                    l--;
                                }
                            }
                            for (int k = j + 1; k < cntSolid; k++) {
                                cv::Point sldCenter;
                                sldCenter.x = (solidLines[k].p1.x + solidLines[k].p2.x) / 2;
                                sldCenter.y = (solidLines[k].p1.y + solidLines[k].p2.y) / 2;
                                float res = a * sldCenter.x + b;
                                if (res > sldCenter.y) {
                                    solidLines[k] = solidLines[cntSolid - 1];
                                    cntSolid--;
                                    k--;
                                }
                            }
                        }
                    }

                    //Dash also positioned too high on the image or too left or too right
                    int maxDashY = 0;
                    for (int i = 0; i < cntDash; i++) {
                        CustomLine l = dashLines[i];
                        int dashCenterX = (l.p1.x + l.p2.x) / 2;
                        int dashCenterY = (l.p1.y + l.p2.y) / 2;
                        if ((l.slope < MIN_ANGLE) && (l.slope > ((-1) * MIN_ANGLE))
                            || (dashCenterY < height / 15) || (dashCenterX > 19 * width / 20)) //|| (dashCenterX < w/20) too left //too high
                        {
                            dashLines[i] = dashLines[cntDash - 1];
                            cntDash--;
                            if (i > 0) {
                                i--;
                            }
                        }

                        if (maxDashY < std::max(dashLines[i].p1.y, dashLines[i].p2.y)) {
                            maxDashY = std::max(dashLines[i].p1.y, dashLines[i].p2.y);
                        }
                    }

                    if (VERBOSE) {
                        const cv::Scalar RED(0, 0, 255);
                        const cv::Scalar BLUE(255, 0, 0);

                        cv::Mat imageWithClassifiedLines;
                        originalImage.copyTo(imageWithClassifiedLines);

                        for(auto i = 0u; i < dashLines.size(); i++) {
                            cv::line(imageWithClassifiedLines, dashLines[i].p1, dashLines[i].p2, RED, 3, 8, 0);
                        }
                        for(auto i = 0u; i < solidLines.size(); i++) {
                            cv::line(imageWithClassifiedLines, solidLines[i].p1, solidLines[i].p2, BLUE, 3, 8, 0);
                        }

                        std::stringstream sstr;
                        sstr << sharedMemory->name() << "-classified lines";
                        const std::string windowName = sstr.str();
                        cv::imshow(windowName.c_str(), imageWithClassifiedLines);
                        cv::waitKey(1);
                    }
                }

                // TODO:
                // Compute Vanishing point
                // Go through list of lines:
                // if slope is < 0: pointing to right, otherwise to left
                // pick the longest
                // discard angles that don't make sense.
                // compute VP

                // Apply Kalman filter
                cv::Mat prediction = KF.predict();
                cv::Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

//                measurement(0) = VP.x;
//                measurement(1) = VP.y;

                // The update phase 
                cv::Mat estimated = KF.correct(measurement);

                cv::Point statePt(estimated.at<float>(0), estimated.at<float>(1));
//                cv::Point measPt(measurement(0), measurement(1));

                // Derive steering values.


                ////////////////////////////////////////////////////////////////
                opendlv::proxy::ActuationRequest ar;
                ar.acceleration(0).steering(0).isValid(true);
                od4.send(ar);
            }

            if (nullptr != iplimage) {
                cvReleaseImageHeader(&iplimage);
            }
        }
        retCode = 0;
    }
    return retCode;
}

