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

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>

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

                // Display image.
                if (VERBOSE) {
                    cv::imshow(sharedMemory->name().c_str(), img);
                    cv::waitKey(1);
                }

                ////////////////////////////////////////////////////////////////
                {
                    // Copy image.
                    img.copyTo(originalImage);
                }

                ////////////////////////////////////////////////////////////////
                {
                    // Turn image into grayscale.
                    cv::cvtColor(img, img, CV_BGR2GRAY);

                    if (VERBOSE) {
                        std::stringstream sstr;
                        sstr << sharedMemory->name() << "-grayscale";
                        const std::string windowName = sstr.str();
                        cv::imshow(windowName.c_str(), img);
                        cv::waitKey(1);
                    }
                }

                ////////////////////////////////////////////////////////////////
                int32_t height = img.size().height;
                int32_t width = img.size().width;
                {
                    // Cropping image.
                    img = img(cv::Rect(1, 6 * height / 16 - 1, width - 1, 8 * height / 16 - 1));

                    if (VERBOSE) {
                        std::stringstream sstr;
                        sstr << sharedMemory->name() << "-cropped";
                        const std::string windowName = sstr.str();
                        cv::imshow(windowName.c_str(), img);
                        cv::waitKey(1);
                    }
                }

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
                // Example for creating and sending a message to other microservices; can
                // be removed when not needed.
                opendlv::proxy::AngleReading ar;
                ar.angle(123.45f);
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

