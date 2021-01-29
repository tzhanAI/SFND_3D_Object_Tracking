/****************************************************************************************
 *                           YOLO DETECT & CLASSIFY OBJECTS
****************************************************************************************/
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "objectDetection2D.hpp"


using namespace std;

// detects objects in an image using the YOLO library and a set of pre-trained objects from the COCO database;
// a set of 80 classes is listed in "coco.names" and pre-trained weights are stored in "yolov3.weights"
void detectObjects(cv::Mat& img, std::vector<BoundingBox>& bBoxes, float confThreshold, float nmsThreshold, 
                   std::string basePath, std::string classesFile, std::string modelConfiguration, std::string modelWeights, bool bVis)
{
    // load all pretrained the classes's name, i.e. person, car..
    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line))
	 classes.push_back(line);
    
    // load neural network
	/* A blob is a N dimentional array stored in contiguous fashion (NCHW)
	 * such as, batch num. N x channel C x height H x width W
	 * For RGB, C = 3. For a training batch of 256 images, N would be 256. */
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    
    // generate 4D blob from input image
    cv::Mat blob;
    vector<cv::Mat> netOutput;				// to store output raw bounding boxes from YOLO
    double scalefactor = 1/255.0;			// The pixel values are scaled to a range of 0 to 1
    cv::Size size = cv::Size(416, 416);		// adjusts the size of the image to the specified size, cv::Size(608, 608);
    cv::Scalar mean = cv::Scalar(0,0,0);
    bool swapRB = false;
    bool crop = false;
    cv::dnn::blobFromImage(img, blob, scalefactor, size, mean, swapRB, crop);
    
    // Get names of output layers
    vector<cv::String> names;
	/* get indices of all unconnected output layers,
	 * which are in fact the last layer of the network. */
    vector<int> outLayers = net.getUnconnectedOutLayers();
	// get  names of all layers in the network
    vector<cv::String> layersNames = net.getLayerNames(); 
    
	// Get the names of the output layers in layersNames
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) 
        names[i] = layersNames[outLayers[i] - 1];
    
    // invoke forward propagation through network
    net.setInput(blob);
    net.forward(netOutput, names);
    
	// every box from YOLOv3 has a confidence score
    // Scan through all bounding boxes and keep only the ones with high confidence
    vector<int> classIds;
	vector<float> confidences;
	vector<cv::Rect> boxes;

	// iter over every cv::Mat in vector netOutput, size() is the number of blob classes?
    for (size_t i = 0; i < netOutput.size(); ++i)
    {
		// iter over every row in cv::Mat
        float* data = (float*)netOutput[i].data;
        for (int j = 0; j < netOutput[i].rows; ++j, data += netOutput[i].cols)
        {
			// cv::Mat i and row j, col range from index 5 to end, stores confidence scores of YOLO classes
            cv::Mat scores = netOutput[i].row(j).colRange(5, netOutput[i].cols);
	
            cv::Point classId;	// person, car..provided in YOLO file
            double confidence;
            
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
            if (confidence > confThreshold)
            {
                cv::Rect box; int cx, cy;
				// box's center (cx, cy)
                cx = (int)(data[0] * img.cols);
                cy = (int)(data[1] * img.rows);
				// box's top left corner (x,y) and width and height
                box.width = (int)(data[2] * img.cols);
                box.height = (int)(data[3] * img.rows);
                box.x = cx - box.width/2; // left
                box.y = cy - box.height/2; // top
                
                boxes.push_back(box);
                classIds.push_back(classId.x);
                confidences.push_back((float)confidence);
            }
        }
    }
    
    // perform non-maxima suppression
    vector<int> indices;					// Non-maximum suppression output
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	
    for(auto it=indices.begin(); it!=indices.end(); ++it) {
        
        BoundingBox bBox;
        bBox.roi = boxes[*it];				// 2d region of interest, cv::Rect
        bBox.classID = classIds[*it];		// i.e. YOLO file, i.e. person, car..
        bBox.confidence = confidences[*it];
        bBox.boxID = (int)bBoxes.size(); 	// zero-based unique identifier for this bounding box
        
        bBoxes.push_back(bBox);
    }
    
    // show results
    if(bVis) {
        
        cv::Mat visImg = img.clone();
        for(auto it=bBoxes.begin(); it!=bBoxes.end(); ++it) {
             
            // retrive the curr bounding box info
            int top, left, width, height;
            top = (*it).roi.y;
            left = (*it).roi.x;
            width = (*it).roi.width;
            height = (*it).roi.height;
			// Draw rectangle displaying the bounding box
            cv::rectangle(visImg, cv::Point(left, top), cv::Point(left+width, top+height),cv::Scalar(0, 255, 0), 2);
            
			// format label info
            string label = cv::format("%.2f", (*it).confidence);
            label = "boxID: " + to_string((*it).boxID) + "; " + classes[((*it).classID)] + ":" + label ;
        
            // Display label at the top of the bounding box
            int baseLine;	//y-coordinate of the baseline relative to the bottom-most text point
            cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
            top = max(top, labelSize.height);
            rectangle(visImg, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
            cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0,0,0),1);
            
        }
        
        string windowName = "Object classification";
        cv::namedWindow( windowName, 1 );
        cv::imshow( windowName, visImg );
        cv::waitKey(0); // wait for key to be pressed
    }
}
