
#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;	// must be false IF KNN will be used latter
    cv::Ptr<cv::DescriptorMatcher> matcher;
	
	// --------------------brutal force matcher -----------------------------
    if (matcherType.compare("MAT_BF") == 0)
    {
		int normType;
		if (descriptorType.compare("DES_BINARY") == 0)
			normType = cv::NORM_HAMMING;
		else
			normType = cv::NORM_L2;	// HOG gradient based
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
	// --------------------FLANN-based matcher -----------------------------
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
		/* OpenCV bug workaround : convert binary descriptors to floating point
		 * due to a bug in current OpenCV implementation */
        if (descSource.type() != CV_32F)
            descSource.convertTo(descSource, CV_32F);
		if (descRef.type() != CV_32F)
			descRef.convertTo(descRef, CV_32F);

		matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // -------------------------- select matching ---------------------------
    if (selectorType.compare("SEL_NN") == 0)
    {
		// Finds nearest neighbor (best match) for each descriptor in desc1
        matcher->match(descSource, descRef, matches); 
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {
		// k nearest neighbors (k=2) matching
		std::vector< std::vector<cv::DMatch> > knn_matches;
		matcher->knnMatch(descSource, descRef, knn_matches, 2);
		
		// filter matches using descriptor distance ratio test
		// filter out ambiguous matches using 1st and 2nd best matches
		const float ratio_thresh = 0.8f;
		for (size_t i = 0; i < knn_matches.size(); i++)
		{
		    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		    {
		        matches.push_back(knn_matches[i][0]);
		    }
		}
    }

	//solution
/*		double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;	*/
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    
	// --------------------BRISK Descriptor -----------------------------
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
	// --------------------BRIEF Descriptor -----------------------------
	else if (descriptorType.compare("BRIEF") == 0)
	{
		extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
	}
	// --------------------ORB Descriptor -----------------------------
	else if (descriptorType.compare("ORB") == 0)
	{
		extractor = cv::ORB::create();
	}
	// --------------------FREAK Descriptor -----------------------------
	else if (descriptorType.compare("FREAK") == 0)
	{
		extractor = cv::xfeatures2d::FREAK::create();
	}
	// --------------------SIFT Descriptor -----------------------------
	else if (descriptorType.compare("SIFT") == 0)
	{
		extractor = cv::SIFT::create();
	}
	// --------------------AKAZE Descriptor -----------------------------
    else
    {
		// AKAZE descriptors can only be used with KAZE or AKAZE keypoints.
		extractor = cv::AKAZE::create();
    }


    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
//    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
//    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the Harris detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
	// Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
	double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // visualize results
	if (bVis)
	{	
		string windowName = "Harris Corner Detector Response Matrix";
		cv::namedWindow(windowName, 4);
		cv::imshow(windowName, dst_norm_scaled);
		cv::waitKey(0);
	}

	double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows
	
//	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
//    cout << "Harris detection with n =" << keypoints.size() << " keypoints in " 
//	<< 1000 * t / 1.0 << " ms" << endl;	

    // visualize keypoints
	if (bVis)
	{
		string windowName = "Harris Corner Detection Results";
		cv::namedWindow(windowName, 5);
		cv::Mat visImage = dst_norm_scaled.clone();
		cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1),
						 cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		cv::imshow(windowName, visImage);
		cv::waitKey(0);
	}
}

// Detect keypoints in image using FAST, BRISK, ORB, AKAZE or SIFT according to string input
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
	// switch between different algorithms solving the same problem.
	cv::Ptr<cv::FeatureDetector> detector;
	
	// --------------------FAST detector -----------------------------
	if (detectorType.compare("FAST") == 0)
	{
		// FAST : Features from accelerated segment test
		int threshold = 30; // difference between intensity of the central pixel and pixels of a circle around this pixel
		bool bNMS = true;   // set TRUE to perform non-maxima suppression on keypoints
		// TYPE_9_16, TYPE_7_12, TYPE_5_8; 
		cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; 
		detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
	}
	// --------------------BRISK detector -----------------------------
	else if (detectorType.compare("BRISK") == 0)
	{
		// BRISK: Binary robust invariant scalable keypoints
		detector = cv::BRISK::create();
	}
	// --------------------SIFT detector -----------------------------
	else if (detectorType.compare("SIFT") == 0)
	{
		// SIFT: Scale invariant feature transform (HOG based)
		detector = cv::SIFT::create();
	}
	// --------------------Oriented BRIEF (ORB) detector ---------------
	else if (detectorType.compare("ORB") == 0)
	{
		// BRIEF: Binary Robust Independent Elementary Features
		int numOfFeatures = 10000;
		detector = cv::ORB::create(numOfFeatures);
	}
	// --------------------AKAZE ----------------------------------------
	else
	{
		// AKAZE: presented Accelerated-KAZE
		detector = cv::AKAZE::create();
	}	
	
	double t = (double)cv::getTickCount();
	detector->detect(img, keypoints);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
//	cout << detectorType << " with n= " << keypoints.size() << " keypoints in " << 
//												1000 * t / 1.0 << " ms" << endl;

	// visualize results
	if (bVis)
	{
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), 
							cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		string windowName = "detect Keypoints With Modern " + detectorType + " Results";
		cv::namedWindow(windowName, 1);	// If a window with the same name already exists, the function does nothing.
		imshow(windowName, visImage);
		cv::waitKey(0);
	}
}
