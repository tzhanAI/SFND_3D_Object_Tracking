
#include <iostream>
#include <algorithm>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // 1. Convert current Lidar point into homogeneous coordinates and store it in the 4D vector X.
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // 2. project Lidar point into camera image plane
        Y = P_rect_xx * R_rect_xx * RT * X;

		// 3. transform Y back into Euclidean coordinates
        cv::Point pt; // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
			// sometimes ROIs that are too large and thus overlap into surrounding,
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

			if (enclosingBoxes.size() > 1)
                break;

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box IF its only enclosed by single bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }
    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // lidar sensor data (x, y, z) world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
			
			// find min and max of all points
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            /* Tranform lidar point (xw, yw) to image coordinates (top down view, origin at top left corner)
		 	 * note: For lidar measurements coming from the back of the vehicle (negative sign), will make x out of
		 	 * bound, thus they do not show up in the resulting image. */
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle 
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 2, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 0.5);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-125, bottom+25), cv::FONT_ITALIC, 1, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-125, bottom+75), cv::FONT_ITALIC, 1, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches,
							  std::vector<cv::DMatch> & kptMisMatches, std::vector<int> & matchIndexMultiBox)
{	
	double dist_mean = 0.0;		// the mean of all distances between key point matches
	double distThd   = 2.0;		// distThd * dist_mean is the threshold for outliers
	double shrinkFactor = 0.1;	// make bounding box smaller by this pct

	// iter over all matches
    for (auto it = kptMatches.begin(); it != kptMatches.end(); it++)
	{
		// iter over all key points in curr frame and check if within input bounding box
		cv::KeyPoint kpCurrFrame = kptsCurr[it->trainIdx];
		cv::KeyPoint kpPrevFrame = kptsPrev[it->queryIdx];

		// create a smaller box for front car in ego lane
		cv::Rect smallerBoundingBox;
		smallerBoundingBox.x = boundingBox.roi.x + shrinkFactor * boundingBox.roi.width / 2.0;
        smallerBoundingBox.y = boundingBox.roi.y + shrinkFactor * boundingBox.roi.height / 2.0;
        smallerBoundingBox.width = boundingBox.roi.width * (1 - shrinkFactor);
        smallerBoundingBox.height = boundingBox.roi.height * (1 - shrinkFactor);
		
		//  keeping only the points within the previous and current ROI
		if (smallerBoundingBox.contains(kpCurrFrame.pt) && 
			smallerBoundingBox.contains(kpPrevFrame.pt))
		{
			// this key point match pair belong to multiple bounding boxes
			if (matchIndexMultiBox[it - kptMatches.begin()] == 1)
				continue;	
			// this is the match in bb, not the same from outer loop
			boundingBox.kptMatches.push_back(*it);
			// dist between two matched keypoints from two successive frames
			dist_mean += cv::norm(kpCurrFrame.pt - kpPrevFrame.pt);
		}
	}

	// the mean of all distances between key point matches
	if (!boundingBox.kptMatches.empty())
		dist_mean /= boundingBox.kptMatches.size();
	else
	{
		cout << "No key point found within the current bounding box.";
		return;
	}

	// iter over all inbound matched key points and check outliers (cal * mean)
	for (size_t i = 0; i < boundingBox.kptMatches.size(); i++)
	{
		cv::KeyPoint kpCurrFrame = kptsCurr[boundingBox.kptMatches[i].trainIdx];
		cv::KeyPoint kpPrevFrame = kptsPrev[boundingBox.kptMatches[i].queryIdx];
		
		if (cv::norm(kpCurrFrame.pt - kpPrevFrame.pt) > distThd * dist_mean)
		{
			// remove outlier from vector
			kptMisMatches.push_back(*(boundingBox.kptMatches.begin() + i));
			boundingBox.kptMatches.erase(boundingBox.kptMatches.begin() + i);
			i--;
		}	
		else
			boundingBox.keypoints.push_back(kpCurrFrame);
	}
//cout << boundingBox.kptMatches.size() << " ";
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between 1. a distance of two key points in curr frame and 2. a distance of two key points in prev frame
	// stores the distance ratios for all combinations
	// any two key points can form a distance (line), the matching points in successive frame form another line 
    vector<double> distRatios;

	// outer keypoint loop
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { 
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

		// inner keypoint loop
        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { 
            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances 
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
			
			// compute distance ratios
			// avoid division by zero
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { 
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    } // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
	// 1. using mean distance ratio
    double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

    // 2. using median distance ratio to remove outlier influence
	std::sort(distRatios.begin(), distRatios.end());
	double medDistRatio;		// median
	int n = distRatios.size();
	
	if (n % 2 == 0)
		medDistRatio = 0.5 * (distRatios[n/2] + distRatios[n/2-1]);
	else
		medDistRatio = distRatios[n/2];

/*	// solution
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; 
*/

    double dT = 1 / frameRate;
	if (medDistRatio != 1)
		TTC = -dT / (1 - medDistRatio);		// use median
	else if (meanDistRatio != 1)
		TTC = -dT / (1 - meanDistRatio);	// use mean
	else
		TTC = NAN;							// no TTC can be calculated
}

// Comparator function to sort lidar point by x
bool cmp(LidarPoint& a, LidarPoint& b)
{ 
	return a.x < b.x; 
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
	double dT = 1.0 / frameRate;        					// time between two measurements in seconds
	double disPrev, disCurr;								// distances used by TTC computation
	
	std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), cmp);	// sort in ascending order based on distance in x
	std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), cmp); // sort in ascending order based on distance in x

	double minXPrev = lidarPointsPrev[0].x;					// closest, 2nd closest, 3rd closest in prev frame
	double secXPrev = lidarPointsPrev[1].x;
	double thrdXPrev = lidarPointsPrev[2].x;

	double minXCurr = lidarPointsCurr[0].x;					// closest, 2nd closest, 3rd closest in curr frame
	double secXCurr = lidarPointsCurr[1].x;
	double thrdXCurr = lidarPointsCurr[2].x;
//cout << minXCurr << " " << secXCurr << " " << thrdXCurr << "   ";

	double outlierThd = 0.02;	// if distance between most and 2nd closest points are within 0.02 assume not a outlier
	int start = 3, end = 13;	// otherwise compute the mean of the points in range of [start, end]
	
	// check outlier in prev frame
	if (secXPrev - minXPrev > outlierThd)
	{
		if (thrdXPrev - secXPrev > outlierThd)
		{
			// multiple outliers, need to use mean
			disPrev = std::accumulate(lidarPointsPrev.begin() + start, lidarPointsPrev.begin() + end, 0.0, 
										[](double sum, const LidarPoint &lidarPoint) { return sum + lidarPoint.x;});
			disPrev /= (double)(end - start);
		}		
		else
			disPrev = secXPrev;
	}
	else
		disPrev = minXPrev;

	// check outlier in curr frame
	if (secXCurr - minXCurr > outlierThd)
	{
		if (thrdXCurr - secXCurr > outlierThd)
		{
			// multiple outliers, need to use mean
			disCurr = std::accumulate(lidarPointsCurr.begin() + start, lidarPointsCurr.begin() + end, 0.0, 
										[](double sum, const LidarPoint &lidarPoint) { return sum + lidarPoint.x;});
			disCurr /= (double)(end - start);
		}		
		else
			disCurr = secXCurr;
	}
	else
		disCurr = minXCurr;
//cout << disCurr << " ";
    // compute TTC from both measurements using constant velocity model
	if (abs(disPrev - disCurr) > std::numeric_limits<double>::epsilon())	// prevent from dividing by 0
    	TTC = disCurr * dT / (disPrev - disCurr);
	else
		TTC = NAN;	// no TTC
}


// match list of 3D objects bounding boxes between current and previous frame
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame, std::vector<int>& matchIndexMultiBox)
{
	// to store the count look up table of all candidate pairs
	std::map<int, std::map<int, int>> boxMatchesCount;
	
    // iter over all key point matches
	for (auto it = matches.begin(); it != matches.end(); it++)
	{
		// in curr frame, find out by which bounding boxes key points are enclosed
		std::vector<int> currBoxIDs;
		cv::KeyPoint kpCurrFrame = currFrame.keypoints[it->trainIdx];
		for (auto it_curr = currFrame.boundingBoxes.begin(); it_curr != currFrame.boundingBoxes.end(); it_curr++)
		{	
			if (it_curr->roi.contains(kpCurrFrame.pt))
				currBoxIDs.push_back(it_curr->boxID);	
		}

		//in prev frame, find out by which bounding boxes key points are enclosed
		std::vector<int> prevBoxIDs;
		cv::KeyPoint kpPrevFrame = prevFrame.keypoints[it->queryIdx];
		for (auto it_prev = prevFrame.boundingBoxes.begin(); it_prev != prevFrame.boundingBoxes.end(); it_prev++)
		{	
			if (it_prev->roi.contains(kpPrevFrame.pt))
				prevBoxIDs.push_back(it_prev->boxID);	
		}
		
		// no matched bounding boxes
		if (currBoxIDs.empty() || prevBoxIDs.empty())
			continue;
		
		// store the key point matches that are enclosed by multiple bound boxes,to be used in the camera ttc 
		if (currBoxIDs.size() > 1)		
			matchIndexMultiBox[it - matches.begin()] = 1;

		// build all possible candidates and their occurrances
		for (size_t i = 0; i < prevBoxIDs.size(); i++)
			for (size_t j = 0; j < currBoxIDs.size(); j++)
			{
				// check if exist already
				if (boxMatchesCount.find(prevBoxIDs[i]) == boxMatchesCount.end() ||
					boxMatchesCount[prevBoxIDs[i]].find(currBoxIDs[j]) == boxMatchesCount[prevBoxIDs[i]].end())
				{
					boxMatchesCount[prevBoxIDs[i]].insert({currBoxIDs[j], 1});
				}
				else
				{
					boxMatchesCount[prevBoxIDs[i]][currBoxIDs[j]]++;
				}
			}
	}

	// match bounding boxes based on occurrances
	for (auto it1 = boxMatchesCount.begin(); it1 != boxMatchesCount.end(); it1++)
	{
		int boxID = -1;
		int count = -1;
		for (auto it2 = it1->second.begin(); it2 != it1->second.end(); it2++)
		{
			// use the pair with the max occurances
			if (it2->second > count)
			{
				boxID = it2->first;
				count = it2->second;
			}
		}
		// assign matched boxes
		bbBestMatches[it1->first] = boxID;
		//cout << it1->first << " " << boxID << endl;
	}
}
