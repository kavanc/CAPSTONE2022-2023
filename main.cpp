#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
    String casc_path = "cpp_version/cascade.xml";

    CascadeClassifier knife_cascade;
    knife_cascade.load(casc_path);

    VideoCapture cap(0);

    cap.set(CAP_PROP_FRAME_WIDTH, 100);

    auto pTime = chrono::high_resolution_clock::now();
    int frameCount = 0;
    cv::Mat frame;

    while (true)
    {
        Mat img;
        cap >> img;
        cvtColor(img, img, COLOR_RGB2GRAY);
        std::vector<Rect> knife;
        knife_cascade.detectMultiScale(img, knife, 1.01, 11, 0, Size(90, 90), Size(180, 180));

        Rect max_coords = Rect(0, 0, 0, 0);
        for (Rect box : knife)
        {
            if (box.width > max_coords.width)
            {
                max_coords = box;
            }
        }

        rectangle(img, max_coords, Scalar(255, 0, 0), 2);

        auto cTime = chrono::high_resolution_clock::now();
        frameCount++;

        if (frameCount % 30 == 0)
        {
            chrono::duration<double> elapsed = cTime - pTime;
            double fps = frameCount / elapsed.count();
            int roundedFps = round(fps);
            cout << "FPS: " << fps << endl;
            pTime = cTime;
            frameCount = 0;
        }

        imshow("Result", img);
        if (waitKey(1) == 'q')
        {
            break;
        }
    }
    return 0;
}