#ifndef VIEWER_H_
#define VIEWER_H_

#include "odom.h"

#include <thread>
#include <mutex>
#include <pangolin/pangolin.h>

using namespace std;


namespace sv::dsol {

    //  Visualization for DSO

    /**
     * viewer implemented by pangolin
     */
    class PangolinDSOViewer {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        PangolinDSOViewer(int w, int h, bool startRunThread = true);

        ~PangolinDSOViewer();

        void run();

        void close();

        void publishPointPoseFrame(OdomStatus& status);
        void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw);
        // void pushLiveFrame( shared_ptr<Frame> image);

        /* call on finish */
        void join();

    private:

        thread runThread;
        bool running = true;
        int w, h;
        std::vector<Sophus::SE3f> allFramePoses;  // trajectory
        std::vector<Eigen::Vector3f> mapPoints;
        cv::Mat frame;

        bool videoImgChanged = true;
        // 3D model rendering
        mutex model3DMutex;

        // timings
        struct timeval last_track;
        struct timeval last_map;

        std::deque<float> lastNTrackingMs;
        std::deque<float> lastNMappingMs;

    };

}

#endif // LDSO_VIEWER_H_
