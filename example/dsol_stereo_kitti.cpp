#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include<vector>
#include<string>
#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

#include "extra.h"
#include "odom.h"
#include "cmap.h"
#include "dataset.h"
#include "logging.h"
#include "ocv.h"
#include "summary.h"
#include "viewer.h"

using namespace std;

ABSL_FLAG(bool, tbb, true, "use tbb");
ABSL_FLAG(int32_t, vis, 1, "visualization");
ABSL_FLAG(int32_t, wait, 1, "wait ms");

ABSL_FLAG(std::string,
          kitti_data_dir,
          "/home/spurs/dataset/kitti_00",
          "dataset dir");


ABSL_FLAG(int32_t, num_levels, 5, "number of levels");
ABSL_FLAG(int32_t, num_kfs, 4, "number of kfs");

ABSL_FLAG(int32_t, height, 376, "height");
ABSL_FLAG(int32_t, width, 1241, "width");

ABSL_FLAG(float, fx, 718.856, "fx");
ABSL_FLAG(float, fy, 718.856, "fy");
ABSL_FLAG(float, cx, 607.1928, "cx");
ABSL_FLAG(float, cy, 185.2157, "cy");
ABSL_FLAG(float, baseline, 0.53716, "baseline");


void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}


namespace sv::dsol {
Dataset dataset;
DirectOdometry odom_;
void Init() {

  {
    OdomCfg cfg;
    cfg.tbb = static_cast<int>(absl::GetFlag(FLAGS_tbb));
    cfg.vis = absl::GetFlag(FLAGS_vis);

    cfg.marg = false;
    cfg.num_kfs = absl::GetFlag(FLAGS_num_kfs);
    cfg.num_levels = absl::GetFlag(FLAGS_num_levels);
    cfg.vis_min_depth = 4.0;
    cfg.reinit = true;
    cfg.init_depth = false;
    cfg.init_stereo = true;
    cfg.init_align = true;
    cfg.min_track_ratio = 0.6;

    odom_.Init(cfg);
  }

  {
    int height = absl::GetFlag(FLAGS_height);
    int width = absl::GetFlag(FLAGS_width);

    cv::Mat intrin = cv::Mat::zeros(1, 5, CV_64FC1);
    intrin.at<double>(0) = absl::GetFlag(FLAGS_fx);
    intrin.at<double>(1) = absl::GetFlag(FLAGS_fy);
    intrin.at<double>(2) = absl::GetFlag(FLAGS_cx);
    intrin.at<double>(3) = absl::GetFlag(FLAGS_cy);
    intrin.at<double>(4) = absl::GetFlag(FLAGS_baseline);

    odom_.camera = Camera::FromMat({width, height}, intrin);
  }

  {
    const int num_kfs = absl::GetFlag(FLAGS_num_kfs);
    odom_.window.Resize(num_kfs);
  }

  {
    SelectCfg cfg;

    cfg.sel_level = 1;
    cfg.cell_size = 16;
    cfg.min_grad = 8;
    cfg.max_grad = 64;
    cfg.nms_size = 1;
    cfg.min_ratio = 0.4;
    cfg.max_ratio = 0.6;
    cfg.reselect = true;

    odom_.selector = PixelSelector(cfg);
  }
  {
    StereoCfg cfg;
    cfg.min_depth = 4.0;
    cfg.half_rows = 2;
    cfg.half_cols = 3;
    cfg.match_level = 3;
    cfg.refine_size = 1;
    cfg.min_zncc = 0.8;
    cfg.min_depth = 1.0;

    odom_.matcher = StereoMatcher(cfg);
  }
  {
    AlignCfg cfg;
    odom_.aligner = FrameAligner(cfg);
  }
  {
    AdjustCfg cfg;
    odom_.adjuster = BundleAdjuster(cfg);
  }

  odom_.cmap = GetColorMap("cm"); //, "jet"));

  LOG(INFO) << odom_;
}

void Run() {
  const int wait = absl::GetFlag(FLAGS_wait);

  MotionModel motion_;
  motion_.Init();
  TumFormatWriter writer("./result.txt");

  // Retrieve paths to images
  std::vector<string> vstrImageLeft;
  std::vector<string> vstrImageRight;
  std::vector<double> vTimestamps;
  LoadImages(absl::GetFlag(FLAGS_kitti_data_dir), vstrImageLeft, vstrImageRight, vTimestamps);
  
  int height = absl::GetFlag(FLAGS_height);
  int width = absl::GetFlag(FLAGS_width);

  PangolinDSOViewer pViewer = PangolinDSOViewer(width, height, true);

  for (int ni = 0; ni < vstrImageLeft.size(); ++ni) {

    LOG(INFO) << fmt::format(fmt::fg(fmt::color::red), "ind: {}", ni);

    // Read left and right images from file
    cv::Mat imLeft = cv::imread(vstrImageLeft[ni], cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
    cv::Mat imRight = cv::imread(vstrImageRight[ni], cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
    double tframe = vTimestamps[ni];

    cv::Mat depth0;

    float dt = vTimestamps[ni] - vTimestamps[ni-1];
    Sophus::SE3d dt_pred;
    if (motion_.Ok() && ni > 0) {
      dt_pred = motion_.PredictDelta(dt);
    }

    auto status = odom_.Estimate(imLeft, imRight, dt_pred, depth0);

    if (status.track.ok) {
        motion_.Correct(status.Twc(), dt);
        writer.Write(ni, status.Twc());
    }

    if (status.track.add_kf) 
        pViewer.publishPointPoseFrame(status);

  }
  std::cout << "please input any to quit: ";
  std::cin.get();

}

}  // namespace sv::dsol

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  sv::dsol::Init();
  sv::dsol::Run();
}
