// Copyright (C) 2013 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <time.h>
#include <theia/theia.h>
#include <string>
#include <vector>

DEFINE_string(img_input_dir, "input", "Directory of two input images.");
DEFINE_string(img_output_dir, "output", "Name of output image file.");

using theia::CascadeHashingFeatureMatcher;
using theia::FloatImage;
using theia::FeatureMatcherOptions;
using theia::ImageCanvas;
using theia::Keypoint;
using theia::L2;
using theia::SiftDescriptorExtractor;

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  FloatImage left_image(FLAGS_img_input_dir + std::string("/img1.png"));
  FloatImage right_image(FLAGS_img_input_dir + std::string("/img2.png"));

  ImageCanvas image_canvas;
  LOG(INFO) << "adding left image";
  image_canvas.AddImage(left_image);
  LOG(INFO) << "adding right image";
  image_canvas.AddImage(right_image);
  LOG(INFO) << "writing";
  image_canvas.Write(FLAGS_img_output_dir +
                     std::string("/sift_descriptors.png"));

  // Detect keypoints.
  VLOG(0) << "detecting keypoints";
  SiftDescriptorExtractor sift_detector;
  CHECK(sift_detector.Initialize());
  std::vector<Keypoint> left_keypoints;
  std::vector<Eigen::VectorXf> left_descriptors;
  sift_detector.DetectAndExtractDescriptors(left_image,
                                            &left_keypoints,
                                            &left_descriptors);
  VLOG(0) << "detected " << left_descriptors.size()
          << " descriptors in left image.";

  VLOG(0) << "detecting keypoints";
  std::vector<Keypoint> right_keypoints;
  std::vector<Eigen::VectorXf> right_descriptors;
  sift_detector.DetectAndExtractDescriptors(right_image,
                                            &right_keypoints,
                                            &right_descriptors);
  VLOG(0) << "detected " << right_descriptors.size()
          << " descriptors in right image.";

  // Match descriptors!
  CascadeHashingFeatureMatcher image_matcher;
  FeatureMatcherOptions options;
  std::vector<theia::FeatureMatch> matches;
  clock_t t = clock();
  image_matcher.Match(options,
                      left_descriptors,
                      right_descriptors,
                      &matches);
  t = clock() - t;
  VLOG(0) << "It took " << (static_cast<float>(t)/CLOCKS_PER_SEC)
          << " to match SIFT descriptors";

  // Get an image canvas to draw the features on.
  image_canvas.DrawMatchedFeatures(0, left_keypoints,
                                   1, right_keypoints,
                                   matches,
                                   0.1);
  LOG(INFO) << "writing";
  image_canvas.Write(FLAGS_img_output_dir +
                     std::string("/sift_descriptors_matched.png"));
}
