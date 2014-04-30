// Copyright (C) 2014 The Regents of the University of California (Regents).
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

#include <Eigen/Core>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <theia/theia.h>

#include <string>
#include <vector>

DEFINE_string(input_bundle_file, "",
              "Input bundle text file to convert. Should end in .out");

DEFINE_string(output_bundle_file, "",
              "Output bundle file in binary format. Should end in .bin");

// This function will load the sift descriptors from the key text files and
// convert them to binary files.
bool ConvertBundleFile(const std::string& input_bundle_file,
                       const std::string& output_bundle_file) {
  // Read text file.
  std::vector<theia::Camera> camera;
  std::vector<Eigen::Vector3d> world_points;
  std::vector<Eigen::Vector3f> world_points_color;
  std::vector<theia::BundleViewList> view_list;
  CHECK(theia::ReadBundleTextFile(input_bundle_file, &camera, &world_points,
                                   &world_points_color, &view_list));

  // Write binary file.
  CHECK(
      theia::WriteBundleBinaryFile(output_bundle_file, camera, world_points,
                                   world_points_color, view_list));
  return true;
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Load the SIFT descriptors into the cameras.
  CHECK(ConvertBundleFile(FLAGS_input_bundle_file, FLAGS_output_bundle_file));

  return 0;
}
