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

#include "theia/data_loader/bundler_binary_file.h"

#include <Eigen/Core>
#include <theia/theia.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace theia {

// The sift key file has the following format:
//
// number_of_keypoints sift_descriptor_dimensions (both as ints)
// for each descriptor:
//   row col scale orientation (all as floats)
//   128 ints describing sift descriptor. Normalizing this 128-vector to unit
//     length will yield the float sift descriptor.
bool ReadSiftKeyBinaryFile(const std::string& sift_key_file,
                           std::vector<Eigen::Vector2d>* feature_position,
                           std::vector<Eigen::VectorXf>* descriptor,
                           std::vector<Keypoint>* keypoint) {
  std::ifstream ifs(sift_key_file.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    LOG(ERROR) << "Could not read the sift key binary file from "
               << sift_key_file;
    return false;
  }

  // Number of descriptors and the length of the descriptors.
  int num_descriptors, len;
  ifs.read(reinterpret_cast<char*>(&num_descriptors), sizeof(num_descriptors));
  ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
  CHECK_EQ(len, 128);

  feature_position->reserve(num_descriptors);
  descriptor->reserve(num_descriptors);
  if (keypoint != nullptr) {
    keypoint->reserve(num_descriptors);
  }
  for (int i = 0; i < num_descriptors; i++) {
    // Keypoint params = y, x, scale, orientation.
    float keypoint_params[4];
    ifs.read(reinterpret_cast<char*>(keypoint_params),
             4 * sizeof(keypoint_params[0]));

    // Set 2D position
    feature_position->push_back(
        Eigen::Vector2d(keypoint_params[1], keypoint_params[0]));
    // Set keypoint if the output variable is not NULL.
    if (keypoint != nullptr) {
      Keypoint kp(keypoint_params[1], keypoint_params[0], Keypoint::SIFT);
      kp.set_scale(keypoint_params[2]);
      kp.set_orientation(keypoint_params[3]);
      keypoint->push_back(kp);
    }

    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> int_desc(128);
    ifs.read(reinterpret_cast<char*>(int_desc.data()),
             int_desc.size() * sizeof(int_desc[0]));
    Eigen::VectorXf float_descriptor = int_desc.cast<float>();
    float_descriptor /= 255.0;
    descriptor->push_back(float_descriptor);
  }
  ifs.close();

  return true;
}

// Outputs the SIFT features in the same format as Lowe's sift key files, but
// stores it as a binary file for faster loading.
bool WriteSiftKeyBinaryFile(
    const std::string& output_sift_key_file,
    const std::vector<Eigen::Vector2d>& feature_position,
    const std::vector<Eigen::VectorXf>& descriptor,
    const std::vector<Keypoint>& keypoint) {
  std::ofstream ofs(output_sift_key_file.c_str(),
                    std::ios::out | std::ios::binary);
  if (!ofs.is_open()) {
    LOG(ERROR) << "Could not write the sift key binary file to "
               << output_sift_key_file;
    return false;
  }

  // Output number of descriptors and descriptor length.
  const int num_descriptors = descriptor.size();
  const int len = 128;
  ofs.write(reinterpret_cast<const char*>(&num_descriptors),
            sizeof(num_descriptors));
  ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));

  // Output the sift descriptors.
  for (int i = 0; i < num_descriptors; i++) {
    // Output the keypoint information (x, y, scale, orientation).
    float kp[4];
    kp[0] = keypoint[i].y();
    kp[1] = keypoint[i].x();
    kp[2] = keypoint[i].scale();
    kp[3] = keypoint[i].orientation();
    ofs.write(reinterpret_cast<const char*>(kp), 4 * sizeof(kp[0]));

    // Output the descriptor.
    const Eigen::VectorXf float_scaled_desc = descriptor[i] * 255.0;
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> int_desc =
        float_scaled_desc.cast<uint8_t>();
    ofs.write(reinterpret_cast<const char*>(int_desc.data()),
              int_desc.size() * sizeof(int_desc[0]));
  }
  ofs.close();
  return true;
}

// The bundle files contain the estimated scene and camera geometry have the
// following format:
//     # Bundle file v0.3
//     <num_cameras> <num_points>   [two integers]
//     <camera1>
//     <camera2>
//        ...
//     <cameraN>
//     <point1>
//     <point2>
//        ...
//     <pointM>
// Each camera entry <cameraI> contains the estimated camera intrinsics and
// extrinsics, and has the form:
//     <f> <k1> <k2>   [the focal length, followed by two radial distortion
//                      coeffs]
//     <R>             [a 3x3 matrix representing the camera rotation]
//     <t>             [a 3-vector describing the camera translation]
// The cameras are specified in the order they appear in the list of images.
//
// Each point entry has the form:
//     <position>      [a 3-vector describing the 3D position of the point]
//     <color>         [a 3-vector describing the RGB color of the point]
//     <view list>     [a list of views the point is visible in]
//
// The view list begins with the length of the list (i.e., the number of cameras
// the point is visible in). The list is then given as a list of quadruplets
// <camera> <key> <x> <y>, where <camera> is a camera index, <key> the index of
// the SIFT keypoint where the point was detected in that camera, and <x> and
// <y> are the detected positions of that keypoint. Both indices are 0-based
// (e.g., if camera 0 appears in the list, this corresponds to the first camera
// in the scene file and the first image in "list.txt"). The pixel positions are
// floating point numbers in a coordinate system where the origin is the center
// of the image, the x-axis increases to the right, and the y-axis increases
// towards the top of the image. Thus, (-w/2, -h/2) is the lower-left corner of
// the image, and (w/2, h/2) is the top-right corner (where w and h are the
// width and height of the image).
bool ReadBundleBinaryFile(const std::string& bundle_file,
                         std::vector<theia::Camera>* camera,
                         std::vector<Eigen::Vector3d>* world_points,
                         std::vector<Eigen::Vector3f>* world_points_color,
                         std::vector<BundleViewList>* view_list) {
  std::ifstream ifs(bundle_file.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    LOG(ERROR) << "Could not read the binary bundle file from "
               << bundle_file;
    return false;
  }

  // Number of 3d points.
  int num_cameras, num_points;
  ifs.read(reinterpret_cast<char*>(&num_cameras), sizeof(num_cameras));
  ifs.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));

  // Resize the camera and points vectors accordingly.
  camera->clear();
  camera->resize(num_cameras);
  // Read each 3D point.
  for (int i = 0; i < num_cameras; i++) {
    Camera& current_camera = camera->at(i);

    // Internal params are focal length and two radial distortion params.
    double internal_params[3];
    ifs.read(reinterpret_cast<char*>(internal_params),
             3 * sizeof(internal_params[0]));

    // Set the calibration matrix.
    const Eigen::Matrix3d calibration_matrix = Eigen::DiagonalMatrix<double, 3>(
        internal_params[0], internal_params[0], 1.0);

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotation_row_major;
    ifs.read(reinterpret_cast<char*>(rotation_row_major.data()),
             sizeof(rotation_row_major));

    // Output translation.
    Eigen::Vector3d translation;
    ifs.read(reinterpret_cast<char*>(translation.data()),
             sizeof(translation));

    // Initialize the pose.
    current_camera.pose_
        .InitializePose(rotation_row_major, translation, calibration_matrix,
                        internal_params[1], internal_params[2], 0.0, 0.0);

    if ((i + 1) % 100 == 0 || i == num_cameras - 1) {
      std::cout << "\r Reading parameters for camera " << i + 1 << " / "
                << num_cameras << std::flush;
    }
  }
  std::cout << std::endl;

  // Read in the point data.
  world_points->clear();
  world_points->resize(num_points);
  world_points_color->clear();
  world_points_color->resize(num_points);
  view_list->clear();
  view_list->reserve(num_points);

  for (int i = 0; i < num_points; i++) {
    // Read in 3D point.
    ifs.read(reinterpret_cast<char*>(world_points->at(i).data()),
             sizeof(world_points->at(i)));

    // Read in color of the point.
    ifs.read(reinterpret_cast<char*>(world_points_color->at(i).data()),
             sizeof(world_points_color->at(i)));

    // Read in the view list.
    int num_views;
    ifs.read(reinterpret_cast<char*>(&num_views), sizeof(num_views));
    // Initialize a BundleViewList such that it allocates memory for all views
    // in the view list. Then we can read in the view list in one shot.
    BundleViewList bl_list(num_views);
    ifs.read(reinterpret_cast<char*>(bl_list.data()), sizeof(bl_list));
    view_list->push_back(bl_list);

    if ((i + 1) % 100 == 0 || i == num_points - 1) {
      std::cout << "\r Reading parameters for point " << i + 1 << " / "
                << num_points << std::flush;
    }
  }
  std::cout << std::endl;
  ifs.close();
  return true;
}

// Outputs the reconstruction contents as a Bundler file in binary format. This
// is so that it may be loaded more quickly.
bool WriteBundleBinaryFile(
    const std::string& output_bundle_file,
    const std::vector<Camera>& camera,
    const std::vector<Eigen::Vector3d>& world_points,
    const std::vector<Eigen::Vector3f>& world_points_color,
    const std::vector<BundleViewList>& view_list) {
  std::ofstream ofs(output_bundle_file.c_str(),
                    std::ios::out | std::ios::binary);
  if (!ofs.is_open()) {
    LOG(ERROR) << "Could not write the bundle binary file to "
               << output_bundle_file;
    return false;
  }

  // Write number of cameras and number of points.
  const int num_cameras = camera.size();
  const int num_points = world_points.size();
  ofs.write(reinterpret_cast<const char*>(&num_cameras), sizeof(num_cameras));
  ofs.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));

  // Write in camera parameters.
  for (int i = 0; i < num_cameras; i++) {
    // Write internal params: focal length and two radial distortion params.
    double internal_params[3];
    internal_params[0] = camera[i].pose_.focal_length();
    double unused_distortion[2];
    camera[i].pose_
        .radial_distortion(&(internal_params[1]), &(internal_params[2]),
                           &(unused_distortion[0]), &(unused_distortion[1]));

    ofs.write(reinterpret_cast<char*>(internal_params),
              3 * sizeof(internal_params[0]));

    // Write rotation matrix. Must transpose it so that it is in row-major order
    // (this is how Bundler stores it, so we want to be consistent).
    const Eigen::Matrix3d rotation_row_major =
        camera[i].pose_.rotation_matrix().transpose();
    ofs.write(reinterpret_cast<const char*>(rotation_row_major.data()),
              sizeof(rotation_row_major));

    // Write translation.
    const Eigen::Vector3d translation = camera[i].pose_.translation();
    ofs.write(reinterpret_cast<const char*>(translation.data()),
              sizeof(translation));

    if ((i + 1) % 100 == 0 || i == num_cameras - 1) {
      std::cout << "\r Writing parameters for camera " << i + 1 << " / "
                << num_cameras << std::flush;
    }
  }
  std::cout << std::endl;

  // Write in point parameters.
  for (int i = 0; i < num_points; i++) {
    // Write position.
    ofs.write(reinterpret_cast<const char*>(world_points[i].data()),
              sizeof(world_points[i]));

    // Write color.
    ofs.write(reinterpret_cast<const char*>(world_points_color[i].data()),
              sizeof(world_points_color[i]));

    // Write number of views.
    const int num_views = view_list[i].size();
    ofs.write(reinterpret_cast<const char*>(&num_views), sizeof(num_views));

    // Write view list all in one shot.
    ofs.write(reinterpret_cast<const char*>(view_list[i].data()),
              sizeof(view_list[i]));

    if ((i + 1) % 100 == 0 || i == num_points - 1) {
      std::cout << "\r Writing parameters for point " << i + 1 << " / "
                << num_points << std::flush;
    }
  }
  std::cout << std::endl;

  ofs.close();
  return true;
}

}  // namespace theia
