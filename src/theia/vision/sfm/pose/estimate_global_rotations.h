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

#ifndef THEIA_VISION_SFM_POSE_ESTIMATE_GLOBAL_ROTATIONS_H_
#define THEIA_VISION_SFM_POSE_ESTIMATE_GLOBAL_ROTATIONS_H_

#include <Eigen/Core>
#include <unordered_map>

#include "theia/util/hash.h"
#include "theia/vision/sfm/types.h"

namespace theia {

// Struct for holding pairwise relative rotations. The rotation specifies the
// transformation from first to second camera. The weight given will determine
// how the pairwise relationship affects the overall system.
//
// TODO(csweeney): Analyze the memory overhead of using rotation matrices
// instead of angle-axis or quaternions.
struct RelativeRotation {
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  double weight = 1.0;
};

// Computes the orientation of views in a global frame given pairwise relative
// rotations between the views. This is done with a linear approximation to
// rotation averaging.
//
// This linear solution follows the method in "Robust Rotation and Translation
// Estimation in Multiview Geometry" by Martinec and Pajdla (CVPR 2007).
bool EstimateGlobalRotationsLinear(
    const std::unordered_map<ViewIdPair, RelativeRotation>& relative_rotations,
    std::unordered_map<ViewId, Eigen::Matrix3d>* global_orientations);

}  // namespace theia

#endif  // THEIA_VISION_SFM_POSE_ESTIMATE_GLOBAL_ROTATIONS_H_
