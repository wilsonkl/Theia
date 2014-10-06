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

#ifndef THEIA_VISION_SFM_VIEW_H_
#define THEIA_VISION_SFM_VIEW_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "theia/vision/sfm/camera/camera.h"
#include "theia/vision/sfm/types.h"
#include "theia/vision/sfm/feature.h"
#include "theia/vision/sfm/view_metadata.h"

namespace theia {

// A view is a core object within the SfM pipeline representing information
// about an image. It contains information about the image features, EXIF data,
// camera pose information, and its status within the Model (i.e. has the pose
// been estimated).
class View {
 public:
  View();
  View(const std::string& name, const ViewId id);

  ~View() {}

  // Typically the filename of the image.
  const std::string& Name() const;

  // Id of the image within a model.
  ViewId Id() const;

  bool IsTrackVisible(const TrackId track_id) const;

  // Set/get whether the view has been posed within a model.
  void SetEstimated(bool is_estimated);
  bool IsEstimated() const;

  // The camera pose of the view.
  const class Camera& Camera() const;
  class Camera* MutableCamera();

  // Metadata about the image. Typically this is simply information taken from
  // the EXIF data.
  const ViewMetadata& Metadata() const;
  ViewMetadata* MutableMetadata();

  // The number of features in this view.
  int NumFeatures() const;

  // Return all track ids.
  std::vector<TrackId> TrackIds() const;

  // Return the Feature given a TrackId if it exists, or return NULL
  // if the view does not see the track.
  const Feature* GetFeature(const TrackId track_id) const;

  // Add a feature to the view.
  void AddFeature(const Feature& feature);

  // Remove a feature from the view.
  bool RemoveFeature(const TrackId track_id);

 private:
  std::string name_;
  ViewId id_;
  ViewMetadata metadata_;

  bool is_estimated_;
  class Camera camera_;
  std::unordered_map<TrackId, Feature> features_;
};

}  // namespace theia

#endif  // THEIA_VISION_SFM_VIEW_H_
