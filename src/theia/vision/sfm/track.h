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

#ifndef THEIA_VISION_SFM_TRACK_H_
#define THEIA_VISION_SFM_TRACK_H_

#include <Eigen/Core>
#include <unordered_set>

#include "theia/vision/sfm/types.h"

namespace theia {

class Track {
 public:
  Track();
  explicit Track(const TrackId track_id);
  ~Track() {}

  TrackId Id() const;
  int NumViews() const;

  // Returns true if the view observes this track.
  bool IsViewVisible(const ViewId view_id) const;

  // Is the 3D point of the track estimated?
  void SetEstimated(const bool is_estimated);
  bool IsEstimated() const;

  // The homogeneous 3D point corresponding to track. This point will be valid
  // when IsEstimated() is true.
  const Eigen::Vector4d& Point() const;
  Eigen::Vector4d* MutablePoint();

  // Add the view to the track.
  void AddView(const ViewId view_id);
  bool RemoveView(const ViewId view_id);

  const std::unordered_set<ViewId>& ViewIds() const;

 private:
  TrackId id_;
  bool is_estimated_;
  std::unordered_set<ViewId> view_ids_;
  Eigen::Vector4d point_;
};

}  // namespace theia

#endif  // THEIA_VISION_SFM_TRACK_H_
