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

#ifndef THEIA_VISION_SFM_MODEL_H_
#define THEIA_VISION_SFM_MODEL_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "theia/vision/sfm/track.h"
#include "theia/vision/sfm/types.h"
#include "theia/vision/sfm/view.h"

namespace theia {

struct Feature;

// A Model is the basic construct for structure from motion reconstructions. It
// contains a set of Views and a set of Tracks (each of which have a unique id)
// that encompass camera and 3D point information. This class is purely a
// container for reconstruction information, and another algorithm must actually
// create and estimate the reconstruction.
//
// We keep track of View names to help determine uniqueness of Views and also to
// make reading/writing Model easy.
class Model {
 public:
  Model();
  ~Model();

  // Add a track to the model with all of its features across views. The feature
  // will be to the View corresponding to the View name (i.e., the string) or a
  // new View will be created if a View with the view name does not already
  // exist. The track will not be estimated. The TrackId returned will be unique
  // or will be kInvalidTrackId if the method fails.
  TrackId AddTrack(const std::vector<std::pair<std::string, Feature> >& track);

  // Removes the track from the model and from any Views that observe this
  // track. Returns true on success and false on failure (e.g., the track does
  // not exist).
  bool RemoveTrack(const TrackId track_id);
  int NumTracks() const;

  // Returns the Track or a nullptr if the track does not exist.
  const class Track* Track(const TrackId track_id) const;
  class Track* MutableTrack(const TrackId track_id);

  // Return all TrackIds in the model.
  std::vector<TrackId> TrackIds() const;

  // Adds a view to the model with the default initialization. The ViewId
  // returned is guaranteed to be unique or will be kInvalidViewId if the method
  // fails.
  ViewId AddView(const std::string& view_name);

  // Removes the view from the model and removes all references to the view in
  // the tracks.
  //
  // NOTE: It is possible to have tracks of length 0 after this method is
  // executed.
  bool RemoveView(const ViewId view_id);
  int NumViews() const;

  // Returns the View or a nullptr if the track does not exist.
  const class View* View(const ViewId view_id) const;
  class View* MutableView(const ViewId view_id);

  // Return all ViewIds in the model.
  std::vector<ViewId> ViewIds() const;

  // Returns to ViewId of the view name, or kInvalidViewId if the view does not
  // exist.
  ViewId ViewIdFromName(const std::string& view_name) const;

 private:
  TrackId next_track_id_;
  ViewId next_view_id_;

  std::unordered_map<std::string, ViewId> view_name_to_id_;
  std::unordered_map<ViewId, class View> views_;
  std::unordered_map<TrackId, class Track> tracks_;
};

}  // namespace theia

#endif  // THEIA_VISION_SFM_MODEL_H_
