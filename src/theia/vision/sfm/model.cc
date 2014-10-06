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

#include "theia/vision/sfm/model.h"

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "theia/util/map_util.h"
#include "theia/vision/sfm/feature.h"
#include "theia/vision/sfm/track.h"
#include "theia/vision/sfm/types.h"
#include "theia/vision/sfm/view.h"

namespace theia {

namespace {

// Return whether the track contains the same view twice.
bool DuplicateViewsExistInTrack(
    const std::vector<std::pair<std::string, Feature> >& track) {
  std::vector<std::string> view_names;
  view_names.reserve(track.size());
  for (const auto& feature : track) {
    view_names.push_back(feature.first);
  }
  std::sort(view_names.begin(), view_names.end());
  return (std::unique(view_names.begin(), view_names.end()) !=
          view_names.end());
}

}  // namespace

Model::Model() : next_track_id_(0), next_view_id_(0) {}

Model::~Model() {}

TrackId Model::AddTrack(
    const std::vector<std::pair<std::string, Feature> >& track) {
  if (track.size() < 2) {
    LOG(WARNING) << "Tracks must have at least 2 observations (" << track.size()
                 << " were given). Cannot add track to the model";
    return kInvalidTrackId;
  }

  if (DuplicateViewsExistInTrack(track)) {
    LOG(WARNING)
        << "Cannot add a track that contains the same view twice to the model.";
    return kInvalidTrackId;
  }

  const TrackId new_track_id = next_track_id_;
  CHECK(!ContainsKey(tracks_, new_track_id))
      << "The model already contains a track with id: " << new_track_id;

  class Track new_track(new_track_id);

  std::vector<ViewId> views_to_add;
  for (const auto& feature : track) {
    // Get the view id or add a new view if it does not already exist.
    ViewId view_id = ViewIdFromName(feature.first);
    if (view_id == kInvalidViewId) {
      view_id = AddView(feature.first);
      // Make sure the view was added properly.
      if (view_id == kInvalidViewId) {
        LOG(WARNING) << "Could not add the track to the model because we could "
                        "not add view " << feature.first << " to the model.";
        return kInvalidTrackId;
      }
    }

    // Add view to track.
    new_track.AddView(view_id);

    // Add track to view.
    class View* view = MutableView(view_id);
    view->AddFeature(Feature(new_track_id, feature.second.x, feature.second.y));
  }

  tracks_.emplace(new_track_id, new_track);
  ++next_track_id_;
  return new_track_id;
}

bool Model::RemoveTrack(const TrackId track_id) {
  class Track* track = FindOrNull(tracks_, track_id);
  if (track == nullptr) {
    LOG(WARNING) << "Cannot remove a track that does not exist";
    return false;
  }

  // Remove track from views.
  for (const ViewId view_id : track->ViewIds()) {
    class View* view = FindOrNull(views_, view_id);
    if (view == nullptr) {
      LOG(WARNING) << "Could not remove a track from the view because the view "
                    "does not exist";
      return false;
    }

    if (!view->RemoveFeature(track_id)) {
      LOG(WARNING) << "Could not remove the track from the view because the "
                    "track is not observed by the view.";
      return false;
    }
  }

  // Delete from the model.
  tracks_.erase(track_id);
  return true;
}

int Model::NumTracks() const {
  return tracks_.size();
}

const class Track* Model::Track(const TrackId track_id) const {
  return FindOrNull(tracks_, track_id);
}

class Track* Model::MutableTrack(const TrackId track_id) {
  return FindOrNull(tracks_, track_id);
}

std::vector<TrackId> Model::TrackIds() const {
  std::vector<TrackId> track_ids;
  track_ids.reserve(tracks_.size());
  for (const auto& track : tracks_) {
    track_ids.push_back(track.first);
  }
  return track_ids;
}

ViewId Model::AddView(const std::string& view_name) {
  if (ContainsKey(view_name_to_id_, view_name)) {
    LOG(WARNING) << "Could not add view with the name " << view_name
               << " because that name already exists in the model.";
    return kInvalidViewId;
  }

  if (view_name.empty()) {
    LOG(WARNING) << "View name was empty. Could not add view to model.";
    return kInvalidViewId;
  }

  const ViewId new_view_id = next_view_id_;
  class View new_view(view_name, new_view_id);
  views_.emplace(new_view_id, new_view);
  view_name_to_id_.emplace(view_name, new_view_id);

  ++next_view_id_;
  return new_view_id;
}

bool Model::RemoveView(const ViewId view_id) {
  class View* view = FindOrNull(views_, view_id);
  if (view == nullptr) {
    LOG(WARNING) << "Could not remove the view from the model because the view "
                    "does not exist.";
    return false;
  }

  for (const TrackId track_id : view->TrackIds()) {
    class Track* track = MutableTrack(track_id);
    if (track == nullptr) {
      LOG(WARNING) << "Could not remove the view from the track because the "
                    "track does not exist";
      return false;
    }

    if (!track->RemoveView(view_id)) {
      LOG(WARNING) << "Could not remove to view from the track";
      return false;
    }
  }

  views_.erase(view_id);
  return true;
}

int Model::NumViews() const {
  return views_.size();
}

const class View* Model::View(const ViewId view_id) const {
  return FindOrNull(views_, view_id);
}

class View* Model::MutableView(const ViewId view_id) {
  return FindOrNull(views_, view_id);
}

std::vector<ViewId> Model::ViewIds() const {
  std::vector<ViewId> view_ids;
  view_ids.reserve(views_.size());
  for (const auto& view : views_) {
    view_ids.push_back(view.first);
  }
  return view_ids;
}

ViewId Model::ViewIdFromName(const std::string& view_name) const {
  return FindWithDefault(view_name_to_id_, view_name, kInvalidViewId);
}

}  // namespace theia
