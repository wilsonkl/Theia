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

#include "gtest/gtest.h"

#include "theia/vision/sfm/model.h"

namespace theia {

const std::vector<std::string> view_names = {"1", "2", "3"};
const std::vector<Feature> features = { Feature(kInvalidTrackId, 1, 1),
                                        Feature(kInvalidTrackId, 2, 2),
                                        Feature(kInvalidTrackId, 3, 3) };

TEST(Model, AddTrack) {
  Model model;

  // Should fail with less than two views.
  const std::vector<std::pair<std::string, Feature> > small_track = {
    { view_names[0], features[0] }
  };
  EXPECT_EQ(model.AddTrack(small_track), kInvalidTrackId);
  EXPECT_EQ(model.NumTracks(), 0);
  EXPECT_EQ(model.NumViews(), 0);

  const std::vector<std::pair<std::string, Feature> > track = {
    { view_names[0], features[0] }, { view_names[1], features[1] }
  };

  const TrackId track_id = model.AddTrack(track);
  EXPECT_NE(track_id, kInvalidTrackId);
  EXPECT_TRUE(model.Track(track_id) != nullptr);
  EXPECT_EQ(model.NumTracks(), 1);
  EXPECT_EQ(model.NumViews(), 2);
  EXPECT_NE(model.ViewIdFromName("1"), kInvalidViewId);
  EXPECT_NE(model.ViewIdFromName("2"), kInvalidViewId);
}

TEST(Model, RemoveTrack) {
  Model model;

  // Should return false when trying to remove a track not in the model.
  EXPECT_FALSE(model.RemoveTrack(kInvalidTrackId));

  const std::vector<std::pair<std::string, Feature> > track = {
    { view_names[0], features[0] }, { view_names[1], features[1] }
  };

  // Should be able to successfully remove the track.
  const TrackId track_id = model.AddTrack(track);
  EXPECT_TRUE(model.RemoveTrack(track_id));
}

TEST(Model, GetTrack) {
  Model model;
  const std::vector<std::pair<std::string, Feature> > track = {
    { view_names[0], features[0] }, { view_names[1], features[1] }
  };
  const TrackId track_id = model.AddTrack(track);
  EXPECT_NE(track_id, kInvalidTrackId);

  const Track* const_track = model.Track(track_id);
  EXPECT_NE(const_track, nullptr);

  Track* mutable_track = model.MutableTrack(track_id);
  EXPECT_NE(mutable_track, nullptr);
}

TEST(Model, AddView) {
  Model model;
  const ViewId view_id = model.AddView(view_names[0]);
  EXPECT_NE(view_id, kInvalidViewId);
  EXPECT_EQ(model.NumViews(), 1);
  EXPECT_EQ(model.NumTracks(), 0);

  // Try to add the view again.
  EXPECT_EQ(model.AddView(view_names[0]), kInvalidViewId);
  EXPECT_EQ(model.NumViews(), 1);
  EXPECT_EQ(model.NumTracks(), 0);
}

TEST(Model, RemoveView) {
  Model model;
  const ViewId view_id1 = model.AddView(view_names[0]);
  const ViewId view_id2 = model.AddView(view_names[1]);
  EXPECT_EQ(model.NumViews(), 2);

  EXPECT_TRUE(model.RemoveView(view_id1));
  EXPECT_EQ(model.NumViews(), 1);
  EXPECT_TRUE(model.RemoveView(view_id2));
  EXPECT_EQ(model.NumViews(), 0);
  EXPECT_FALSE(model.RemoveView(kInvalidViewId));
}

TEST(Model, GetView) {
  Model model;
  const ViewId view_id = model.AddView(view_names[0]);
  EXPECT_NE(view_id, kInvalidViewId);

  const View* const_view = model.View(view_id);
  EXPECT_NE(const_view, nullptr);

  View* mutable_view = model.MutableView(view_id);
  EXPECT_NE(mutable_view, nullptr);
}

TEST(Model, ViewIdFromName) {
  Model model;
  const ViewId view_id = model.AddView(view_names[0]);

  const ViewId retrieved_view_id = model.ViewIdFromName(view_names[0]);
  EXPECT_EQ(view_id, retrieved_view_id);

  EXPECT_EQ(model.ViewIdFromName(view_names[1]), kInvalidViewId);
}

}  // namespace theia
