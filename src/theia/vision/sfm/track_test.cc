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
#include <vector>
#include "gtest/gtest.h"

#include "theia/vision/sfm/types.h"
#include "theia/vision/sfm/track.h"

namespace theia {

TEST(Track, Constructors) {
  // Default constructor.
  Track track;
  EXPECT_EQ(track.Id(), kInvalidTrackId);
  EXPECT_TRUE(!track.IsEstimated());
  EXPECT_EQ(track.Point(), Eigen::Vector4d::Zero());

  // With the id.
  Track track2(1);
  EXPECT_EQ(track2.Id(), 1);

  // Copy constructor.
  Track track3(track2);
  EXPECT_EQ(track3.Id(), 1);
  // Change track 2 and make sure it does not change track 3.
  track2.SetEstimated(true);
  EXPECT_TRUE(track2.IsEstimated());
  EXPECT_TRUE(!track3.IsEstimated());

  // Assign operator.
  Track track4 = track3;
  EXPECT_EQ(track3.Id(), 1);
  // Change track 3 and make sure it does not change track 4.
  track3.SetEstimated(true);
  EXPECT_TRUE(track3.IsEstimated());
  EXPECT_TRUE(!track4.IsEstimated());
}

TEST(Track, GettersAndSetters) {
  Track track(0);

  EXPECT_EQ(track.Id(), 0);

  track.SetEstimated(true);
  EXPECT_TRUE(track.IsEstimated());

  *track.MutablePoint() = Eigen::Vector4d::Ones();
  EXPECT_EQ(track.Point(), Eigen::Vector4d::Ones());
}

TEST(Track, AddRemoveViews) {
  Track track;

  EXPECT_EQ(track.NumViews(), 0);

  std::vector<ViewId> views = { 0, 1, 2 };

  for (const ViewId& view_id : views) {
    track.AddView(view_id);
    EXPECT_TRUE(track.IsViewVisible(view_id));
  }

  EXPECT_EQ(track.NumViews(), 3);

  for (const ViewId& view : views) {
    EXPECT_TRUE(track.RemoveView(view));
  }

  EXPECT_EQ(track.NumViews(), 0);
  EXPECT_TRUE(!track.RemoveView(kInvalidTrackId));
}

}  // namespace theia
