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

#ifndef THEIA_VISION_MATCHING_CASCADE_HASHER_H_
#define THEIA_VISION_MATCHING_CASCADE_HASHER_H_

#include <Eigen/Core>
#include <stdint.h>
#include <bitset>
#include <vector>

namespace theia {

struct FeatureMatch;

typedef Eigen::Matrix<float, 128, 1> Vector128f;
typedef std::vector<int> Bucket;

struct HashedSiftDescriptor {
  Vector128f sift_desc;
  std::bitset<128> hash_code;
  // Each bucket_ids[x] = y means the descriptor belongs to bucket y in bucket
  // group x.
  std::vector<uint16_t> bucket_ids;
};

struct HashedImage {
  std::vector<HashedSiftDescriptor> hashed_desc;

  // buckets[bucket_group][bucket_id] = bucket (container of sift ids).
  std::vector<std::vector<Bucket> > buckets;
};

// This hasher will hash SIFT descriptors with a two-step hashing system. The
// first generates a hash code and the second determines which buckets the
// descriptors belong to. Descriptors in the same bucket are likely to be good
// matches.
//
// Implementation is based on the paper "Fast and Accurate Image Matching with
// Cascade Hashing for 3D Reconstruction" by Cheng et al (CVPR 2014). When using
// this class we ask that you please cite this paper.
class CascadeHasher {
 public:
  CascadeHasher() {}
  bool Initialize();

  // Creates the hash codes for the sift descriptors and returns the hashed
  // information.
  void CreateHashedSiftDescriptors(
      const std::vector<Eigen::VectorXf>& sift_desc,
      HashedImage* hashed_image) const;

  // Matches images with a fast matching scheme based on the hash codes
  // previously generated.
  void MatchImages(const HashedImage& desc1,
                   const HashedImage& desc2,
                   const double lowes_ratio,
                   std::vector<FeatureMatch>* matches) const;

 private:
  // Creates the hash code for each descriptor and determines which buckets each
  // descriptor belongs to.
  void CreateHashedDescriptors(HashedImage* hashed_image) const;

  // Builds the buckets for an image based on the bucket ids and groups of the
  // sift descriptors.
  void BuildBuckets(HashedImage* hashed_image) const;

  // the number of dimensions of Hash code
  static const int kDimHashData = 128;

  // the number of bucket bits
  static const int kNumBucketBits = 8;

  // the number of bucket groups
  static const int kNumBucketGroups = 6;

  // the number of buckets in each group
  static constexpr int kNumBucketsPerGroup = 1 << kNumBucketBits;

  // Projection matrix of the primary hashing function
  Vector128f primary_hash_projection[kDimHashData];
  // Projection matrix of the secondary hashing function
  Vector128f secondary_hash_projection[kNumBucketGroups][kNumBucketBits];
};

}  // namespace theia

#endif  // THEIA_VISION_MATCHING_CASCADE_HASHER_H_
