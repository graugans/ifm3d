/*
 * Copyright (C) 2017 Love Park Robotics, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distribted on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ifm3d/image/image_buffer.h>
#include <cstdint>
#include <vector>
#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <ifm3d/fg/byte_buffer.h>
#include <image_buffer_impl.hpp>

//--------------------------------
// ImageBuffer class
//--------------------------------

ifm3d::ImageBuffer::ImageBuffer()
  : ifm3d::ByteBuffer(),
    pImpl(new ifm3d::ImageBuffer::Impl()),
    extrinsics_({ 0., 0., 0., 0., 0., 0. }),
    exposure_times_({ 0,0,0 })
{ }

ifm3d::ImageBuffer::~ImageBuffer() = default;

ifm3d::ImageBuffer::ImageBuffer(const ifm3d::ImageBuffer& src_buff)
  : ifm3d::ByteBuffer()
{
  this->SetBytes(const_cast<std::vector<std::uint8_t>&>(src_buff.bytes_),
                 true);
}

ifm3d::ImageBuffer&
ifm3d::ImageBuffer::operator= (const ifm3d::ImageBuffer& src_buff)
{
  if (this == &src_buff)
    {
      return *this;
    }

  this->SetBytes(const_cast<std::vector<std::uint8_t>&>(src_buff.bytes_),
                 true);

  return *this;
}

cv::Mat
ifm3d::ImageBuffer::DistanceImage()
{
  this->Organize();
  return this->pImpl->DistanceImage();
}

cv::Mat
ifm3d::ImageBuffer::UnitVectors()
{
  this->Organize();
  return this->pImpl->UnitVectors();
}

cv::Mat
ifm3d::ImageBuffer::GrayImage()
{
  this->Organize();
  return this->pImpl->GrayImage();
}

cv::Mat
ifm3d::ImageBuffer::AmplitudeImage()
{
  this->Organize();
  return this->pImpl->AmplitudeImage();
}

cv::Mat
ifm3d::ImageBuffer::RawAmplitudeImage()
{
  this->Organize();
  return this->pImpl->RawAmplitudeImage();
}

cv::Mat
ifm3d::ImageBuffer::ConfidenceImage()
{
  this->Organize();
  return this->pImpl->ConfidenceImage();
}

cv::Mat
ifm3d::ImageBuffer::XYZImage()
{
  this->Organize();
  return this->pImpl->XYZImage();
}

pcl::PointCloud<ifm3d::PointT>::Ptr
ifm3d::ImageBuffer::Cloud()
{
  this->Organize();
  return this->pImpl->Cloud();
}

std::vector<float>
ifm3d::ImageBuffer::Extrinsics()
{
  this->Organize();
  return this->pImpl->Extrinsics();
}

std::vector<std::uint32_t>
ifm3d::ImageBuffer::ExposureTimes()
{
  this->Organize();
  return this->pImpl->ExposureTimes();
}

float
ifm3d::ImageBuffer::IlluTemp()
{
  this->Organize();
  return this->pImpl->IlluTemp();
}

void
ifm3d::ImageBuffer::Organize()
{
  if (! this->Dirty())
  {
    return;
  }

  std::size_t INVALID_IDX = std::numeric_limits<std::size_t>::max();
  std::size_t cidx = INVALID_IDX;
  std::size_t extidx = INVALID_IDX;

  cidx =
    ifm3d::get_chunk_index(this->bytes_, ifm3d::image_chunk::CONFIDENCE);
  extidx =
    ifm3d::get_chunk_index(this->bytes_,
      ifm3d::image_chunk::EXTRINSIC_CALIBRATION);

  if (cidx == INVALID_IDX)
  {
    throw ifm3d::error_t(IFM3D_IMG_CHUNK_NOT_FOUND);
  }

  bool EXTRINSICS_OK = extidx != INVALID_IDX;

  std::size_t cincr = 1; // uint8_t
  std::size_t extincr = extidx != INVALID_IDX ? 4 : 0; // float32

  std::uint32_t pixel_data_offset =
    ifm3d::mkval<std::uint32_t>(this->bytes_.data() + cidx + 8);
  extidx += EXTRINSICS_OK ? pixel_data_offset : 0;


  // Parse out the extrinsics
  if (EXTRINSICS_OK)
  {
    for (std::size_t i = 0; i < 6; ++i, extidx += extincr)
    {
      this->extrinsics_[i] =
        ifm3d::mkval<float>(this->bytes_.data() + extidx);
    }
  }
  else
  {
    LOG(WARNING) << "Extrinsics are invalid!";
  }

  // OK, now we want to see if the temp illu and exposure times are present,
  // if they are, we want to parse them out and store them in the image buffer.
  // Since the extrinsics are invariant and should *always* be present, we use
  // the current index of the extrinsics.
  if (EXTRINSICS_OK)
  {
    std::size_t extime_idx = extidx;
    int bytes_left = this->bytes_.size() - extime_idx;

    // Read extime (6 bytes string + 3x 4 bytes uint32_t)
    if (bytes_left >= 18
      && std::equal(this->bytes_.begin() + extidx,
        this->bytes_.begin() + extidx + 6,
        std::begin("extime")))
    {
      extime_idx += 6;
      bytes_left -= 6;

      // 3 exposure times
      for (std::size_t i = 0; i < 3; ++i)
      {
        if ((bytes_left - 6) <= 0)
        {
          break;
        }

        std::uint32_t extime =
          ifm3d::mkval<std::uint32_t>(
            this->bytes_.data() + extime_idx);

        this->exposure_times_.at(i) = extime;

        extime_idx += 4;
        bytes_left -= 4;
      }
    }
    else
    {
      std::fill(this->exposure_times_.begin(),
        this->exposure_times_.end(), 0);
    }

    // Read temp_illu (9 bytes string + 4 bytes float)
    if (bytes_left >= 13
      && std::equal(this->bytes_.begin() + extidx,
        this->bytes_.begin() + extidx + 8,
        std::begin("temp_illu")))
    {
      extime_idx += 9;
      bytes_left -= 9;

      this->pImpl->illu_temp_ =
        ifm3d::mkval<float>(this->bytes_.data() + extime_idx);

      extime_idx += 4;
      bytes_left -= 4;

      DLOG(INFO) << "IlluTemp= " << this->pImpl->illu_temp_;
    }
    else
    {
      this->pImpl->illu_temp_ = 0;
    }
  }
  else
  {
    LOG(WARNING) << "Checking for illu temp and exposure times skipped (cant trust extidx)";
  }


  this->pImpl->Organize(this->bytes_);
  this->_SetDirty(false);
}