#pragma once
#include <array>
#include <vector>

#ifdef _WIN32
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT __attribute__((visibility("default")))
#endif

namespace motreg {
namespace api {

struct API_EXPORT ObjBBox {
  /// sequence index 0 ~ 199
  int sequence;

  /// timestamp in unit ns
  double timestamp;

  /// box bottom center x y z
  std::array<double, 3> boxBottomCtrXYZ;

  /// box rotation x y z w
  std::array<double, 4> boxRotationXYZW;

  /// whether to fix this box during optimization or not
  /// set to false by default
  bool boxFixed = false;

  /// the object line speed v (along the object's x axis)
  /// and the rotation speed w (around the object's z axis)
  std::array<double, 2> motionVW = {0., 0.};

  /// the forward motion error of the object (expressed in lee algebra)
  std::array<double, 6> errMotionFwd = {0., 0., 0., 0., 0., 0.};

  /// the backward motion error of the object (expressed in lee algebra)
  std::array<double, 6> errMotionBwd = {0., 0., 0., 0., 0., 0.};

  /// the discrepancy between the regularized box and
  /// the annotated box of the object (expressed in lee algebra)
  std::array<double, 6> errLabel = {0., 0., 0., 0., 0., 0.};
};

struct API_EXPORT MotionModelParams {
  /// the optimization loss weight of forward/backward motion that push each
  /// pose in the sequence close to the integration of the object's speed
  /// this is the 6 diagonal elements of the 6*6 weight matrix
  std::array<double, 6> weightMotion = {1., 1., 1., 1., 1., 1.};

  /// the optimization loss weight of motion consistency that push the
  /// speed v and w stay almost constant in a short period of time
  /// this is the 2 diagonal elements of the 2*2 weight matrix
  std::array<double, 2> weightMotionConsistency = {1., 1.};

  /// the optimization loss weight of the label that push the regularized
  /// box close to the annotated box
  /// this is the 6 diagonal elements of the 6*6 weight matrix
  std::array<double, 6> weightObjPose2Label = {0.2, 0.2, 0.2, 0.5, 0.5, 0.2};

  /// box bottom center distance to the object's base-link origin
  /// set to 0 by default
  double objCtr2Origin = 0;

  /// whether to fix the value of objCtr2Origin during optimization
  /// or let the motion model guess it through optimization
  /// set to true by default
  bool fixObjCtr2Origin = true;

  /// optimization steps
  /// set to 100 by default
  int steps = 100;

  /// whether to enable logging to cout during optimization
  bool verbose = false;
};

class API_EXPORT MotionModel {
  void *pimpl;

public:
  MotionModel(const std::vector<ObjBBox> &input,
              const MotionModelParams &params);
  MotionModel() = delete;
  MotionModel(const MotionModel &) = delete;
  MotionModel &operator=(const MotionModel &) = delete;
  MotionModel(MotionModel &&) = delete;
  MotionModel &operator=(MotionModel &&) = delete;

  ~MotionModel();

  /// query the regularized box by its sequence idx
  /// note the queried sequence idx shall be between the min & max of
  /// the input sequence idx
  ObjBBox output(int sequence) const;
  /// batch query the regularized boxes correspond to the input boxes
  /// sequence always returned in the ascending order of sequence idx
  std::vector<ObjBBox> output() const;
  /// batch query the regularized boxes for a batch of sequence indices
  /// returned in the order same to the queried batch sequence indices
  std::vector<ObjBBox> output(const std::vector<int> &sequences) const;
};
} // namespace api
} // namespace motreg