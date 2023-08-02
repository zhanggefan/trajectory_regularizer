#include "motreg.h"
#include "motreg_implement.tlh"

template motreg::se3<double> API_EXPORT motreg::details::errorFwdMotion<double>(
    const double dt, const motreg::Motion<double> &vw_0,
    const motreg::SE3<double> &T_0, const motreg::SE3<double> &T_1);
template motreg::se3<double> API_EXPORT motreg::details::errorBwdMotion<double>(
    const double dt, const motreg::Motion<double> &vw_1,
    const motreg::SE3<double> &T_1, const motreg::SE3<double> &T_0);
template motreg::se3<double> API_EXPORT motreg::details::errorLabelDiff<double>(
    const double ctr2base, const motreg::SE3<double> &T_base,
    const motreg::SE3<double> &Z_ctr);
template std::tuple<motreg::details::JErr2Motion<double>,
                    motreg::details::JErr2SE3<double>,
                    motreg::details::JErr2SE3<double>>
    API_EXPORT motreg::details::errorFwdMotionJacobian<double>(
        const motreg::se3<double> &fwdErr, const double dt,
        const motreg::Motion<double> &vw_0, const motreg::SE3<double> &T_0,
        const motreg::SE3<double> &T_1);
template std::tuple<motreg::details::JErr2Motion<double>,
                    motreg::details::JErr2SE3<double>,
                    motreg::details::JErr2SE3<double>>
    API_EXPORT motreg::details::errorBwdMotionJacobian<double>(
        const motreg::se3<double> &bwdErr, const double dt,
        const motreg::Motion<double> &vw_1, const motreg::SE3<double> &T_1,
        const motreg::SE3<double> &T_0);
template std::tuple<motreg::details::JErr2Scalar<double>,
                    motreg::details::JErr2SE3<double>>
    API_EXPORT motreg::details::errorLabelDiffJacobian<double>(
        const motreg::se3<double> &labelErr, const double ctr2base,
        const motreg::SE3<double> &T_base, const motreg::SE3<double> &Z_ctr);

template class API_EXPORT motreg::part::vertex::Ctr2BaseVertex<double>;
template class API_EXPORT motreg::part::vertex::SE3Vertex<double>;
template class API_EXPORT motreg::part::vertex::MotionVertex<double>;
template class API_EXPORT motreg::part::edge::EdgeFwdMotion<double>;
template class API_EXPORT motreg::part::edge::EdgeBwdMotion<double>;
template class API_EXPORT motreg::part::edge::EdgeLabelDiff<double>;
template class API_EXPORT motreg::part::edge::EdgeConstantMotion<double>;

template class API_EXPORT motreg::model::MotionModel<double>;
#ifdef _WIN32
template class API_EXPORT motreg::model::MotionModel<double>::MotionModelParams;
template API_EXPORT
motreg::model::MotionModel<double>::MotionModelParams::MotionModelParams();
template class API_EXPORT motreg::model::MotionModel<double>::ObjBBox;
#endif

motreg::api::ObjBBox
cast(const motreg::model::MotionModel<double>::ObjBBox &objBBox) {
  const auto &poseR = objBBox.pose.so3().unit_quaternion();
  const auto &poseT = objBBox.pose.translation();
  const auto &m = objBBox.motion;
  const auto &emf = objBBox.errMotionFwd;
  const auto &emb = objBBox.errMotionBwd;
  const auto &el = objBBox.errLabel;
  return {objBBox.sequence,
          objBBox.timestamp,
          {poseT.x(), poseT.y(), poseT.z()},
          {poseR.x(), poseR.y(), poseR.z(), poseR.w()},
          objBBox.poseFixed,
          {m[0], m[1]},
          {emf[0], emf[1], emf[2], emf[3], emf[4], emf[5]},
          {emb[0], emb[1], emb[2], emb[3], emb[4], emb[5]},
          {el[0], el[1], el[2], el[3], el[4], el[5]}};
}

motreg::model::MotionModel<double>::ObjBBox
cast(const motreg::api::ObjBBox &objBBox) {
  const auto &poseR = objBBox.boxRotationXYZW;
  const auto &poseT = objBBox.boxBottomCtrXYZ;
  const auto &m = objBBox.motionVW;
  const auto &emf = objBBox.errMotionFwd;
  const auto &emb = objBBox.errMotionBwd;
  const auto &el = objBBox.errLabel;
  return {objBBox.sequence,
          objBBox.timestamp,
          {Eigen::Quaterniond(poseR[3], poseR[0], poseR[1], poseR[2]),
           Eigen::Vector3d(poseT[0], poseT[1], poseT[2])},
          objBBox.boxFixed,
          {m[0], m[1]},
          Eigen::Map<const Eigen::Matrix<double, 6, 1>>(emf.data()),
          Eigen::Map<const Eigen::Matrix<double, 6, 1>>(emb.data()),
          Eigen::Map<const Eigen::Matrix<double, 6, 1>>(el.data())};
}

motreg::model::MotionModel<double>::MotionModelParams
cast(const motreg::api::MotionModelParams &params) {
  auto weightMotionDiag =
      Eigen::Map<const Eigen::Matrix<double, 6, 1>>(params.weightMotion.data());
  Eigen::Vector2d weightMotionConsistencyDiag(
      params.weightMotionConsistency[0], params.weightMotionConsistency[1]);
  auto weightObjPose2LabelDiag = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(
      params.weightObjPose2Label.data());
  return {weightMotionDiag.asDiagonal(),
          weightMotionConsistencyDiag.asDiagonal(),
          weightObjPose2LabelDiag.asDiagonal(),
          params.objCtr2Origin,
          params.fixObjCtr2Origin,
          params.steps,
          params.verbose};
}

motreg::api::MotionModel::MotionModel(
    const std::vector<motreg::api::ObjBBox> &input,
    const motreg::api::MotionModelParams &params) {
  std::vector<motreg::model::MotionModel<double>::ObjBBox> inputCast;
  for (const auto &item : input) {
    inputCast.push_back(cast(item));
  }
  pimpl = reinterpret_cast<void *>(
      new motreg::model::MotionModel<double>(inputCast, cast(params)));
}

motreg::api::MotionModel::~MotionModel() {
  delete reinterpret_cast<motreg::model::MotionModel<double> *>(pimpl);
}

motreg::api::ObjBBox motreg::api::MotionModel::output(int sequence) const {
  return cast(
      reinterpret_cast<motreg::model::MotionModel<double> *>(pimpl)->output(
          sequence));
}

std::vector<motreg::api::ObjBBox> motreg::api::MotionModel::output() const {
  auto output =
      reinterpret_cast<motreg::model::MotionModel<double> *>(pimpl)->output();
  std::vector<motreg::api::ObjBBox> outputCast;
  for (const auto &item : output) {
    outputCast.push_back(cast(item));
  }
  return outputCast;
}

std::vector<motreg::api::ObjBBox>
motreg::api::MotionModel::output(const std::vector<int> &sequences) const {
  auto output =
      reinterpret_cast<motreg::model::MotionModel<double> *>(pimpl)->output(
          sequences);
  std::vector<motreg::api::ObjBBox> outputCast;
  for (const auto &item : output) {
    outputCast.push_back(cast(item));
  }
  return outputCast;
}