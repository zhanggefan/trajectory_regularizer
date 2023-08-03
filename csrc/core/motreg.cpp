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

motreg::api::IMotionModel *
getDllMotionModel(const std::vector<motreg::api::ObjBBox> &input,
                  const motreg::api::MotionModelParams &params) {
  return new motreg::model::MotionModel<double>(input, params);
}
void releaseDllMotionModel(motreg::api::IMotionModel *motionModel) {
  delete motionModel;
}