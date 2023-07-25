#include "motion.tlh"

template motreg::se3<double> motreg::details::errorFwdMotion<double>(
    const double dt, const motreg::Motion<double> &vw_0,
    const motreg::SE3<double> &T_0, const motreg::SE3<double> &T_1);
template motreg::se3<double> motreg::details::errorBwdMotion<double>(
    const double dt, const motreg::Motion<double> &vw_1,
    const motreg::SE3<double> &T_1, const motreg::SE3<double> &T_0);
template motreg::se3<double>
motreg::details::errorLabelDiff<double>(const double ctr2base,
                                        const motreg::SE3<double> &T_base,
                                        const motreg::SE3<double> &Z_ctr);
template std::tuple<motreg::details::JErr2Motion<double>,
                    motreg::details::JErr2SE3<double>,
                    motreg::details::JErr2SE3<double>>
motreg::details::errorFwdMotionJacobian<double>(
    const motreg::se3<double> &fwdErr, const double dt,
    const motreg::Motion<double> &vw_0, const motreg::SE3<double> &T_0,
    const motreg::SE3<double> &T_1);
template std::tuple<motreg::details::JErr2Motion<double>,
                    motreg::details::JErr2SE3<double>,
                    motreg::details::JErr2SE3<double>>
motreg::details::errorBwdMotionJacobian<double>(
    const motreg::se3<double> &bwdErr, const double dt,
    const motreg::Motion<double> &vw_1, const motreg::SE3<double> &T_1,
    const motreg::SE3<double> &T_0);
template std::tuple<motreg::details::JErr2Scalar<double>,
                    motreg::details::JErr2SE3<double>>
motreg::details::errorLabelDiffJacobian<double>(
    const motreg::se3<double> &labelErr, const double ctr2base,
    const motreg::SE3<double> &T_base, const motreg::SE3<double> &Z_ctr);

template class motreg::part::vertex::Ctr2BaseVertex<double>;
template class motreg::part::vertex::SE3Vertex<double>;
template class motreg::part::vertex::MotionVertex<double>;
template class motreg::part::edge::EdgeFwdMotion<double>;
template class motreg::part::edge::EdgeBwdMotion<double>;
template class motreg::part::edge::EdgeLabelDiff<double>;
template class motreg::part::edge::EdgeConstantMotion<double>;

template class motreg::model::MotionModel<double>;
