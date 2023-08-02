#pragma once
#include <Eigen/Core>
#include <g2o/core/base_fixed_sized_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/sparse_optimizer.h>
#include <sophus/se3.hpp>

namespace motreg {

using Sophus::SE3;
template <class T> using se3 = typename SE3<T>::Tangent;
template <class T> using Motion = Sophus::Vector2<T>;

namespace details {

template <class T> using JErr2Motion = Sophus::Matrix<T, 6, 2>;
template <class T> using JErr2Scalar = Sophus::Vector6<T>;
template <class T> using JErr2SE3 = Sophus::Matrix<T, 6, 6>;

template <class T>
se3<T> errorFwdMotion(const T dt, const Motion<T> &vw_0, const SE3<T> &T_0,
                      const SE3<T> &T_1);
template <class T>
se3<T> errorBwdMotion(const T dt, const Motion<T> &vw_1, const SE3<T> &T_1,
                      const SE3<T> &T_0);
template <class T>
se3<T> errorLabelDiff(const T ctr2base, const SE3<T> &T_base,
                      const SE3<T> &Z_ctr);
template <class T>
std::tuple<JErr2Motion<T>, JErr2SE3<T>, JErr2SE3<T>>
errorFwdMotionJacobian(const se3<T> &fwdErr, const T dt, const Motion<T> &vw_0,
                       const SE3<T> &T_0, const SE3<T> &T_1);

template <class T>
std::tuple<JErr2Motion<T>, JErr2SE3<T>, JErr2SE3<T>>
errorBwdMotionJacobian(const se3<T> &bwdErr, const T dt, const Motion<T> &vw_1,
                       const SE3<T> &T_1, const SE3<T> &T_0);

template <class T>
std::tuple<JErr2Scalar<T>, JErr2SE3<T>>
errorLabelDiffJacobian(const se3<T> &labelErr, const T ctr2base,
                       const SE3<T> &T_base, const SE3<T> &Z_ctr);

} // namespace details

namespace part {
namespace vertex {
template <class T> class Ctr2BaseVertex : public g2o::BaseVertex<1, T> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool read(std::istream &is) override;
  bool write(std::ostream &os) const override;
  void setToOriginImpl() override;
  void oplusImpl(const double *update) override;
};

template <class T> class SE3Vertex : public g2o::BaseVertex<6, SE3<T>> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool read(std::istream &is) override;
  bool write(std::ostream &os) const override;
  void setToOriginImpl() override;
  void oplusImpl(const double *update) override;
};

template <class T> class MotionVertex : public g2o::BaseVertex<2, Motion<T>> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool read(std::istream &is) override;
  bool write(std::ostream &os) const override;
  void setToOriginImpl() override;
  void oplusImpl(const double *update) override;
};
} // namespace vertex

namespace edge {
template <class T>
class EdgeFwdMotion
    : public g2o::BaseFixedSizedEdge<6, T, vertex::MotionVertex<T>,
                                     vertex::SE3Vertex<T>,
                                     vertex::SE3Vertex<T>> {
public:
  using Base =
      g2o::BaseFixedSizedEdge<6, T, vertex::MotionVertex<T>,
                              vertex::SE3Vertex<T>, vertex::SE3Vertex<T>>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool read(std::istream &is) override;
  bool write(std::ostream &os) const override;
  void computeError() override;
  void linearizeOplus() override;
  auto jacobian() { return this->_jacobianOplus; };
  explicit EdgeFwdMotion(T dt);
  T dt_;
};

template <class T>
class EdgeBwdMotion
    : public g2o::BaseFixedSizedEdge<6, T, vertex::MotionVertex<T>,
                                     vertex::SE3Vertex<T>,
                                     vertex::SE3Vertex<T>> {
public:
  using Base =
      g2o::BaseFixedSizedEdge<6, T, vertex::MotionVertex<T>,
                              vertex::SE3Vertex<T>, vertex::SE3Vertex<T>>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool read(std::istream &is) override;
  bool write(std::ostream &os) const override;
  void computeError() override;
  void linearizeOplus() override;
  auto jacobian() { return this->_jacobianOplus; };
  explicit EdgeBwdMotion(T dt);
  T dt_;
};

template <class T>
class EdgeLabelDiff
    : public g2o::BaseFixedSizedEdge<6, SE3<T>, vertex::Ctr2BaseVertex<T>,
                                     vertex::SE3Vertex<T>> {
public:
  using Base = g2o::BaseFixedSizedEdge<6, SE3<T>, vertex::Ctr2BaseVertex<T>,
                                       vertex::SE3Vertex<T>>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool read(std::istream &is) override;
  bool write(std::ostream &os) const override;
  void computeError() override;
  void linearizeOplus() override;
  auto jacobian() { return this->_jacobianOplus; };
  EdgeLabelDiff();
};

template <class T>
class EdgeConstantMotion
    : public g2o::BaseFixedSizedEdge<2, T, vertex::MotionVertex<T>,
                                     vertex::MotionVertex<T>> {
public:
  using Base = g2o::BaseFixedSizedEdge<2, T, vertex::MotionVertex<T>,
                                       vertex::MotionVertex<T>>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool read(std::istream &is) override;
  bool write(std::ostream &os) const override;
  void computeError() override;
  void linearizeOplus() override;
  auto jacobian() { return this->_jacobianOplus; };
  EdgeConstantMotion();
};
} // namespace edge
} // namespace part

namespace model {
template <class T> class MotionModel {
public:
  using VertexObjCtr2Origin = motreg::part::vertex::Ctr2BaseVertex<T>;
  using VertexObjPose = motreg::part::vertex::SE3Vertex<T>;
  using VertexObjMotion = motreg::part::vertex::MotionVertex<T>;

  using EdgeObjMotionFwd = motreg::part::edge::EdgeFwdMotion<T>;
  using EdgeObjMotionBwd = motreg::part::edge::EdgeBwdMotion<T>;
  using EdgeMotionConsistency = motreg::part::edge::EdgeConstantMotion<T>;
  using EdgeObjPose2Label = motreg::part::edge::EdgeLabelDiff<T>;

  struct ObjBBox {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int sequence;
    T timestamp;
    typename VertexObjPose::EstimateType pose;
    bool poseFixed;
    typename VertexObjMotion::EstimateType motion;
    typename EdgeObjMotionFwd::ErrorVector errMotionFwd;
    typename EdgeObjMotionBwd::ErrorVector errMotionBwd;
    typename EdgeObjPose2Label::ErrorVector errLabel;
  };

  struct MotionModelParams {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static_assert(
        std::is_same<typename EdgeObjMotionFwd::InformationType,
                     typename EdgeObjMotionBwd::InformationType>::value);
    typename EdgeObjMotionFwd::InformationType weightMotion =
        EdgeObjMotionFwd::InformationType::Identity();
    typename EdgeMotionConsistency::InformationType weightMotionConsistency =
        EdgeMotionConsistency::InformationType::Identity();
    typename EdgeObjPose2Label::InformationType weightObjPose2Label =
        EdgeObjPose2Label::InformationType::Identity();
    T objCtr2Origin = 0.;
    bool fixObjCtr2Origin = true;
    int steps = 100;
    bool verbose = false;
  };

private:
  g2o::SparseOptimizer optimizer_;
  std::unique_ptr<g2o::OptimizationAlgorithm> algorithm_;
  int vertexId_ = 0;
  int edgeId_ = 0;
  std::unique_ptr<VertexObjCtr2Origin> vertexObjCtr2Origin_;
  std::vector<std::unique_ptr<VertexObjPose>> verticesObjPose_;
  std::vector<std::unique_ptr<VertexObjMotion>> verticesObjMotion_;

  std::vector<std::unique_ptr<EdgeObjMotionFwd>> edgesObjMotionFwd_;
  std::vector<std::unique_ptr<EdgeObjMotionBwd>> edgesObjMotionBwd_;
  std::vector<std::unique_ptr<EdgeMotionConsistency>> edgesMotionConsistency_;
  std::vector<std::unique_ptr<EdgeObjPose2Label>> edgesObjPose2Label_;
  std::vector<T> timestamps_;

  std::vector<ObjBBox> input_;

public:
  explicit MotionModel(const std::vector<ObjBBox> &input,
                       const MotionModelParams &params);
  ObjBBox output(int sequence) const;
  std::vector<ObjBBox> output() const;
  std::vector<ObjBBox> output(const std::vector<int> &sequences) const;

  std::vector<T> debugVerticesTimestamps() const { return timestamps_; };
  const std::unique_ptr<VertexObjCtr2Origin> &debugVertexObjCtr2Origin() const {
    return vertexObjCtr2Origin_;
  };
  const std::vector<std::unique_ptr<VertexObjPose>> &
  debugVerticesObjPose() const {
    return verticesObjPose_;
  };
  const std::vector<std::unique_ptr<VertexObjMotion>> &
  debugVerticesObjMotion() const {
    return verticesObjMotion_;
  };
};
} // namespace model

} // namespace motreg
