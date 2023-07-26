#include "core/motion.h"
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/sparse_optimizer.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(ext, m) {
  auto utils = m.def_submodule("utils");
  pybind11::class_<Sophus::SE3<double>>(utils, "SE3")
      .def(pybind11::init<Eigen::Matrix4d>())
      .def("__repr__",
           [](const Sophus::SE3<double> &self) {
             std::stringstream ss;
             ss << "<Sophus::SE3>\n";
             Eigen::internal::print_matrix(
                 ss, self.matrix(),
                 Eigen::IOFormat{4, 0, ", ", "\n", "[", "]", "[", "]"});
             return ss.str();
           })
      .def("matrix", &Sophus::SE3<double>::matrix)
      .def("log", &Sophus::SE3<double>::log)
      .def_static("exp", &Sophus::SE3<double>::exp);

  auto details = m.def_submodule("details");
  details.def("errorFwdMotion", &motreg::details::errorFwdMotion<double>)
      .def("errorBwdMotion", &motreg::details::errorBwdMotion<double>)
      .def("errorLabelDiff", &motreg::details::errorLabelDiff<double>)
      .def("errorFwdMotionJacobian",
           &motreg::details::errorFwdMotionJacobian<double>)
      .def("errorBwdMotionJacobian",
           &motreg::details::errorBwdMotionJacobian<double>)
      .def("errorLabelDiffJacobian",
           &motreg::details::errorLabelDiffJacobian<double>);

  auto g2o = m.def_submodule("g2o");
  pybind11::class_<g2o::HyperGraph::Vertex>(g2o, "HyperGraph::Vertex");
  pybind11::class_<g2o::OptimizableGraph::Vertex, g2o::HyperGraph::Vertex>(
      g2o, "OptimizableGraph::Vertex", pybind11::multiple_inheritance());
  pybind11::class_<g2o::HyperGraph::Edge>(g2o, "HyperGraph::Edge");
  pybind11::class_<g2o::OptimizableGraph::Edge, g2o::HyperGraph::Edge>(
      g2o, "OptimizableGraph::Edge", pybind11::multiple_inheritance());

  pybind11::class_<g2o::JacobianWorkspace>(g2o, "JacobianWorkspace")
      .def(pybind11::init<>())
      .def("updateSize",
           static_cast<void (g2o::JacobianWorkspace::*)(
               const g2o::HyperGraph::Edge *, bool)>(
               &g2o::JacobianWorkspace::updateSize),
           pybind11::arg("edge"), pybind11::arg("reset") = false)
      .def("allocate", &g2o::JacobianWorkspace::allocate)
      .def("setZero", &g2o::JacobianWorkspace::setZero);

  using Ctr2BaseVertex = motreg::part::vertex::Ctr2BaseVertex<double>;
  pybind11::class_<Ctr2BaseVertex, g2o::OptimizableGraph::Vertex>(
      g2o, "Ctr2BaseVertex", pybind11::multiple_inheritance())
      .def(pybind11::init<>())
      .def("setToOrigin", &Ctr2BaseVertex::setToOriginImpl)
      .def_property("estimate", &Ctr2BaseVertex::estimate,
                    &Ctr2BaseVertex::setEstimate)
      .def_property("fixed", &Ctr2BaseVertex::fixed, &Ctr2BaseVertex::setFixed)
      .def_property("id", &Ctr2BaseVertex::id, &Ctr2BaseVertex::setId)
      .def_property_readonly("dimension", &Ctr2BaseVertex::dimension);

  using SE3Vertex = motreg::part::vertex::SE3Vertex<double>;
  pybind11::class_<SE3Vertex, g2o::OptimizableGraph::Vertex>(
      g2o, "SE3Vertex", pybind11::multiple_inheritance())
      .def(pybind11::init<>())
      .def("setToOrigin", &SE3Vertex::setToOriginImpl)
      .def_property("estimate", &SE3Vertex::estimate, &SE3Vertex::setEstimate)
      .def_property("fixed", &SE3Vertex::fixed, &SE3Vertex::setFixed)
      .def_property("id", &SE3Vertex::id, &SE3Vertex::setId)
      .def_property_readonly("dimension", &SE3Vertex::dimension);

  using MotionVertex = motreg::part::vertex::MotionVertex<double>;
  pybind11::class_<MotionVertex, g2o::OptimizableGraph::Vertex>(
      g2o, "MotionVertex", pybind11::multiple_inheritance())
      .def(pybind11::init<>())
      .def("setToOrigin", &MotionVertex::setToOriginImpl)
      .def_property("estimate", &MotionVertex::estimate,
                    &MotionVertex::setEstimate)
      .def_property("fixed", &MotionVertex::fixed, &MotionVertex::setFixed)
      .def_property("id", &MotionVertex::id, &MotionVertex::setId)
      .def_property_readonly("dimension", &MotionVertex::dimension);

  using EdgeFwdMotion = motreg::part::edge::EdgeFwdMotion<double>;
  pybind11::class_<EdgeFwdMotion, g2o::OptimizableGraph::Edge>(
      g2o, "EdgeFwdMotion", pybind11::multiple_inheritance())
      .def(pybind11::init<double>())
      .def("computeError", &EdgeFwdMotion::computeError)
      .def("linearizeOplus", &EdgeFwdMotion::linearizeOplus)
      .def("linearizeOplus",
           static_cast<void (EdgeFwdMotion::Base::*)(g2o::JacobianWorkspace &)>(
               &EdgeFwdMotion::Base::linearizeOplus))
      .def("setVertex", &EdgeFwdMotion::setVertex)
      .def_property("information",
                    static_cast<const EdgeFwdMotion::InformationType &(
                        EdgeFwdMotion::*)() const>(&EdgeFwdMotion::information),
                    &EdgeFwdMotion::setInformation)
      .def_property("id", &EdgeFwdMotion::id, &EdgeFwdMotion::setId)
      .def_property_readonly(
          "error", static_cast<const EdgeFwdMotion::ErrorVector &(
                       EdgeFwdMotion::*)() const>(&EdgeFwdMotion::error))
      .def_property_readonly("jacobian", &EdgeFwdMotion::jacobian);

  using EdgeBwdMotion = motreg::part::edge::EdgeBwdMotion<double>;
  pybind11::class_<EdgeBwdMotion, g2o::OptimizableGraph::Edge>(
      g2o, "EdgeBwdMotion", pybind11::multiple_inheritance())
      .def(pybind11::init<double>())
      .def("computeError", &EdgeBwdMotion::computeError)
      .def("linearizeOplus", &EdgeBwdMotion::linearizeOplus)
      .def("linearizeOplus",
           static_cast<void (EdgeBwdMotion::Base::*)(g2o::JacobianWorkspace &)>(
               &EdgeBwdMotion::Base::linearizeOplus))
      .def("setVertex", &EdgeBwdMotion::setVertex)
      .def_property("information",
                    static_cast<const EdgeBwdMotion::InformationType &(
                        EdgeBwdMotion::*)() const>(&EdgeBwdMotion::information),
                    &EdgeBwdMotion::setInformation)
      .def_property("id", &EdgeBwdMotion::id, &EdgeBwdMotion::setId)
      .def_property_readonly(
          "error", static_cast<const EdgeBwdMotion::ErrorVector &(
                       EdgeBwdMotion::*)() const>(&EdgeBwdMotion::error))
      .def_property_readonly("jacobian", &EdgeBwdMotion::jacobian);

  using EdgeLabelDiff = motreg::part::edge::EdgeLabelDiff<double>;
  pybind11::class_<EdgeLabelDiff, g2o::OptimizableGraph::Edge>(
      g2o, "EdgeLabelDiff", pybind11::multiple_inheritance())
      .def(pybind11::init<>())
      .def("computeError", &EdgeLabelDiff::computeError)
      .def("linearizeOplus", &EdgeLabelDiff::linearizeOplus)
      .def("linearizeOplus",
           static_cast<void (EdgeLabelDiff::Base::*)(g2o::JacobianWorkspace &)>(
               &EdgeLabelDiff::Base::linearizeOplus))
      .def("setVertex", &EdgeLabelDiff::setVertex)
      .def_property("measurement", &EdgeLabelDiff::measurement,
                    &EdgeLabelDiff::setMeasurement)
      .def_property("information",
                    static_cast<const EdgeLabelDiff::InformationType &(
                        EdgeLabelDiff::*)() const>(&EdgeLabelDiff::information),
                    &EdgeLabelDiff::setInformation)
      .def_property("id", &EdgeLabelDiff::id, &EdgeLabelDiff::setId)
      .def_property_readonly(
          "error", static_cast<const EdgeLabelDiff::ErrorVector &(
                       EdgeLabelDiff::*)() const>(&EdgeLabelDiff::error))
      .def_property_readonly("jacobian", &EdgeLabelDiff::jacobian);

  using EdgeConstantMotion = motreg::part::edge::EdgeConstantMotion<double>;
  pybind11::class_<EdgeConstantMotion, g2o::OptimizableGraph::Edge>(
      g2o, "EdgeConstantMotion", pybind11::multiple_inheritance())
      .def(pybind11::init<>())
      .def("computeError", &EdgeConstantMotion::computeError)
      .def("linearizeOplus", &EdgeConstantMotion::linearizeOplus)
      .def("linearizeOplus", static_cast<void (EdgeConstantMotion::Base::*)(
                                 g2o::JacobianWorkspace &)>(
                                 &EdgeConstantMotion::Base::linearizeOplus))
      .def("setVertex", &EdgeConstantMotion::setVertex)
      .def_property(
          "information",
          static_cast<const EdgeConstantMotion::InformationType &(
              EdgeConstantMotion::*)() const>(&EdgeConstantMotion::information),
          &EdgeConstantMotion::setInformation)
      .def_property("id", &EdgeConstantMotion::id, &EdgeConstantMotion::setId)
      .def_property_readonly(
          "error",
          static_cast<const EdgeConstantMotion::ErrorVector &(
              EdgeConstantMotion::*)() const>(&EdgeConstantMotion::error))
      .def_property_readonly("jacobian", &EdgeConstantMotion::jacobian);

  pybind11::class_<g2o::OptimizationAlgorithmProperty>(
      g2o, "OptimizationAlgorithmProperty")
      .def(pybind11::init<>())
      .def_readwrite("name", &g2o::OptimizationAlgorithmProperty::name)
      .def_readwrite("desc", &g2o::OptimizationAlgorithmProperty::desc)
      .def_readwrite("type", &g2o::OptimizationAlgorithmProperty::type)
      .def_readwrite("requiresMarginalize",
                     &g2o::OptimizationAlgorithmProperty::requiresMarginalize)
      .def_readwrite("poseDim", &g2o::OptimizationAlgorithmProperty::poseDim)
      .def_readwrite("landmarkDim",
                     &g2o::OptimizationAlgorithmProperty::landmarkDim);

  pybind11::class_<g2o::OptimizationAlgorithm>(g2o, "OptimizationAlgorithm");

  pybind11::class_<g2o::SparseOptimizer>(g2o, "SparseOptimizer")
      .def(pybind11::init<>())
      .def("initializeOptimization",
           static_cast<bool (g2o::SparseOptimizer::*)(int)>(
               &g2o::SparseOptimizer::initializeOptimization),
           pybind11::arg("level") = 0)
      .def("optimize", &g2o::SparseOptimizer::optimize,
           pybind11::arg("iterations"), pybind11::arg("online") = false)
      .def("addVertex", static_cast<bool (g2o::SparseOptimizer::*)(
                            g2o::OptimizableGraph::Vertex *v)>(
                            &g2o::SparseOptimizer::addVertex))
      .def("addEdge",
           static_cast<bool (g2o::SparseOptimizer::*)(
               g2o::OptimizableGraph::Edge *v)>(&g2o::SparseOptimizer::addEdge))
      .def_property("verbose", &g2o::SparseOptimizer::verbose,
                    &g2o::SparseOptimizer::setVerbose)
      .def_property("algorithm", &g2o::SparseOptimizer::algorithm,
                    &g2o::SparseOptimizer::setAlgorithm);

  using MotionModel = motreg::model::MotionModel<double>;
  pybind11::class_<MotionModel::ObjBBox, std::unique_ptr<MotionModel::ObjBBox>>(
      utils, "ObjBBox")
      .def(pybind11::init([](const pybind11::dict &objDict) {
        auto objBBox = std::make_unique<MotionModel::ObjBBox>();
        objBBox->timestamp = objDict["timestamp"].cast<double>();
        objBBox->sequence = objDict["sequence"].cast<int>();
        objBBox->pose =
            Sophus::SE3<double>(objDict["pose"].cast<Eigen::Matrix4d>());
        objBBox->poseFixed =
            getattr(objDict, "poseFixed", pybind11::bool_(false)).cast<bool>();
        auto motion = getattr(objDict, "motion", pybind11::none());
        objBBox->motion =
            motion.is_none()
                ? MotionModel::VertexObjMotion::EstimateType::Zero()
                : motion.cast<MotionModel::VertexObjMotion::EstimateType>();
        return objBBox;
      }))
      .def_readwrite("sequence", &MotionModel::ObjBBox::sequence)
      .def_readwrite("timestamp", &MotionModel::ObjBBox::timestamp)
      .def_readwrite("pose", &MotionModel::ObjBBox::pose)
      .def_readwrite("poseFixed", &MotionModel::ObjBBox::poseFixed)
      .def_readwrite("motion", &MotionModel::ObjBBox::motion)
      .def_readwrite("errMotionFwd", &MotionModel::ObjBBox::errMotionFwd)
      .def_readwrite("errMotionBwd", &MotionModel::ObjBBox::errMotionBwd)
      .def_readwrite("errLabel", &MotionModel::ObjBBox::errLabel)
      .def("toDict", [](const MotionModel::ObjBBox &self) {
        using namespace pybind11::literals;
        return pybind11::dict(
            "sequence"_a = self.sequence, "timestamp"_a = self.timestamp,
            "pose"_a = self.pose.matrix(), "poseFixed"_a = self.poseFixed,
            "motion"_a = self.motion, "errMotionFwd"_a = self.errMotionFwd,
            "errMotionBwd"_a = self.errMotionBwd, "errLabel"_a = self.errLabel);
      });

  pybind11::class_<MotionModel::MotionModelParams,
                   std::unique_ptr<MotionModel::MotionModelParams>>(
      utils, "MotionModelParams")
      .def(pybind11::init<>())
      .def(pybind11::init([](const pybind11::dict &paramsDict) {
        auto params = std::make_unique<MotionModel::MotionModelParams>();
        auto weightMotion =
            getattr(paramsDict, "weightMotion", pybind11::none());
        params->weightMotion =
            weightMotion.is_none()
                ? MotionModel::EdgeObjMotionFwd::InformationType::Identity()
                : weightMotion
                      .cast<MotionModel::EdgeObjMotionFwd::InformationType>();
        auto weightMotionConsistency =
            getattr(paramsDict, "weightMotionConsistency", pybind11::none());
        params->weightMotionConsistency =
            weightMotionConsistency.is_none()
                ? MotionModel::EdgeMotionConsistency::InformationType::
                      Identity()
                : weightMotionConsistency.cast<
                      MotionModel::EdgeMotionConsistency::InformationType>();
        auto weightObjPose2Label =
            getattr(paramsDict, "weightObjPose2Label", pybind11::none());
        params->weightObjPose2Label =
            weightObjPose2Label.is_none()
                ? MotionModel::EdgeObjPose2Label::InformationType::Identity()
                : weightObjPose2Label
                      .cast<MotionModel::EdgeObjPose2Label::InformationType>();
        params->objCtr2Origin =
            getattr(paramsDict, "objCtr2Origin", pybind11::float_(0.))
                .cast<decltype(params->objCtr2Origin)>();
        params->fixObjCtr2Origin =
            getattr(paramsDict, "fixObjCtr2Origin", pybind11::bool_(true))
                .cast<bool>();
        params->steps =
            getattr(paramsDict, "steps", pybind11::int_(100)).cast<int>();
        params->verbose =
            getattr(paramsDict, "verbose", pybind11::bool_(false)).cast<bool>();
        return params;
      }))
      .def_readwrite("weightMotion",
                     &MotionModel::MotionModelParams::weightMotion)
      .def_readwrite("weightMotionConsistency",
                     &MotionModel::MotionModelParams::weightMotionConsistency)
      .def_readwrite("weightObjPose2Label",
                     &MotionModel::MotionModelParams::weightObjPose2Label)
      .def_readwrite("objCtr2Origin",
                     &MotionModel::MotionModelParams::objCtr2Origin)
      .def_readwrite("fixObjCtr2Origin",
                     &MotionModel::MotionModelParams::fixObjCtr2Origin)
      .def_readwrite("steps", &MotionModel::MotionModelParams::steps)
      .def_readwrite("verbose", &MotionModel::MotionModelParams::verbose)
      .def("toDict", [](const MotionModel::MotionModelParams &self) {
        using namespace pybind11::literals;
        return pybind11::dict(
            "weightMotion"_a = self.weightMotion,
            "weightMotionConsistency"_a = self.weightMotionConsistency,
            "objCtr2Origin"_a = self.objCtr2Origin,
            "fixObjCtr2Origin"_a = self.fixObjCtr2Origin,
            "weightObjPose2Label"_a = self.weightObjPose2Label,
            "steps"_a = self.steps, "verbose"_a = self.verbose);
      });

  pybind11::class_<MotionModel, std::unique_ptr<MotionModel>>(m, "MotionModel")
      .def(pybind11::init<const std::vector<MotionModel::ObjBBox> &,
                          const MotionModel::MotionModelParams &>(),
           pybind11::arg("trajectory"), pybind11::arg("MotionModelParams"))
      .def("output",
           static_cast<MotionModel::ObjBBox (MotionModel::*)(int) const>(
               &MotionModel::output))
      .def("output", static_cast<std::vector<MotionModel::ObjBBox> (
                         MotionModel::*)(void) const>(&MotionModel::output))
      .def("output",
           static_cast<std::vector<MotionModel::ObjBBox> (MotionModel::*)(
               const std::vector<int> &) const>(&MotionModel::output))
      .def_property_readonly("debugVerticesTimestamps",
                             &MotionModel::debugVerticesTimestamps)
      .def_property_readonly("debugVertexObjCtr2Origin",
                             [](const MotionModel &self) {
                               return self.debugVertexObjCtr2Origin().get();
                             })
      .def("debugVerticesObjPose",
           [](const MotionModel &self) {
             pybind11::list ret;
             for (const auto &objPose : self.debugVerticesObjPose()) {
               auto pyObjPose = pybind11::cast(
                   *objPose, pybind11::return_value_policy::reference);
               ret.append(pyObjPose);
             }
             return ret;
           })
      .def("debugVerticesObjMotion", [](const MotionModel &self) {
        pybind11::list ret;
        for (const auto &objMotion : self.debugVerticesObjMotion()) {
          auto pyObjMotion = pybind11::cast(
              *objMotion, pybind11::return_value_policy::reference);
          ret.append(pyObjMotion);
        }
        return ret;
      });
}
