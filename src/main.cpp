#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "Lodestar/src/systems/StateSpace.hpp"
#include "Lodestar/src/analysis/ZeroOrderHold.hpp"
#include "Lodestar/src/analysis/BilinearTransformation.hpp"
#include "Lodestar/src/analysis/LinearSystemInverse.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j)
{
    return i + j;
}

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(lodestar, m)
{
    m.doc() = R"pbdoc(
        Lodestar Python bindings
        -----------------------

        .. currentmodule:: lodestar

        .. autosummary::
           :toctree: _generate

           StateSpace
           analysis
           ZeroOrderHold
           BilinearTransformation
           LinearSystemInverse
    )pbdoc";

//    m.attr("StateSpace") = "systems.StateSpace";
//    m.import(".systems.StateSpace");

    using StateSpaceDynamic = ls::systems::StateSpace<double, -1, -1, -1>;
    using ZOH = ls::analysis::ZeroOrderHold;
    using BLTF = ls::analysis::BilinearTransformation;
    using LTIINV = ls::analysis::LinearSystemInverse;

//    py::module m_zoh = m.def_submodule("ZeroOrderHold");
//    m_zoh.def("c2d", static_cast<StateSpaceDynamic (ZOH::*)(const StateSpaceDynamic&, double)>(&ZOH::c2d));

    py::module m_systems = m.def_submodule("systems", "Dynamical systems.");

    m_systems.doc() = R"pbdoc(
            Lodestar systems module Python bindings
            ----------------------------------------

            .. currentmodule:: lodestar.systems

            .. autosummary::
               :toctree: _generate

               StateSpace
    )pbdoc";

    py::class_<StateSpaceDynamic>(m_systems, "StateSpace",
                                  "Linear time-invariant state space system.")
            .def(py::init<>(), "Default constructor.")
            .def(py::init<StateSpaceDynamic::TDStateMatrix *, StateSpaceDynamic::TDInputMatrix *, StateSpaceDynamic::TDOutputMatrix *, StateSpaceDynamic::TDFeedforwardMatrix *>(),
                 "Constructs a linear state space system from four matrices.",
                 "A"_a, "B"_a, "C"_a, "D"_a)
            .def(py::init<const StateSpaceDynamic &>(), "Copy constructor.")
            .def("stateDim", &StateSpaceDynamic::stateDimDynamic, "Returns the state dimension.")
            .def("inputDim", &StateSpaceDynamic::inputDimDynamic, "Returns the input dimension.")
            .def("outputDim", &StateSpaceDynamic::outputDimDynamic, "Returns the output dimension.")
            .def("isDiscrete", &StateSpaceDynamic::isDiscrete, "Returns whether the system is a discrete-time system.")
            .def("isStable", &StateSpaceDynamic::isStable,
                 "Checks if the system is stable.",
                 "tolerance"_a = 0)
            .def("getSamplingPeriod", &StateSpaceDynamic::getSamplingPeriod, "Returns the system's sampling period.")
            .def("setSamplingPeriod", &StateSpaceDynamic::setSamplingPeriod,
                 "Sets the system's sampling period.",
                 "dt"_a)
            .def("setDiscreteParams",
                 static_cast<void (StateSpaceDynamic::*)(double, bool)>(&StateSpaceDynamic::setDiscreteParams),
                 "Sets the system's discrete-time parameters.",
                 "dt"_a, "discrete"_a = true)
//        .def("copyMatrices", static_cast<void (StateSpaceDynamic::*)(const StateSpaceDynamic &)>(&StateSpaceDynamic::copyMatrices))
            .def("getA", &StateSpaceDynamic::getA, "Returns the system's state matrix.")
            .def("setA",
                 static_cast<void (StateSpaceDynamic::*)(StateSpaceDynamic::TDStateMatrix *)>(&StateSpaceDynamic::setA),
                 "Sets the systems state matrix",
                 "A"_a)
            .def("getB", &StateSpaceDynamic::getB, "Returns the system's input matrix.")
            .def("setB",
                 static_cast<void (StateSpaceDynamic::*)(StateSpaceDynamic::TDInputMatrix *)>(&StateSpaceDynamic::setB),
                 "Sets the system's input matrix.",
                 "B"_a)
            .def("getC", &StateSpaceDynamic::getC, "Returns the system's output matrix.")
            .def("setC", static_cast<void (StateSpaceDynamic::*)(
                         StateSpaceDynamic::TDOutputMatrix *)>(&StateSpaceDynamic::setC),
                 "Sets the system's output matrix",
                 "C"_a)
            .def("getD", &StateSpaceDynamic::getD, "Returns the system's feedforward matrix.")
            .def("setD", static_cast<void (StateSpaceDynamic::*)(
                         StateSpaceDynamic::TDFeedforwardMatrix *)>(&StateSpaceDynamic::setD),
                 "Sets the system's feedforward matrix."
                 "D"_a);

    py::module m_analysis = m.def_submodule("analysis", "Analysis routines.");

    m_analysis.doc() = R"pbdoc(
        Lodestar analysis module Python bindings
        ----------------------------------------

        .. currentmodule:: lodestar.analysis

        .. autosummary::
           :toctree: _generate

           ZeroOrderHold
           BilinearTransformation
           LinearSystemInverse
    )pbdoc";

    m_analysis.def_submodule("ZeroOrderHold",
                             "Routines for computing zero-order hold transformation on state space systems.")
            .def("c2d", static_cast<StateSpaceDynamic (*)(const StateSpaceDynamic &, double)>(&ZOH::c2d),
                 "Generates zero-order hold discretization from a continuous-time state space system.",
                 "ss"_a, "dt"_a)
            .def("c2d", static_cast<StateSpaceDynamic (*)(const Eigen::MatrixXd &, const Eigen::MatrixXd &,
                                                          const Eigen::MatrixXd &, const Eigen::MatrixXd &,
                                                          double)>(&ZOH::c2d),
                 "Generates zero-order hold discretization from a continuous-time state space system.",
                 "A"_a, "B"_a, "C"_a, "D"_a, "dt"_a)
            .def("d2c", static_cast<StateSpaceDynamic (*)(const StateSpaceDynamic &)>(&ZOH::d2c),
                 "Reverts a zero-order hold discretization on a discrete-time state space system.",
                 "ss"_a)
            .def("d2c", static_cast<StateSpaceDynamic (*)(const StateSpaceDynamic &, double)>(&ZOH::d2c),
                 "Reverts a zero-order hold discretization on a discrete-time state space system.",
                 "ss"_a, "dt"_a)
            .def("d2c", static_cast<StateSpaceDynamic (*)(const Eigen::MatrixXd &, const Eigen::MatrixXd &,
                                                          const Eigen::MatrixXd &, const Eigen::MatrixXd &,
                                                          double)>(&ZOH::d2c),
                 "Reverts a zero-order hold discretization on a discrete-time state space system.",
                 "A"_a, "B"_a, "C"_a, "D"_a, "dt"_a);


    m_analysis.def_submodule("BilinearTransformation",
                             "Routines for converting a state space system from continuous-to discrete-time and vice versa using the generalized bilinear transformation.")
            .def("c2d",
                 static_cast<StateSpaceDynamic (*)(const StateSpaceDynamic &, double, double)>(&BLTF::c2d),
                 "Generates generalized bilinear transform of a continuous-time state space system.",
                 "ss"_a, "dt"_a, "alpha"_a = 1)
            .def("c2d", static_cast<StateSpaceDynamic (*)(const Eigen::MatrixXd &, const Eigen::MatrixXd &,
                                                          const Eigen::MatrixXd &, const Eigen::MatrixXd &,
                                                          double, double)>(&BLTF::c2d),
                 "Generates generalized bilinear transform of a continuous-time state space system.",
                 "A"_a, "B"_a, "C"_a, "D"_a, "dt"_a, "alpha"_a = 1)
            .def("d2c", static_cast<StateSpaceDynamic (*)(const StateSpaceDynamic &, double)>(&BLTF::d2c),
                 "Generates generalized bilinear transform of a discrete-time state space system.",
                 "ss"_a, "alpha"_a)
            .def("d2c",
                 static_cast<StateSpaceDynamic (*)(const StateSpaceDynamic &, double, double)>(&BLTF::d2c),
                 "Generates generalized bilinear transform of a discrete-time state space system.",
                 "ss"_a, "dt"_a, "alpha"_a)
            .def("d2c", static_cast<StateSpaceDynamic (*)(const Eigen::MatrixXd &, const Eigen::MatrixXd &,
                                                          const Eigen::MatrixXd &, const Eigen::MatrixXd &,
                                                          double)>(&ZOH::d2c),
                 "Generates generalized bilinear transform of a discrete-time state space system.",
                 "A"_a, "B"_a, "C"_a, "D"_a, "dt"_a);

    m_analysis.def_submodule("LinearSystemInverse",
                             "Routines for computing the  inverse of a continuous-time state space systems.")
            .def("inverse", static_cast<StateSpaceDynamic (*)(const StateSpaceDynamic &)>(&LTIINV::inverse),
                 "Generates the inverse of a continuous-time state space system.",
                 "ss"_a);

//    m.def("add", &add, R"pbdoc(
//        Add two numbers
//
//        Some other explanation about the add function.
//    )pbdoc", py::arg("i"), py::arg("j"));
//
//    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
//        Subtract two numbers
//
//        Some other explanation about the subtract function.
//    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
