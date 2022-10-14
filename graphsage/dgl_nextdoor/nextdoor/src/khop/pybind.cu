#include <torch/script.h>
#include <torch/custom_class.h>
#include <string>

#include "main.cu"

TORCH_LIBRARY(my_classes, m)
{
  m.class_<NextdoorKHopSampler>("NextdoorKHopSampler")
      .def(torch::init<std::string>())
      .def("initSampling", &NextdoorKHopSampler::initSampling)
      .def("sample", &NextdoorKHopSampler::Sample)
      .def("finalSamples", &NextdoorKHopSampler::finalSamples)
      .def("finalSampleLength", &NextdoorKHopSampler::finalSampleLength)
      .def("freeDeviceMemory", &NextdoorKHopSampler::freeDeviceMemory);
}
