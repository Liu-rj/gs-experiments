#include <libNextDoor.hpp>
#include <main.cu>
#include "khop.cu"
#include <string>
#include <vector>

typedef KHopApp KHopSampling;

// Declare NextDoorData
static NextDoorData<KHopSample, KHopSampling> nextDoorData;

// #include "check_results.cu"
// int main(int argc, char *argv[])
// {
//   // Call appMain to run as a standalone executable
//   return appMain<KHopSample, KHopSampling>(argc, argv, checkSampledVerticesResult<KHopSample, KHopSampling>);
// }

static CSR *graph_csr = nullptr;
static Graph graph;
static std::vector<GPUCSRPartition> gpuCSRPartitions;

struct NextdoorKHopSampler : torch::CustomClassHolder
{

  NextdoorKHopSampler() {}
  NextdoorKHopSampler(std::string filename)
      : _file_name(filename) {}

  void initSampling()
  {
    graph_csr = loadGraph(graph, &_file_name[0], (char *)"edge-list", (char *)"binary");
    assert(graph_csr != nullptr);
    allocNextDoorDataOnGPU(graph_csr, nextDoorData);
    gpuCSRPartitions = transferCSRToGPUs(nextDoorData, graph_csr);
    nextDoorData.gpuCSRPartitions = gpuCSRPartitions;
  }

  void Sample()
  {
    doSampleParallelSampling(graph_csr, nextDoorData);
  }

  torch::Tensor finalSamples()
  {
    return getFinalSamples(nextDoorData);
  }

  int64_t finalSampleLength()
  {
    const size_t numSamples = nextDoorData.samples.size();
    const size_t finalSampleSize = getFinalSampleSize<KHopApp>();
    return numSamples * finalSampleSize;
  }

  void freeDeviceMemory()
  {
    freeDeviceData(nextDoorData);
  }

  torch::Tensor _finalSamples;
  std::string _file_name;
};
