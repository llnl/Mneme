#pragma once
#include <dlfcn.h>
#include <optional>

#include "mneme/MnemeConfig.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeUtils.hpp"

#ifdef MNEME_ENABLE_HIP
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#define hipErrCheck(CALL)                                                      \
  {                                                                            \
    hipError_t err = CALL;                                                     \
    if (err != hipSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hipGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }

#define hiprtcErrCheck(CALL)                                                   \
  {                                                                            \
    hiprtcResult err = CALL;                                                   \
    if (err != HIPRTC_SUCCESS) {                                               \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hiprtcGetErrorString(err));                                       \
      abort();                                                                 \
    }                                                                          \
  }

#elif defined(MNEME_ENABLE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>

#define cudaErrCheck(CALL)                                                     \
  {                                                                            \
    cudaError_t err = CALL;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hipGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }

#define cuErrCheck(CALL)                                                       \
  {                                                                            \
    CUresult err = CALL;                                                       \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *name = nullptr, *desc = nullptr;                             \
      cuGetErrorName(err, &name);                                              \
      cuGetErrorString(err, &desc);                                            \
      fprintf(stderr, "CUDA Driver Error [%s]: %s\n", name ? name : "Unknown", \
              desc ? desc : "No description");                                 \
      abort();                                                                 \
    }                                                                          \
  }

#endif

namespace mneme {
enum DeviceVendors { HIP, CUDA };

enum FuncAttributes { REGISTER_USAGE, LOCALMEM_USAGE, CONSTMEM_USAGE };

template <DeviceVendors Type> struct DeviceTraits;

#if defined(MNEME_ENABLE_HIP)
template <> struct DeviceTraits<DeviceVendors::HIP> {
  using DeviceError_t = hipError_t;
  using DeviceStream_t = hipStream_t;
  using KernelFunction_t = hipFunction_t;
  using MemoryAllocationHandle_t = hipMemGenericAllocationHandle_t;
  using DeviceModule_t = hipModule_t;
  using DevicePtr_t = hipDeviceptr_t;
  using DeviceHandle_t = hipDevice_t;
  using DeviceContext_t = hipCtx_t;
  using DeviceFunction_t = hipFunction_t;
  using DeviceEvent_t = hipEvent_t;
  static constexpr auto DeviceSuccess = hipSuccess;

  static inline auto *getRTLib() { return dlopen("libamdhip64.so", RTLD_NOW); }
  static constexpr const char *getLaunchKernelFnName() {
    return "hipLaunchKernel";
  }
  static constexpr const char *getDeviceMallocFnName() { return "hipMalloc"; }
  static constexpr const char *getPinnedMallocFnName() {
    return "hipHostMalloc";
  }
  static constexpr const char *getManagedMallocFnName() {
    return "hipMallocManaged";
  }
  static constexpr const char *getDeviceFreeFnName() { return "hipFree"; }
  static constexpr const char *getPinnedFreeFnName() { return "hipHostFree"; }

  static const char *getDeviceGetIDFnName() { return "hipGetDevice"; }
  static const char *getDeviceSetIDFnName() { return "hipSetDevice"; }

  static constexpr bool hasFatBinEnd = false;

  static inline std::optional<std::string>
  DeviceErrorCheck(hipError_t ErrorCode) {
    if (ErrorCode == hipSuccess)
      return std::nullopt;
    return std::string(hipGetErrorString(ErrorCode));
  }

  static hipError_t DeviceStreamSynchronize(hipStream_t Stream) {
    return hipStreamSynchronize(Stream);
  }

  static hipError_t DeviceMemset(void *DevPtr, int Value, size_t Bytes) {
    auto EC = hipMemset(DevPtr, Value, Bytes);
    return EC;
  }

  static hipError_t DeviceMalloc(void **ptr, size_t size) {
    return hipMalloc(ptr, size);
  }

  static hipError_t DeviceFree(void *ptr) { return hipFree(ptr); }

  static hipError_t DeviceCopy(void *Dest, void *Src, size_t SizeBytes,
                               hipMemcpyKind Kind) {
    return hipMemcpy(Dest, Src, SizeBytes, Kind);
  }

  static hipError_t DeviceSynchronize() { return hipDeviceSynchronize(); }

  static std::string GetDeviceArch() {
    DeviceHandle_t Dev;
    DeviceContext_t Ctx;
    auto EC = DeviceErrorCheck(hipInit(0));
    if (EC)
      LOG_FATAL("Could not initialize device\n EC:" + EC.value());

    EC = DeviceErrorCheck(hipGetDevice(&Dev));
    if (EC)
      LOG_FATAL("Could not get device\n EC:" + EC.value());

    hipDeviceProp_t device_prop;

    // Get properties of the current device
    EC = DeviceErrorCheck(hipGetDeviceProperties(&device_prop, Dev));
    if (EC)
      LOG_FATAL("Could not get device properties\n EC:" + EC.value());

    std::string arch_name = device_prop.gcnArchName;
    LOG_DEBUG("Architecture name is {}", arch_name);

    std::string HipArch = arch_name.substr(0, arch_name.find(':'));

    return std::string(HipArch);
  }

  static DeviceModule_t getDeviceModuleFromImage(const void *Image) {
    hipModule_t HipModule;

    auto EC = DeviceErrorCheck(hipModuleLoadData(&HipModule, Image));
    if (EC)
      LOG_FATAL("Error with loading data from module\nEC:" + EC.value());
    return HipModule;
  }

  static void DeviceModuleUnload(DeviceModule_t Module) {
    auto EC = DeviceErrorCheck(hipModuleUnload(Module));
    if (EC)
      LOG_FATAL("Cannot unload module\nEC:" + EC.value());
  }

  static std::pair<void *, size_t>
  getGlobalAddrFromModule(hipModule_t &HipModule,
                          const std::string &GlobalName) {
    size_t Size;
    hipDeviceptr_t DevPtr;
    auto EC = DeviceErrorCheck(
        hipModuleGetGlobal(&DevPtr, &Size, HipModule, GlobalName.c_str()));
    if (EC)
      LOG_FATAL("Could not load global variable '" + GlobalName +
                "' from device module\n:EC:" + EC.value());
    return std::make_pair((void *)DevPtr, Size);
  }

  static hipFunction_t getKernelFunctionFromImage(hipModule_t &HipModule,
                                                  std::string &KernelName) {
    hipFunction_t KernelFunc;
    auto EC = DeviceErrorCheck(
        hipModuleGetFunction(&KernelFunc, HipModule, KernelName.c_str()));
    if (EC)
      LOG_FATAL("Error with loading kernel from Module");

    return KernelFunc;
  }

  static hipError_t getDeviceCount(int &devCount) {
    return hipGetDeviceCount(&devCount);
  }

  static hipError_t setDevice(int DeviceId) { return hipSetDevice(DeviceId); }

  static hipError_t getDevice(int &DeviceId) { return hipGetDevice(&DeviceId); }

  static hipError_t launchKernelFunction(hipFunction_t KernelFunc, dim3 GridDim,
                                         dim3 BlockDim, void **KernelArgs,
                                         uint64_t ShmemSize,
                                         hipStream_t Stream) {
    return hipModuleLaunchKernel(KernelFunc, GridDim.x, GridDim.y, GridDim.z,
                                 BlockDim.x, BlockDim.y, BlockDim.z, ShmemSize,
                                 Stream, KernelArgs, nullptr);
  }

  static inline uint64_t
  getPageSize(int DeviceID,
              const hipMemAllocationGranularity_flags Granularity) {
    uint64_t PageSize;
    hipMemAllocationProp Prop = {};
    Prop.type = hipMemAllocationTypePinned;
    Prop.location.type = hipMemLocationTypeDevice;
    Prop.location.id = DeviceID;
    // TODO: I could not find any documentation regarding the compressionType in
    // HIP. I will leave unitialized a.t.m.
    // Prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

    hipErrCheck(hipMemGetAllocationGranularity(&PageSize, &Prop, Granularity));
    return PageSize;
  }

  static constexpr hipMemcpyKind MemcpyHostToDeviceKind() {
    return hipMemcpyHostToDevice;
  }

  static constexpr hipMemcpyKind MemcpyDeviceToHostKind() {
    return hipMemcpyDeviceToHost;
  }

  static void mmap(hipMemGenericAllocationHandle_t &MHandle, void *Addr,
                   uint64_t Size, int DeviceID) {
    hipMemAllocationProp Prop = {};
    Prop.type = hipMemAllocationTypePinned;
    Prop.location.type = hipMemLocationTypeDevice;
    Prop.location.id = DeviceID;
    hipErrCheck(hipMemCreate(&MHandle, Size, &Prop, 0));
    hipErrCheck(hipMemMap((void *)Addr, Size, 0, MHandle, 0));

    hipMemAccessDesc ADesc = {};
    ADesc.location.type = hipMemLocationTypeDevice;
    ADesc.location.id = DeviceID;
    ADesc.flags = hipMemAccessFlagsProtReadWrite;

    hipErrCheck(hipMemSetAccess(Addr, Size, &ADesc, 1));
    grantHostAccess(Addr, Size, DeviceID);
  }

  // On an APU (MI300A) hipMalloc memory is host-accessible and applications and
  // GPU-aware MPI rely on it, but a VMM mapping is GPU-only: host reads see the
  // reservation placeholder (zeros on ROCm 6.4.0, SIGSEGV on 6.4.3+). HIP
  // before 7.1 ignores hipMemLocationTypeHost in hipMemSetAccess, so ask ROCr
  // directly.
  static void grantHostAccess(void *Addr, uint64_t Size, int DeviceID) {
    int Integrated = 0;
    hipErrCheck(hipDeviceGetAttribute(&Integrated, hipDeviceAttributeIntegrated,
                                      DeviceID));
    if (!Integrated)
      return;

    static hsa_agent_t CpuAgent = [] {
      hsa_agent_t Agent{0};
      hsa_iterate_agents(
          [](hsa_agent_t A, void *Data) {
            hsa_device_type_t Type;
            hsa_agent_get_info(A, HSA_AGENT_INFO_DEVICE, &Type);
            if (Type != HSA_DEVICE_TYPE_CPU)
              return HSA_STATUS_SUCCESS;
            *static_cast<hsa_agent_t *>(Data) = A;
            return HSA_STATUS_INFO_BREAK;
          },
          &Agent);
      return Agent;
    }();
    if (!CpuAgent.handle)
      LOG_FATAL("Cannot find the HSA CPU agent to grant host access");

    hsa_amd_memory_access_desc_t Desc{HSA_ACCESS_PERMISSION_RW, CpuAgent};
    if (hsa_amd_vmem_set_access(Addr, Size, &Desc, 1) != HSA_STATUS_SUCCESS)
      LOG_FATAL("Cannot grant host access to device memory at {} size {}", Addr,
                Size);
  }

  static uint64_t getMinPageSize(int DeviceID) {
    return getPageSize(DeviceID, hipMemAllocationGranularityMinimum);
  }

  static void *getVirtualAddress(uint64_t Size, void *VA, uint64_t Alignment) {
    hipDeviceptr_t devPtr = 0;

    hipErrCheck(hipMemAddressReserve(&devPtr, Size, Alignment,
                                     reinterpret_cast<hipDeviceptr_t>(VA), 0));
    return (void *)devPtr;
  }

  static void unmap(hipMemGenericAllocationHandle_t &MHandle, void *Addr,
                    uintptr_t Size) {
    LOG_DEBUG("Unmapping Addr:{} SIZE:{}", Addr, Size);
    hipErrCheck(hipMemUnmap(Addr, Size));
    hipErrCheck(hipMemRelease(MHandle));
  }

  static size_t getFixedMemorySize() {
    static uint64_t PageSize{Config::get().getPageSizeBytesOrDefault(64)};
    return PageSize;
  }

  static void freeVirtualAddress(void *Addr, size_t Size) {
    LOG_DEBUG("Releasing Device Virtual Address Pages:{} Size:{}", Addr, Size);
    auto EC = DeviceErrorCheck(hipMemAddressFree(Addr, Size));
    if (EC) {
      LOG_FATAL("Could not release VA addresses " + EC.value());
    }
  }

  static hipError_t DeviceStreamCreate(hipStream_t *Stream) {
    return hipStreamCreate(Stream);
  }

  static hipError_t deviceStreamDestroy(hipStream_t Stream) {
    return hipStreamDestroy(Stream);
  }

  static hipError_t deviceLaunchKernel(const void *function_address,
                                       dim3 numBlocks, dim3 dimBlocks,
                                       void **args, size_t sharedMemBytes,
                                       hipStream_t stream) {
    return hipLaunchKernel(function_address, numBlocks, dimBlocks, args,
                           sharedMemBytes, stream);
  }

  static hipError_t deviceGetSymbolAddress(void **devPtr, const void *symbol) {
    return hipGetSymbolAddress(devPtr, symbol);
  }

  static hipError_t deviceEventCreate(hipEvent_t *event) {
    return hipEventCreate(event);
  }

  static hipError_t deviceEventRecord(hipEvent_t event, hipStream_t stream) {
    return hipEventRecord(event, stream);
  }

  static hipError_t deviceEventDestroy(hipEvent_t event) {
    return hipEventDestroy(event);
  }

  static hipError_t deviceEventSynchronize(hipEvent_t event) {
    return hipEventSynchronize(event);
  }

  static hipError_t deviceEventElapsedTime(float *ms, hipEvent_t start,
                                           hipEvent_t stop) {
    return hipEventElapsedTime(ms, start, stop);
  }

  static hipError_t deviceGetAttribute(hipFunction_t &Func,
                                       FuncAttributes Attribute, int &Value) {
    LOG_DEBUG("Going to request attributes of {}", (void *)Func);

    switch (Attribute) {
    case REGISTER_USAGE:
      return hipFuncGetAttribute(&Value, HIP_FUNC_ATTRIBUTE_NUM_REGS, Func);
    case LOCALMEM_USAGE:
      return hipFuncGetAttribute(&Value, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                 Func);
    case CONSTMEM_USAGE:
      return hipFuncGetAttribute(&Value, HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
                                 Func);
    default:
      LOG_FATAL("Request unknown attribute");
      break;
    }
    return hipErrorInvalidValue;
  }
};
#elif defined(MNEME_ENABLE_CUDA)
template <> struct DeviceTraits<DeviceVendors::CUDA> {

  using DeviceError_t = cudaError_t;
  using DeviceDriverError_t = CUresult;

  using DeviceStream_t = cudaStream_t;

  using KernelFunction_t = CUfunction;

  using MemoryAllocationHandle_t = CUmemGenericAllocationHandle;

  using DeviceModule_t = CUmodule;
  using DevicePtr_t = CUdeviceptr;

  using DeviceHandle_t = CUdevice;
  using DeviceContext_t = CUcontext;
  using DeviceFunction_t = CUfunction;

  using DeviceEvent_t = cudaEvent_t;
  static constexpr auto DeviceSuccess = cudaSuccess;
  static constexpr auto DeviceDriverSuccess = CUDA_SUCCESS;

  static inline auto *getRTLib() { return dlopen("libcudart.so", RTLD_NOW); }
  static constexpr const char *getLaunchKernelFnName() {
    return "cudaLaunchKernel";
  }
  static constexpr const char *getDeviceMallocFnName() { return "cudaMalloc"; }
  static constexpr const char *getPinnedMallocFnName() {
    return "cudaMallocHost";
  }
  static constexpr const char *getManagedMallocFnName() {
    return "cudaMallocManaged";
  }
  static constexpr const char *getDeviceFreeFnName() { return "cudaFree"; }
  static constexpr const char *getPinnedFreeFnName() { return "cudaFreeHost"; }
  static constexpr const char *getUURegisterFunctionFnName() {
    return "__cudaRegisterFunction";
  }
  static const char *getUURegisterVarFnName() { return "__cudaRegisterVar"; }
  static const char *getUURegisterFatbinFnName() {
    return "__cudaRegisterFatBinary";
  }

  static const char *getUURegisterFatbinEndFnName() {
    return "__cudaRegisterFatBinaryEnd";
  }

  static const char *getUUUnRegisterFatBinaryFnName() {
    return "__cudaUnregisterFatBinary";
  }

  static const char *getDeviceGetIDFnName() { return "cudaGetDevice"; }
  static const char *getDeviceSetIDFnName() { return "cudaSetDevice"; }

  static constexpr bool hasFatBinEnd = true;

  static inline std::optional<std::string>
  DeviceErrorCheck(DeviceError_t ErrorCode) {
    if (ErrorCode == DeviceSuccess)
      return std::nullopt;
    return std::string(cudaGetErrorString(ErrorCode));
  }

  static inline std::optional<std::string>
  DeviceErrorCheck(DeviceDriverError_t ErrorCode) {
    if (ErrorCode == DeviceDriverSuccess)
      return std::nullopt;

    if (ErrorCode == CUDA_ERROR_DEINITIALIZED) {
      LOG_WARN("Device is deinitialized ignoring errors");
      return std::nullopt;
    }

    const char *name = nullptr, *desc = nullptr;
    cuGetErrorName(ErrorCode, &name);
    cuGetErrorString(ErrorCode, &desc);
    auto EC = std::string("Error:") + std::to_string(ErrorCode) + ":";
    if (name)
      EC += std::string(name);
    if (desc)
      EC += std::string(" description:") + std::string(name);

    return EC;
  }

  static DeviceError_t DeviceStreamSynchronize(DeviceStream_t Stream) {
    return cudaStreamSynchronize(Stream);
  }

  static DeviceError_t DeviceMemset(void *DevPtr, int Value, size_t Bytes) {
    auto EC = cudaMemset(DevPtr, Value, Bytes);
    return EC;
  }

  static DeviceError_t DeviceMalloc(void **ptr, size_t size) {
    return cudaMalloc(ptr, size);
  }

  static DeviceError_t DeviceFree(void *ptr) { return cudaFree(ptr); }

  static DeviceError_t DeviceCopy(void *Dest, void *Src, size_t SizeBytes,
                                  cudaMemcpyKind Kind) {
    return cudaMemcpy(Dest, Src, SizeBytes, Kind);
  }

  static DeviceError_t DeviceSynchronize() { return cudaDeviceSynchronize(); }

  static std::string GetDeviceArch() {
    DeviceHandle_t Dev;
    DeviceContext_t Ctx;
    auto EC = DeviceErrorCheck(cuInit(0));
    if (EC)
      LOG_FATAL("Could not initialize device\n EC:" + EC.value());

    EC = DeviceErrorCheck(cudaGetDevice(&Dev));
    if (EC)
      LOG_FATAL("Could not get device\n EC:" + EC.value());

    EC = DeviceErrorCheck(cuCtxGetCurrent(&Ctx));
    if (EC)
      LOG_FATAL("Could not get current CUDA context\nEC:" + EC.value());

    if (!Ctx) {
      EC = DeviceErrorCheck(cuCtxCreate(&Ctx, 0, Dev));
      if (EC)
        LOG_FATAL("Could not create new context\nEC:" + EC.value());
    }

    int CCMajor;
    EC = DeviceErrorCheck(cuDeviceGetAttribute(
        &CCMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, Dev));
    if (EC)
      LOG_FATAL("Could not get device major attribute\n EC:" + EC.value());
    int CCMinor;
    DeviceErrorCheck(cuDeviceGetAttribute(
        &CCMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, Dev));
    if (EC)
      LOG_FATAL("Could not get device minor attribute\n EC:" + EC.value());

    std::string DeviceArch = "sm_" + std::to_string(CCMajor * 10 + CCMinor);

    return DeviceArch;
  }

  static DeviceModule_t getDeviceModuleFromImage(const void *Image) {
    DeviceModule_t cudaModule;

    auto EC = DeviceErrorCheck(cuModuleLoadData(&cudaModule, Image));
    if (EC)
      LOG_FATAL("Error with loading data from module\nEC:" + EC.value());
    return cudaModule;
  }

  static void DeviceModuleUnload(DeviceModule_t Module) {
    auto EC = DeviceErrorCheck(cuModuleUnload(Module));
    if (EC)
      LOG_FATAL("Cannot unload module\nEC:" + EC.value());
  }

  static std::pair<void *, size_t>
  getGlobalAddrFromModule(DeviceModule_t &cudaModule,
                          const std::string &GlobalName) {
    size_t Size;
    DevicePtr_t DevPtr;
    auto EC = DeviceErrorCheck(
        cuModuleGetGlobal(&DevPtr, &Size, cudaModule, GlobalName.c_str()));
    if (EC)
      LOG_FATAL("Could not load global variable '" + GlobalName +
                "' from device module\n:EC:" + EC.value());
    return std::make_pair((void *)DevPtr, Size);
  }

  static DeviceFunction_t getKernelFunctionFromImage(DeviceModule_t &Module,
                                                     std::string &KernelName) {
    DeviceFunction_t KernelFunc;
    auto EC = DeviceErrorCheck(
        cuModuleGetFunction(&KernelFunc, Module, KernelName.c_str()));
    if (EC)
      LOG_FATAL("Error with loading kernel from Module");

    return KernelFunc;
  }

  static cudaError_t getDeviceCount(int &devCount) {
    return cudaGetDeviceCount(&devCount);
  }

  static cudaError_t setDevice(int DeviceId) { return cudaSetDevice(DeviceId); }

  static cudaError_t getDevice(int &DeviceId) {
    return cudaGetDevice(&DeviceId);
  }

  static DeviceDriverError_t launchKernelFunction(DeviceFunction_t KernelFunc,
                                                  dim3 GridDim, dim3 BlockDim,
                                                  void **KernelArgs,
                                                  uint64_t ShmemSize,
                                                  cudaStream_t Stream) {
    return cuLaunchKernel(KernelFunc, GridDim.x, GridDim.y, GridDim.z,
                          BlockDim.x, BlockDim.y, BlockDim.z, ShmemSize, Stream,
                          KernelArgs, nullptr);
  }

  static inline uint64_t
  getPageSize(int DeviceID,
              const CUmemAllocationGranularity_flags Granularity) {
    uint64_t PageSize;
    CUmemAllocationProp Prop = {};
    Prop.type = CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    Prop.location.type = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    Prop.location.id = DeviceID;

    auto EC = DeviceErrorCheck(
        cuMemGetAllocationGranularity(&PageSize, &Prop, Granularity));
    if (EC)
      LOG_FATAL("Could not get Allocation Granularity\nEC:" + EC.value());
    return PageSize;
  }

  static constexpr cudaMemcpyKind MemcpyHostToDeviceKind() {
    return cudaMemcpyHostToDevice;
  }

  static constexpr cudaMemcpyKind MemcpyDeviceToHostKind() {
    return cudaMemcpyDeviceToHost;
  }

  static void mmap(MemoryAllocationHandle_t &MHandle, void *Addr,
                   uintptr_t Size, int DeviceID) {
    CUmemAllocationProp Prop = {};
    Prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    Prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    Prop.location.id = DeviceID;
    auto EC = DeviceErrorCheck(cuMemCreate(&MHandle, Size, &Prop, 0));
    if (EC)
      LOG_FATAL("Cannot create memory handle\nEC:" + EC.value());
    EC = DeviceErrorCheck(cuMemMap((DevicePtr_t)Addr, Size, 0, MHandle, 0));
    if (EC)
      LOG_FATAL("Cannot map memory handle\nEC:" + EC.value());

    CUmemAccessDesc ADesc = {};
    ADesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ADesc.location.id = DeviceID;
    ADesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    EC = DeviceErrorCheck(cuMemSetAccess((DevicePtr_t)Addr, Size, &ADesc, 1));
    if (EC)
      LOG_FATAL("Cannot set memory Access\nEC:" + EC.value());
  }

  static uint64_t getMinPageSize(int DeviceID) {
    return getPageSize(DeviceID, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  }

  static void *getVirtualAddress(uint64_t Size, void *VA, uint64_t Alignment) {
    DevicePtr_t devPtr = 0;

    cuErrCheck(cuMemAddressReserve(&devPtr, Size, Alignment,
                                   reinterpret_cast<DevicePtr_t>(VA), 0));
    return (void *)devPtr;
  }

  static void unmap(MemoryAllocationHandle_t &MHandle, void *Addr,
                    uintptr_t Size) {
    LOG_DEBUG("Unmapping Addr:{} SIZE:{}", Addr, Size);
    auto EC = DeviceErrorCheck(cuMemUnmap((DevicePtr_t)Addr, Size));
    if (EC)
      LOG_FATAL("Cannot set unmap memory\nEC:" + EC.value());

    EC = DeviceErrorCheck(cuMemRelease(MHandle));
    if (EC)
      LOG_FATAL("Cannot set cuMemRelease Handle\nEC:" + EC.value());
  }

  static size_t getFixedMemorySize() {
    static uint64_t PageSize{Config::get().getPageSizeBytesOrDefault(2)};
    return PageSize;
  }

  static void freeVirtualAddress(void *Addr, size_t Size) {
    LOG_DEBUG("Releasing Device Virtual Address Pages:{} Size:{}", Addr, Size);
    auto Ret = cuMemAddressFree((DevicePtr_t)Addr, Size);
    LOG_DEBUG("Done from driver call ({} {})", Addr, Size);
    auto EC = DeviceErrorCheck(Ret);
    if (EC) {
      LOG_FATAL("Could not release VA addresses " + EC.value());
    }
  }

  static bool compareDeviceBlobs(const char *Blob1, const char *Blob2,
                                 uint64_t NumBytes);

  static DeviceError_t DeviceStreamCreate(DeviceStream_t *Stream) {
    return cudaStreamCreate(Stream);
  }

  static DeviceError_t deviceStreamDestroy(DeviceStream_t Stream) {
    return cudaStreamDestroy(Stream);
  }

  static DeviceError_t deviceLaunchKernel(const void *kernelFunc, dim3 gridDim,
                                          dim3 blockDim, void **kernelArgs,
                                          size_t sharedMemBytes,
                                          DeviceStream_t stream) {
    auto EC = DeviceErrorCheck(cudaLaunchKernel(
        kernelFunc, gridDim, blockDim, kernelArgs, sharedMemBytes, stream));
    if (EC)
      LOG_WARN("Launching Kernel return error {}", EC.value());

    return cudaGetLastError();
  }

  static DeviceError_t deviceEventCreate(DeviceEvent_t *event) {
    return cudaEventCreate(event);
  }

  static DeviceError_t deviceEventRecord(DeviceEvent_t event,
                                         DeviceStream_t stream) {
    return cudaEventRecord(event, stream);
  }

  static DeviceError_t deviceEventDestroy(DeviceEvent_t event) {
    return cudaEventDestroy(event);
  }

  static DeviceError_t deviceEventSynchronize(DeviceEvent_t event) {
    return cudaEventSynchronize(event);
  }

  static DeviceError_t deviceEventElapsedTime(float *ms, DeviceEvent_t start,
                                              DeviceEvent_t stop) {
    return cudaEventElapsedTime(ms, start, stop);
  }

  static DeviceError_t deviceGetSymbolAddress(void **devPtr,
                                              const void *symbol) {
    return cudaGetSymbolAddress(devPtr, symbol);
  }

  static DeviceDriverError_t deviceGetAttribute(cudaFunction_t &Func,
                                                FuncAttributes Attribute,
                                                int &Value) {
    LOG_DEBUG("Going to request attributes of {}", (void *)Func);

    switch (Attribute) {
    case REGISTER_USAGE:
      return cuFuncGetAttribute(&Value, CU_FUNC_ATTRIBUTE_NUM_REGS, Func);

    case LOCALMEM_USAGE:
      return cuFuncGetAttribute(&Value, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                Func);

    case CONSTMEM_USAGE:
      return cuFuncGetAttribute(&Value, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
                                Func);
    default:
      LOG_FATAL("Request unknown attribute");
      break;
    }
  }
};
#else
#endif

} // namespace mneme
