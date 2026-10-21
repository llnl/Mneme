#include "llvm/core.h"
#include "trace.hpp"
#include <cstring>
#include <llvm/Support/CBindingWrapping.h>
#include <llvm/Support/MemoryBuffer.h>
#include <mneme/MnemePython.hpp>

using namespace mneme;
using namespace mneme::python;
using namespace llvm;

extern "C" {
API_EXPORT(void *) MnemePY_getDeviceObject(LLVMMemoryBufferRef Buffer) {
  trace::Scope Trace("module.load");
  llvm::MemoryBuffer *DeviceObject = llvm::unwrap(Buffer);
  auto VendorModule = DeviceVendorTraits::getDeviceModuleFromImage(
      DeviceObject->getBufferStart());
  // NOTE: Device modules are effectively a  pointer to some structure.
  // so we can do this. We need to check also what happens in cuda.
  return reinterpret_cast<void *>(VendorModule);
}

API_EXPORT(void) MnemePY_DisposeDeviceObject(void *WModule) {
  trace::Scope Trace("module.unload");
  auto VendorModule =
      reinterpret_cast<DeviceVendorTraits::DeviceModule_t>(WModule);
  DeviceVendorTraits::DeviceModuleUnload(VendorModule);
  return;
}

API_EXPORT(void *)
MnemePY_getKernelFunctionFromImage(void *WrappedModule,
                                   const char *KernelName) {
  trace::Scope Trace("module.resolve_kernel");
  auto Module =
      reinterpret_cast<DeviceVendorTraits::DeviceModule_t>(WrappedModule);
  std::string KName(KernelName);
  auto Func = DeviceVendorTraits::getKernelFunctionFromImage(Module, KName);
  // NOTE: Device modules are effectively a  pointer to some structure.
  // so we can do this. We need to check also what happens in cuda.
  auto DevFunc = reinterpret_cast<void *>(Func);
  LOG_DEBUG("Device Function pointer is {}", DevFunc);
  return DevFunc;
}

API_EXPORT(int)
MnemePy_getRegisterUsage(void *Func) {
  int UsedRegisters = -1;
  auto DevFunc = reinterpret_cast<DeviceVendorTraits::DeviceFunction_t>(Func);
  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::deviceGetAttribute(
          DevFunc, FuncAttributes::REGISTER_USAGE, UsedRegisters));
  if (EC)
    LOG_FATAL("Error when reading register usage: " + EC.value());
  return UsedRegisters;
}

API_EXPORT(int)
MnemePy_getLocalMemUsage(void *Func) {
  int LocalMemUsage = -1;
  auto DevFunc = reinterpret_cast<DeviceVendorTraits::DeviceFunction_t>(Func);
  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::deviceGetAttribute(
          DevFunc, FuncAttributes::LOCALMEM_USAGE, LocalMemUsage));
  if (EC)
    LOG_FATAL("Error when reading register usage: " + EC.value());
  return LocalMemUsage;
}

API_EXPORT(int)
MnemePy_getConstMemUsage(void *Func) {
  int ConstMemUsage = 0;
  auto DevFunc = reinterpret_cast<DeviceVendorTraits::DeviceFunction_t>(Func);
  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::deviceGetAttribute(
          DevFunc, FuncAttributes::CONSTMEM_USAGE, ConstMemUsage));
  if (EC)
    LOG_FATAL("Error when reading register usage: " + EC.value());
  return ConstMemUsage;
}

API_EXPORT(void)
MnemePy_launchKernelFunction(void *Func, dim3 Grid, dim3 Block) {
  auto DevFunc = reinterpret_cast<DeviceVendorTraits::DeviceFunction_t>(Func);
  int Arg = 42;
  void *KernelArgs[] = {&Arg};

  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::launchKernelFunction(DevFunc, Grid, Block, KernelArgs,
                                               0, 0));
  if (EC)
    LOG_FATAL("Error When Launching Kernel: " + EC.value());

  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::DeviceSynchronize());

  if (EC)
    LOG_FATAL("Error When Launching Kernel: " + EC.value());
}

API_EXPORT(const char *) MnemePy_getDeviceArch() {
  auto Arch = DeviceVendorTraits::GetDeviceArch();
  auto *ret = strdup(Arch.c_str());
  return ret;
}

API_EXPORT(void)
MnemePy_profile(void *WrappedModule, void *Func, dim3 Grid, dim3 Block,
                MnemeDeviceMemStateRef Prologue,
                MnemeDeviceMemStateRef Epilogue, int SharedMemSize,
                int repeats) {
  auto VendorModule =
      reinterpret_cast<DeviceVendorTraits::DeviceModule_t>(WrappedModule);

  int DeviceId;
  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::getDevice(DeviceId));
  if (EC)
    LOG_FATAL("Error when requesting active device\n", EC.value());

  LOG_INFO("Executing replay run on device {}", DeviceId);

  DeviceVendorTraits::DeviceStream_t ReplayStream;
  auto DevFunc = reinterpret_cast<DeviceVendorTraits::DeviceFunction_t>(Func);
  auto PrologueState = unwrap(Prologue);
  auto EpilogueState = unwrap(Epilogue);
  EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::DeviceStreamCreate(&ReplayStream));
  if (EC)
    LOG_FATAL("Error when creating a stream for replay\n" + EC.value());

  {
    trace::Scope Trace("state.globals");
    PrologueState->initializeGlobals(VendorModule);
  }

  for (int i = 0; i < repeats; i++) {
    trace::Scope Iteration("iteration", i);
    LOG_DEBUG("Run {}/{}", i + 1, repeats);

    {
      trace::Scope Trace("state.restore");
      PrologueState->reset();
    }
    auto Args = PrologueState->getArgs();

    {
      trace::Scope Trace("sync.pre_launch");
      EC = DeviceVendorTraits::DeviceErrorCheck(
          DeviceVendorTraits::DeviceStreamSynchronize(ReplayStream));
    }

    if (EC)
      LOG_FATAL("Error when synchronizing device stream " + EC.value());

    {
      trace::Scope Trace("launch");
      EC = DeviceVendorTraits::DeviceErrorCheck(
          DeviceVendorTraits::launchKernelFunction(DevFunc, Grid, Block, Args,
                                                   SharedMemSize, ReplayStream));
    }
    if (EC)
      LOG_FATAL("Error When Launching Kernel: " + EC.value());

    {
      trace::Scope Trace("sync.post_launch");
      EC = DeviceVendorTraits::DeviceErrorCheck(
          DeviceVendorTraits::DeviceStreamSynchronize(ReplayStream));
    }

    if (EC)
      LOG_FATAL("Error When synchronizing with kernel stream: " + EC.value());
  }

  {
    trace::Scope Trace("sync.device");
    EC = DeviceVendorTraits::DeviceErrorCheck(
        DeviceVendorTraits::DeviceSynchronize());
  }

  if (EC)
    LOG_FATAL("Error when synchronizing device" + EC.value());
}

API_EXPORT(int) MnemePy_getNumArgs(MnemeDeviceMemStateRef WState) {
  auto State = unwrap(WState);
  return State->getNumArgs();
}

API_EXPORT(void **) MnemePy_getArgs(MnemeDeviceMemStateRef WState) {
  auto State = unwrap(WState);
  return reinterpret_cast<void **>(State->getArgs());
}

API_EXPORT(int) MnemePy_getDeviceCount() {
  int DevCount;
  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::getDeviceCount(DevCount));
  if (EC)
    LOG_FATAL("Error when getting the number of available devices\nEC: {}",
              EC.value());
  return DevCount;
}

API_EXPORT(void) MnemePy_setDevice(int DevId) {
  auto EC = DeviceVendorTraits::DeviceErrorCheck(
      DeviceVendorTraits::setDevice(DevId));
  if (EC)
    LOG_FATAL("Error when setting the device id\nEC: {}", EC.value());
}
}
