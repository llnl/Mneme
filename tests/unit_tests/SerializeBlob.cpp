#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeMemory.hpp"
#include <cstdint>
#include <random>

using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
using MnemeMemoryBlobDevice = MnemeMemoryBlob<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
using MnemeMemoryBlobDevice = MnemeMemoryBlob<DeviceVendors::CUDA>;
#endif

void initializeRandomBuffer(uint8_t *Buffer, size_t Size) {
  // Random number generation setup
  std::mt19937 gen(4); // Mersenne Twister random number generator
  std::uniform_int_distribution<uint8_t> dis(
      0, 255); // Uniform distribution for char range

  // Fill the buffer with random values
  for (int I = 0; I < Size; I++) {
    Buffer[I] = dis(gen);
  }
}

int main(int argc, char **argv) {
  // We allocate some "fake" data 
  uint8_t *HData = new uint8_t[128];
  initializeRandomBuffer(HData, 128);
  uint8_t *DData;

  auto EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceMalloc((void **)&DData, 128));
  if (EC)
    LOG_FATAL("Could not allocate device data");

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
      DData, HData, 128, MnemeDeviceRT::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Could not allocate device data");

  MnemeMemoryBlobDevice Blob(128L, DData, 128L);
  Blob.setHostData(std::unique_ptr<uint8_t[]>(new uint8_t[128]));

  llvm::SmallVector<char, 128> Buffer;

  // Create a raw_svector_ostream using the buffer
  llvm::raw_svector_ostream Stream(Buffer);
  Stream << Blob;

  auto Buff = const_cast<const char *>(Buffer.data());
  auto entry = MnemeMemoryBlobDevice::fromBuffer(Buff);
  auto Addr = entry.first;
  auto SBlob = std::move(entry.second);

  auto Ret = [&]() {
    if (reinterpret_cast<void *>(Blob.getBlobAddr()) != Addr) {
      std::cerr << "Blob differerent Addresses " << Addr << " "
                << Blob.getBlobAddr() << "\n";
      return -1;
    }

    if (Blob.getActualSize() != SBlob.getActualSize()) {
      std::cerr << "Sizes differ" << Blob.getActualSize() << " "
                << SBlob.getActualSize() << "\n";
      return -1;
    }

    if (Blob.getSize() != SBlob.getSize()) {
      std::cerr << "Sizes differ" << Blob.getActualSize() << " "
                << SBlob.getActualSize() << "\n";
      return -1;
    }
    uint8_t *data = SBlob.getHostData();
    if (std::memcmp(reinterpret_cast<void *>(HData),
                    reinterpret_cast<void *>(data), 128) != 0) {
      std::cerr << "Memory differs between GV and GVR\n";
      return -1;
    }
    return 0;
  }();

  HData = nullptr;

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceFree(DData));
  if (EC)
    LOG_FATAL("Could not release device memory\n");

  return Ret;
}
