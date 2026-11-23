#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeMemory.hpp"
#include <cstdint>
#include <random>
#include <vector>

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

Metadata typed(BuiltinDType Type, const char *Tag) {
  Metadata Md;
  Md.builtin = Type;
  Md.tag = std::string(Tag);
  return Md;
}

bool expectAnnotations(const MnemeMemoryBlobDevice &Blob,
                       const std::vector<MemoryAnnotation> &Expected,
                       const char *Step) {
  if (Blob.getAnnotations() == Expected)
    return true;
  std::cerr << "Unexpected annotations after " << Step << ":";
  for (const auto &Subregion : Blob.getAnnotations())
    std::cerr << " [" << Subregion.Range.Offset << ", "
              << Subregion.Range.endOffset() << ") "
              << Subregion.MD.tag.value_or("no_tag");
  std::cerr << "\n";
  return false;
}

// Later annotations overwrite earlier ones on the bytes they cover, and
// subregions with equal metadata stay separate.
bool checkOverwrites() {
  MnemeMemoryBlobDevice Blob(128, nullptr, 128);
  auto Whole = typed(BuiltinDType::F64, "whole");
  auto First = typed(BuiltinDType::F32, "first");
  auto Second = typed(BuiltinDType::F32, "second");

  bool Ok = Blob.setAnnotation({32, 32}, First) == AnnotationStatus::Ok &&
            expectAnnotations(
                Blob, {{{0, 32}, {}}, {{32, 32}, First}, {{64, 64}, {}}},
                "a region inside the default subregion");
  Ok = Ok && Blob.setAnnotation({48, 32}, Second) == AnnotationStatus::Ok &&
       expectAnnotations(Blob,
                         {{{0, 32}, {}},
                          {{32, 16}, First},
                          {{48, 32}, Second},
                          {{80, 48}, {}}},
                         "an overlapping second region");
  Ok = Ok && Blob.setAnnotation({80, 16}, Second) == AnnotationStatus::Ok &&
       expectAnnotations(Blob,
                         {{{0, 32}, {}},
                          {{32, 16}, First},
                          {{48, 32}, Second},
                          {{80, 16}, Second},
                          {{96, 32}, {}}},
                         "an adjacent region with equal metadata");
  Ok = Ok && Blob.setAnnotation({0, 128}, Whole) == AnnotationStatus::Ok &&
       expectAnnotations(Blob, {{{0, 128}, Whole}}, "a whole-blob annotation");
  return Ok;
}

// A rejected annotation leaves the blob unchanged.
bool checkAlignment() {
  MnemeMemoryBlobDevice Blob(128, nullptr, 128);
  auto F32 = typed(BuiltinDType::F32, "f32");
  auto F64 = typed(BuiltinDType::F64, "f64");
  auto U8 = typed(BuiltinDType::U8, "u8");

  bool Ok = true;
  if (Blob.setAnnotation({2, 8}, F32) !=
          AnnotationStatus::StartsInsideElement ||
      Blob.setAnnotation({0, 6}, F32) != AnnotationStatus::EndsInsideElement) {
    std::cerr << "Did not reject a misaligned annotation for its rule\n";
    Ok = false;
  }
  Ok = Ok && expectAnnotations(Blob, {{{0, 128}, {}}}, "misaligned regions");

  Ok = Ok && Blob.setAnnotation({0, 128}, F64) == AnnotationStatus::Ok;
  if (Blob.setAnnotation({4, 4}, U8) != AnnotationStatus::StartsInsideElement ||
      Blob.setAnnotation({8, 4}, U8) != AnnotationStatus::EndsInsideElement) {
    std::cerr << "Did not reject a region that splits an F64 element\n";
    Ok = false;
  }
  Ok = Ok && expectAnnotations(Blob, {{{0, 128}, F64}},
                               "regions that split an F64 element");

  MnemeMemoryBlobDevice Odd(12, nullptr, 12);
  if (Odd.setAnnotation({0, 12}, F64) != AnnotationStatus::EndsInsideElement) {
    std::cerr << "Accepted an F64 annotation on a 12-byte blob\n";
    Ok = false;
  }
  Ok = Ok && expectAnnotations(Odd, {{{0, 12}, {}}},
                               "an F64 annotation on a 12-byte blob");
  return Ok;
}

// A loaded annotation list must divide the blob into valid subregions.
bool checkLoadedSubregions() {
  MnemeMemoryBlobDevice Blob(128, nullptr, 128);
  auto Md = typed(BuiltinDType::F32, "loaded");
  const std::vector<std::vector<MemoryAnnotation>> Invalid{
      {{{0, 32}, Md}, {{40, 88}, Md}},
      {{{0, 64}, Md}, {{32, 96}, Md}},
      {{{0, 64}, Md}, {{64, 32}, Md}},
  };
  bool Ok = true;
  for (const auto &Subregions : Invalid) {
    if (Blob.replaceAnnotations(Subregions)) {
      std::cerr << "Loaded an invalid annotation list\n";
      Ok = false;
    }
  }
  const std::vector<MemoryAnnotation> Valid{{{0, 64}, Md}, {{64, 64}, {}}};
  return Ok && Blob.replaceAnnotations(Valid) &&
         expectAnnotations(Blob, Valid, "loading valid subregions");
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
  Blob.writeWithAnnotations(Stream);

  auto Buff = const_cast<const char *>(Buffer.data());
  auto entry = MnemeMemoryBlobDevice::readWithAnnotations(Buff);
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
    if (!checkOverwrites() || !checkAlignment() || !checkLoadedSubregions())
      return -1;
    return 0;
  }();

  HData = nullptr;

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceFree(DData));
  if (EC)
    LOG_FATAL("Could not release device memory\n");

  return Ret;
}
