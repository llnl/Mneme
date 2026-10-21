#include "mneme/DeviceTraits.hpp"
#include "mneme/MnemeAnnotation.hpp"
#include "mneme/MnemeAnnotationInternal.hpp"
#include "mneme/MnemeKernelInfo.hpp"
#include "mneme/MnemeLogger.hpp"
#include "mneme/MnemeSnapshot.hpp"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <optional>
#include <random>

using namespace mneme;

#ifdef MNEME_ENABLE_HIP
constexpr DeviceVendors Vendor = DeviceVendors::HIP;
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
using MnemeMemoryBlobDevice = MnemeMemoryBlob<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
constexpr DeviceVendors Vendor = DeviceVendors::CUDA;
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
using MnemeMemoryBlobDevice = MnemeMemoryBlob<DeviceVendors::CUDA>;
#endif

template <typename T> void initializeRandomBuffer(T *Buffer, size_t Size) {
  // Random number generation setup
  std::mt19937 gen(4); // Mersenne Twister random number generator
  std::uniform_int_distribution<T> dis(
      0, 255); // Uniform distribution for char range

  // Fill the buffer with random values
  for (int I = 0; I < Size; I++) {
    Buffer[I] = dis(gen);
  }
}

int main(int argc, char **argv) {
  // We allocate some "fake" globals
  auto initializeDeviceData = [&] {
    uint8_t *HData = new uint8_t[128];
    initializeRandomBuffer(HData, 128);
    uint8_t *DData;

    auto EC = MnemeDeviceRT::DeviceErrorCheck(
        MnemeDeviceRT::DeviceMalloc((void **)&DData, 128));
    if (EC) {
      std::cout << " Here " << EC.value() << "\n";
      LOG_FATAL("Could not allocate device data");
    }

    EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
        DData, HData, 128, MnemeDeviceRT::MemcpyHostToDeviceKind()));
    if (EC) {
      std::cout << " Here " << EC.value() << "\n";
      LOG_FATAL("Could not allocate device data");
    }
    return std::make_pair(DData, HData);
  };

  auto BlobData = initializeDeviceData();
  auto GlobalData = initializeDeviceData();

  MnemeMemoryBlobDevice Blob(128L, BlobData.first, 128L);
  mneme::Metadata Md;
  Md.builtin = BuiltinDType::F64;
  Md.norm = Norm::L2;
  Md.threshold = 0.5;
  Md.threshold_kind = ThresholdKind::Relative;
  Md.tag = std::string("Test");
  Blob.setMetadata(Md);

  Blob.setHostData(std::unique_ptr<uint8_t[]>(new uint8_t[128]));

  proteus::runtime::GlobalMetadata GV{GlobalData.second, GlobalData.first, 128};

  std::string KernelName("TestKernel");
  std::shared_ptr<KernelInfo> TestKernel =
      std::make_shared<KernelInfo>(KernelName);

  int NumArgs = 4;
  llvm::SmallVector<size_t> ArgSizes(4);
  initializeRandomBuffer(ArgSizes.data(), 4);
  void **Args = new void *[NumArgs];
  for (auto A = 0; A < NumArgs; A++) {
    Args[A] = reinterpret_cast<void *>(new uint8_t[ArgSizes[A]]);
  }

  TestKernel->setArgSizes(ArgSizes);

  // Create a raw_svector_ostream using the buffer
  proteus::runtime::GlobalMetadataMap GVars;
  GVars.try_emplace("Test", GV);
  llvm::DenseMap<void *, MnemeMemoryBlobDevice> DeviceMemMap;
  DeviceMemMap.try_emplace((void *)BlobData.first, std::move(Blob));
  std::filesystem::path SnapshotFN("./test.mneme");

  auto PrologueGlobals = std::make_shared<GlobalSnapshotData>();
  SnapshotInput<Vendor> In{GVars, DeviceMemMap, TestKernel->KernelArgSizes,
                           Args, nullptr};
  BytesWriter<Vendor> PrologueWriter(PrologueGlobals);
  size_t MeasuredBytes = PrologueWriter.measure(In);
  PrologueWriter.write(SnapshotFN, In);

  auto ReadSnap = SnapshotFormatRegistry<Vendor>::open(SnapshotFN.string())
                      ->read(KernelName, BaseSnapshotSource<Vendor>());
  auto &ReadGVars = ReadSnap.GlobalVars;
  auto &ReadDeviceMemMap = ReadSnap.DeviceMemory;
  auto &RTestKernel = ReadSnap.KInfo;

  auto ValidateGlobalMem = [&]() {
    auto it = ReadGVars.find("Test");
    if (it == ReadGVars.end())
      return 2;
    auto &RGV = it->second;

    if (RGV.VarSize != GV.VarSize) {
      std::cerr << "VarSize differs " << RGV.VarSize << " " << GV.VarSize
                << "\n";
      return 2;
    }

    if (std::memcmp(GV.HostAddr, RGV.HostAddr, 128) != 0) {
      std::cerr << "Memory differs between GV and GVR\n";
      return 2;
    }
    std::cerr << "Global Memory is correct\n";
    return 0;
  }();

  auto ValidateDeviceMem = [&]() {
    for (auto &RKV : ReadDeviceMemMap) {
      auto &RBlob = RKV.second;
      if (!DeviceMemMap.contains(RKV.first)) {
        std::cerr << "Address does not exist in Device Map " << std::hex
                  << RKV.first << std::dec << "\n";
        return 1;
      }

      auto &WBlob = DeviceMemMap[RKV.first];

      if (RBlob.getActualSize() != WBlob.getActualSize()) {
        std::cerr << "Actual Sizes differ " << RBlob.getActualSize() << " "
                  << WBlob.getActualSize() << "\n";
        return 1;
      }

      if (RBlob.getSize() != WBlob.getSize()) {
        std::cerr << "Sizes differ" << WBlob.getSize() << " " << RBlob.getSize()
                  << "\n";
        return 1;
      }

      if (!metadata::isDefault(RBlob.getMetadata())) {
        std::cerr << "Snapshot stored blob metadata\n";
        return 1;
      }

      uint8_t *WData = WBlob.getHostData().get();
      uint8_t *RData = RBlob.getHostData().get();
      if (std::memcmp(reinterpret_cast<void *>(WData),
                      reinterpret_cast<void *>(RData), 128) != 0) {
        std::cerr << "Memory differs between GV and GVR\n";
        return 1;
      }
    }
    return 0;
  }();

  auto ValidateKernelArgs = [&]() {
    auto &WKernel = *TestKernel;
    auto &RKernel = *RTestKernel;
    if (WKernel.getNumArgs() != RKernel.getNumArgs()) {
      std::cerr << "Number of recorded arguments differ "
                << WKernel.getNumArgs() << " and read " << RKernel.getNumArgs()
                << "\n";
      return 4;
    }

    auto WArgSizes = WKernel.getArgSizes();
    auto RArgSizes = RKernel.getArgSizes();
    for (auto A = 0; A < WKernel.getNumArgs(); A++) {
      if (WArgSizes[A] != RArgSizes[A]) {
        std::cerr << "The size of argument " << A
                  << " differs WAS:" << WArgSizes[A] << " RAS:" << RArgSizes[A]
                  << "\n";
        return 4;
      }
    }

    auto WArgData = WKernel.getArgData();
    auto RArgData = RKernel.getArgData();
    for (auto A = 0; A < WKernel.getNumArgs(); A++) {
      if (std::memcmp(Args[A], RArgData[A].get(), WArgSizes[A]) != 0) {
        if (WArgSizes[A] != RArgSizes[A]) {
          std::cerr << "The Memory of argument " << A << "differs \n";
          return 4;
        }
      }
    }
    return 0;
  }();

  // make some changes to the device memory and global memory to test diffing
  BlobData.second[0] ^= 0x7;
  BlobData.second[63] ^= 0x11;
  BlobData.second[64] ^= 0x23;
  BlobData.second[127] ^= 0x42;
  GlobalData.second[2] ^= 0x5;
  GlobalData.second[3] ^= 0x9;
  GlobalData.second[4] ^= 0x13;

  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
      BlobData.first, BlobData.second, 128,
      MnemeDeviceRT::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Could not update device blob data");

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceCopy(
      GlobalData.first, GlobalData.second, 128,
      MnemeDeviceRT::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Could not update device global data");

  std::filesystem::path DiffSnapshotFN("./test.epilogue.mneme");
  DiffWriter<Vendor> DiffSnapshotWriter(PrologueGlobals);
  size_t MeasuredDiff = DiffSnapshotWriter.measure(In);
  DiffSnapshotWriter.write(DiffSnapshotFN, In);

  auto DiffSnap =
      SnapshotFormatRegistry<Vendor>::open(DiffSnapshotFN.string())
          ->read(KernelName, BaseSnapshotSource<Vendor>(SnapshotFN.string()));
  auto &DiffGVars = DiffSnap.GlobalVars;
  auto &DiffDeviceMemMap = DiffSnap.DeviceMemory;
  auto &DiffKernel = DiffSnap.KInfo;

  auto ValidateDiffGlobalMem = [&]() {
    auto it = DiffGVars.find("Test");
    if (it == DiffGVars.end())
      return 8;
    auto &RGV = it->second;

    if (RGV.VarSize != GV.VarSize) {
      std::cerr << "Diff VarSize differs " << RGV.VarSize << " " << GV.VarSize
                << "\n";
      return 8;
    }

    if (std::memcmp(GlobalData.second, RGV.HostAddr, 128) != 0) {
      std::cerr << "Diff global memory did not reconstruct epilogue data\n";
      return 8;
    }
    return 0;
  }();

  auto ValidateDiffDeviceMem = [&]() {
    auto it = DiffDeviceMemMap.find((void *)BlobData.first);
    if (it == DiffDeviceMemMap.end()) {
      std::cerr << "Diff device map is missing blob address\n";
      return 16;
    }

    auto &RBlob = it->second;
    if (RBlob.getActualSize() != 128 || RBlob.getSize() != 128) {
      std::cerr << "Diff blob sizes differ\n";
      return 16;
    }

    if (!metadata::isDefault(RBlob.getMetadata())) {
      std::cerr << "Diff snapshot stored blob metadata\n";
      return 16;
    }

    if (std::memcmp(BlobData.second, RBlob.getHostData().get(), 128) != 0) {
      std::cerr << "Diff device memory did not reconstruct epilogue data\n";
      return 16;
    }
    return 0;
  }();

  auto ValidateDiffKernelArgs = [&]() {
    if (TestKernel->getNumArgs() != DiffKernel->getNumArgs()) {
      std::cerr << "Diff snapshot did not inherit prologue arguments\n";
      return 32;
    }

    auto WArgSizes = TestKernel->getArgSizes();
    auto RArgSizes = DiffKernel->getArgSizes();
    auto RArgData = DiffKernel->getArgData();
    for (auto A = 0; A < TestKernel->getNumArgs(); A++) {
      if (WArgSizes[A] != RArgSizes[A] ||
          std::memcmp(Args[A], RArgData[A].get(), WArgSizes[A]) != 0) {
        std::cerr << "Diff snapshot argument " << A
                  << " differs from prologue\n";
        return 32;
      }
    }
    return 0;
  }();

  auto ResetBlobBase = [&]() {
    auto PrologueBlobIt = ReadDeviceMemMap.find((void *)BlobData.first);
    auto HostData = std::unique_ptr<uint8_t[]>(new uint8_t[128]);
    std::memcpy(HostData.get(), PrologueBlobIt->second.getHostData().get(),
                128);
    DeviceMemMap[(void *)BlobData.first].setHostData(std::move(HostData));
  };

  llvm::SmallVector<size_t> EmptyArgSizes;
  SnapshotInput<Vendor> InNoArgs{GVars, DeviceMemMap, EmptyArgSizes, nullptr,
                                 nullptr};

  ResetBlobBase();
  std::filesystem::path SparseBestSnapshotFN("./test.best.sparse.mneme");
  BestWriter<Vendor>(PrologueGlobals).write(SparseBestSnapshotFN, InNoArgs);
  auto ValidateBestSparse = [&]() {
    if (!SnapshotFormatRegistry<Vendor>::open(SparseBestSnapshotFN.string())
             ->requiresBaseSnapshot()) {
      std::cerr << "Best sparse snapshot should choose diff\n";
      return 64;
    }
    auto BestSparseSnap =
        SnapshotFormatRegistry<Vendor>::open(SparseBestSnapshotFN.string())
            ->read(KernelName, BaseSnapshotSource<Vendor>(SnapshotFN.string()));
    auto It = BestSparseSnap.DeviceMemory.find((void *)BlobData.first);
    if (It == BestSparseSnap.DeviceMemory.end() ||
        std::memcmp(BlobData.second, It->second.getHostData().get(), 128) !=
            0) {
      std::cerr << "Best sparse snapshot did not reconstruct epilogue data\n";
      return 64;
    }
    return 0;
  }();

  auto PrologueBlobIt = ReadDeviceMemMap.find((void *)BlobData.first);
  auto PrologueGlobalIt = ReadGVars.find("Test");
  auto *PrologueBlob = PrologueBlobIt->second.getHostData().get();
  auto *PrologueGlobal =
      static_cast<uint8_t *>(PrologueGlobalIt->second.HostAddr);
  // Alternating changes force many one-byte diff ranges, making the bytes
  // snapshot smaller than the diff snapshot.
  for (size_t I = 0; I < 128; ++I) {
    BlobData.second[I] = (I % 2 == 0) ? (PrologueBlob[I] ^ 0xff)
                                      : PrologueBlob[I];
    GlobalData.second[I] = (I % 2 == 0) ? (PrologueGlobal[I] ^ 0xff)
                                        : PrologueGlobal[I];
  }

  EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceCopy(BlobData.first, BlobData.second, 128,
                                MnemeDeviceRT::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Could not update fragmented device blob data");

  EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceCopy(GlobalData.first, GlobalData.second, 128,
                                MnemeDeviceRT::MemcpyHostToDeviceKind()));
  if (EC)
    LOG_FATAL("Could not update fragmented device global data");

  ResetBlobBase();
  std::filesystem::path FragmentedBestSnapshotFN(
      "./test.best.fragmented.mneme");
  BestWriter<Vendor>(PrologueGlobals).write(FragmentedBestSnapshotFN, InNoArgs);
  auto ValidateBestFragmented = [&]() {
    if (SnapshotFormatRegistry<Vendor>::open(FragmentedBestSnapshotFN.string())
            ->requiresBaseSnapshot()) {
      std::cerr << "Best fragmented snapshot should choose bytes\n";
      return 128;
    }
    auto BestFragmentedSnap =
        SnapshotFormatRegistry<Vendor>::open(FragmentedBestSnapshotFN.string())
            ->read(KernelName, BaseSnapshotSource<Vendor>(SnapshotFN.string()));
    auto It = BestFragmentedSnap.DeviceMemory.find((void *)BlobData.first);
    if (It == BestFragmentedSnap.DeviceMemory.end() ||
        std::memcmp(BlobData.second, It->second.getHostData().get(), 128) !=
            0) {
      std::cerr
          << "Best fragmented snapshot did not reconstruct epilogue data\n";
      return 128;
    }
    return 0;
  }();

  // "Best" mode decides on measure(), so it must match the written size.
  auto ValidateMeasure = [&]() {
    auto ActualBytes = std::filesystem::file_size(SnapshotFN);
    if (MeasuredBytes != ActualBytes) {
      std::cerr << "Bytes snapshot measured " << MeasuredBytes << " but wrote "
                << ActualBytes << "\n";
      return 512;
    }

    auto ActualDiff = std::filesystem::file_size(DiffSnapshotFN);
    if (MeasuredDiff != ActualDiff) {
      std::cerr << "Diff snapshot measured " << MeasuredDiff << " but wrote "
                << ActualDiff << "\n";
      return 512;
    }
    return 0;
  }();

  // Pin all three prefixes SnapshotHeader::parse recognises.
  auto ValidateHeaderParse = [&]() {
    char Container[SnapshotHeader::Size];
    std::memcpy(Container, SnapshotHeader::Magic,
                sizeof(SnapshotHeader::Magic));
    uint32_t Kind = 2;
    uint32_t Version = 7;
    std::memcpy(Container + 8, &Kind, sizeof(Kind));
    std::memcpy(Container + 12, &Version, sizeof(Version));
    auto [ContainerHeader, ContainerOffset] =
        SnapshotHeader::parse(llvm::StringRef(Container, sizeof(Container)));
    if (ContainerHeader.Kind != SnapshotKind::Diff ||
        ContainerHeader.Version != 7 || ContainerOffset != 16) {
      std::cerr << "Container prefix did not parse as {Diff, 7, 16}\n";
      return 256;
    }

    std::string Legacy("MNEME_DIFF_V1");
    Legacy += "arbitrary diff payload";
    auto [LegacyHeader, LegacyOffset] = SnapshotHeader::parse(Legacy);
    if (LegacyHeader.Kind != SnapshotKind::Diff || LegacyHeader.Version != 1 ||
        LegacyOffset != 13) {
      std::cerr << "Legacy diff magic did not parse as {Diff, 1, 13}\n";
      return 256;
    }

    std::string Headerless("arbitrary bytes snapshot payload");
    auto [BytesHeader, BytesOffset] = SnapshotHeader::parse(Headerless);
    if (BytesHeader.Kind != SnapshotKind::Bytes || BytesHeader.Version != 0 ||
        BytesOffset != 0) {
      std::cerr << "Unprefixed buffer did not parse as {Bytes, 0, 0}\n";
      return 256;
    }

    auto parseFilePrefix = [](const std::filesystem::path &Path) {
      std::ifstream FileIn(Path, std::ios::binary);
      std::string Prefix(SnapshotHeader::Size, '\0');
      FileIn.read(Prefix.data(), Prefix.size());
      return SnapshotHeader::parse(Prefix);
    };

    auto [PrologueHeader, PrologueOffset] = parseFilePrefix(SnapshotFN);
    if (PrologueHeader.Kind != SnapshotKind::Bytes ||
        PrologueHeader.Version != 1 || PrologueOffset != 16) {
      std::cerr << "Prologue file did not parse as {Bytes, 1, 16}\n";
      return 256;
    }

    auto [DiffFileHeader, DiffFileOffset] = parseFilePrefix(DiffSnapshotFN);
    if (DiffFileHeader.Kind != SnapshotKind::Diff ||
        DiffFileHeader.Version != 2 || DiffFileOffset != 16) {
      std::cerr << "Diff file did not parse as {Diff, 2, 16}\n";
      return 256;
    }
    return 0;
  }();

  // Older layouts follow each blob with its metadata, which replay of old
  // recordings still compares with. A diff carries the epilogue's metadata.
  auto ValidateLegacyMetadata = [&]() {
    auto writeMetadata = [](llvm::raw_ostream &OS, const Metadata &M) {
      util::writeScalar(OS, M.builtin);
      util::writeScalar(OS, M.threshold);
      util::writeScalar(OS, M.threshold_kind);
      util::writeScalar(OS, M.norm);
      util::writeScalar(OS, M.tag->size());
      util::writeBytes(OS, llvm::StringRef(*M.tag));
    };
    // Two blobs and an argument, so a reader that skips a blob's metadata
    // record misaligns.
    void *Addr = reinterpret_cast<void *>(0x1000);
    void *Addr2 = reinterpret_cast<void *>(0x2000);
    uint8_t Data[16] = {};
    uint8_t Data2[8] = {};
    uint32_t Arg = 0xdeadbeef;
    mneme::Metadata Md2 = Md;
    Md2.threshold = 0.75;
    Md2.tag = std::string("Second");
    mneme::Metadata EpiMd = Md;
    EpiMd.threshold = 0.25;
    mneme::Metadata EpiMd2 = Md2;
    EpiMd2.threshold = 0.125;

    std::filesystem::path LegacyBytesFN("./test.legacy.bytes.mneme");
    std::filesystem::path LegacyDiffFN("./test.legacy.diff.mneme");
    {
      std::error_code FileEC;
      llvm::raw_fd_ostream OS(LegacyBytesFN.string(), FileEC);
      util::writeScalar(OS, size_t{0});
      util::writeScalar(OS, size_t{2});
      BlobHeader{sizeof(Data), sizeof(Data), Addr}.write(OS);
      util::writeBytes(OS, llvm::ArrayRef<uint8_t>(Data));
      writeMetadata(OS, Md);
      BlobHeader{sizeof(Data2), sizeof(Data2), Addr2}.write(OS);
      util::writeBytes(OS, llvm::ArrayRef<uint8_t>(Data2));
      writeMetadata(OS, Md2);
      util::writeScalar(OS, size_t{1});
      util::writeScalar(OS, sizeof(Arg));
      util::writeScalar(OS, Arg);
    }
    {
      std::error_code FileEC;
      llvm::raw_fd_ostream OS(LegacyDiffFN.string(), FileEC);
      OS << "MNEME_DIFF_V1";
      util::writeScalar(OS, size_t{0});
      util::writeScalar(OS, size_t{2});
      BlobHeader{sizeof(Data), sizeof(Data), Addr}.write(OS);
      writeMetadata(OS, EpiMd);
      util::writeScalar(OS, size_t{1});
      util::writeScalar(OS, size_t{3});
      util::writeScalar(OS, size_t{1});
      util::writeScalar(OS, uint8_t{0xab});
      BlobHeader{sizeof(Data2), sizeof(Data2), Addr2}.write(OS);
      writeMetadata(OS, EpiMd2);
      util::writeScalar(OS, size_t{1});
      util::writeScalar(OS, size_t{7});
      util::writeScalar(OS, size_t{1});
      util::writeScalar(OS, uint8_t{0xcd});
    }

    auto Bytes = SnapshotFormatRegistry<Vendor>::open(LegacyBytesFN.string())
                     ->read(KernelName, BaseSnapshotSource<Vendor>());
    auto Diff = SnapshotFormatRegistry<Vendor>::open(LegacyDiffFN.string())
                    ->read(KernelName,
                           BaseSnapshotSource<Vendor>(LegacyBytesFN.string()));
    auto BytesIt = Bytes.DeviceMemory.find(Addr);
    auto BytesIt2 = Bytes.DeviceMemory.find(Addr2);
    auto DiffIt = Diff.DeviceMemory.find(Addr);
    auto DiffIt2 = Diff.DeviceMemory.find(Addr2);
    if (BytesIt == Bytes.DeviceMemory.end() ||
        BytesIt2 == Bytes.DeviceMemory.end() ||
        DiffIt == Diff.DeviceMemory.end() ||
        DiffIt2 == Diff.DeviceMemory.end()) {
      std::cerr << "Legacy snapshot is missing a blob\n";
      return 1024;
    }

    const auto &BytesMd = BytesIt->second.getMetadata();
    const auto &BytesMd2 = BytesIt2->second.getMetadata();
    const auto &DiffMd = DiffIt->second.getMetadata();
    const auto &DiffMd2 = DiffIt2->second.getMetadata();
    if (BytesMd.builtin != BuiltinDType::F64 || BytesMd.norm != Norm::L2 ||
        BytesMd.threshold != 0.5 ||
        BytesMd.threshold_kind != ThresholdKind::Relative ||
        BytesMd.tag != std::optional<std::string>("Test") ||
        BytesMd2.threshold != 0.75 ||
        BytesMd2.tag != std::optional<std::string>("Second") ||
        DiffMd.threshold != 0.25 || DiffMd2.threshold != 0.125) {
      std::cerr << "Legacy blob metadata differs\n";
      return 1024;
    }

    const auto &KInfo = *Bytes.KInfo;
    if (KInfo.KernelArgSizes.size() != 1 ||
        KInfo.KernelArgSizes[0] != sizeof(Arg) ||
        std::memcmp(KInfo.ArgData[0].get(), &Arg, sizeof(Arg)) != 0) {
      std::cerr << "Legacy snapshot arguments differ\n";
      return 1024;
    }

    Data[3] = 0xab;
    Data2[7] = 0xcd;
    if (std::memcmp(Data, DiffIt->second.getHostData().get(), sizeof(Data)) !=
            0 ||
        std::memcmp(Data2, DiffIt2->second.getHostData().get(),
                    sizeof(Data2)) != 0) {
      std::cerr << "Legacy diff did not reconstruct epilogue data\n";
      return 1024;
    }
    return 0;
  }();

  auto Ret = ValidateGlobalMem | ValidateDeviceMem | ValidateKernelArgs |
             ValidateDiffGlobalMem | ValidateDiffDeviceMem |
             ValidateDiffKernelArgs | ValidateBestSparse |
             ValidateBestFragmented | ValidateMeasure | ValidateHeaderParse |
             ValidateLegacyMetadata;

  delete[] GlobalData.second;
  delete[] BlobData.second;

  EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceFree(GlobalData.first));
  if (EC)
    LOG_FATAL("Could not release device memory\n");

  EC = MnemeDeviceRT::DeviceErrorCheck(
      MnemeDeviceRT::DeviceFree(BlobData.first));
  if (EC)
    LOG_FATAL("Could not release device memory\n");

  return Ret;
}
