#pragma once

#include <mneme/MnemeReplay.hpp>

namespace mneme::python {

template <DeviceVendors V> class StateLoader {
  using RT = DeviceTraits<V>;
  using Blob = MnemeMemoryBlob<V>;
  using Registry = SnapshotFormatRegistry<V>;

  const size_t BufferSize;
  void *Staging = nullptr;

  static void check(typename RT::DeviceError_t Status) {
    if (auto Error = RT::DeviceErrorCheck(Status))
      LOG_FATAL("Loading snapshot: {}", *Error);
  }

  void upload(void *Destination, llvm::StringRef Bytes) {
    if (!Staging)
      check(RT::mallocPinned(&Staging, BufferSize));

    // stream each chunk into the pinned buffer
    auto *Dst = static_cast<uint8_t *>(Destination);
    while (!Bytes.empty()) {
      auto Chunk = Bytes.take_front(BufferSize);
      std::memcpy(Staging, Chunk.data(), Chunk.size());
      check(RT::DeviceCopy(Dst, Staging, Chunk.size(),
                           RT::MemcpyHostToDeviceKind()));
      Dst += Chunk.size();
      Bytes = Bytes.drop_front(Chunk.size());
    }
  }

  static void copyDevice(void *Dst, void *Src, size_t Size) {
    if (Size)
      check(RT::DeviceCopy(Dst, Src, Size, RT::MemcpyDeviceToDeviceKind()));
  }

  // A full snapshot allocates each destination once. A diff visits its
  // changed ranges in file order and writes into existing destinations.
  void readDevice(const SnapshotReader<V> &Reader, Snapshot<V> &Snap) {
    Reader.readInto(
        Snap,
        [&](BlobHeader H) -> Blob & {
          return Snap.DeviceMemory
              .try_emplace(H.DevAddr, H.ActualSize, nullptr, H.Size,
                           MemoryLocation::Device)
              .first->second;
        },
        [&](Blob &B, size_t Offset, llvm::StringRef Data) {
          upload(static_cast<uint8_t *>(B.getDeviceData()) + Offset, Data);
        });
  }

  static Snapshot<V> clone(const PrologueState<V> &Pro) {
    Snapshot<V> Snap;
    const auto &K = Pro.kernelInfo();
    Snap.KInfo = std::make_shared<KernelInfo>(K.getName());
    Snap.KInfo->KernelArgSizes = K.KernelArgSizes;
    Snap.KInfo->ArgData.resize(K.ArgData.size());

    // copy arg data into clone
    for (size_t I = 0; I < K.ArgData.size(); ++I) {
      const char *P = reinterpret_cast<const char *>(K.ArgData[I].get());
      Snap.KInfo->setArgData(P, I);
    }

    // copy globals into clone
    for (const auto &[Name, G] : Pro.getGlobalVars()) {
      auto &Copy =
          Snap.GlobalVars.try_emplace(Name, G.DevAddr, G.VarSize).first->second;
      std::memcpy(Copy.HostAddr, G.HostAddr, G.VarSize);
    }

    // copy data d-to-d into clone
    for (const auto &[Addr, B] : Pro.getDeviceMemory()) {
      auto &Copy = Snap.DeviceMemory
                       .try_emplace(Addr, B.getActualSize(), nullptr,
                                    B.getSize(), MemoryLocation::Device)
                       .first->second;
      copyDevice(Copy.getDeviceData(), B.getDeviceData(), B.getSize());
    }

    return Snap;
  }

public:
  explicit StateLoader(size_t BufferSize) : BufferSize(BufferSize) {
    if (!BufferSize)
      LOG_FATAL("Snapshot buffer size must be positive");
  }

  ~StateLoader() {
    if (Staging)
      (void)RT::freePinned(Staging);
  }

  StateLoader(const StateLoader &) = delete;
  StateLoader &operator=(const StateLoader &) = delete;

  std::unique_ptr<PrologueState<V>> prologue(const char *Name,
                                             const char *File) {
    auto Reader = Registry::open(File);
    if (Reader->requiresBaseSnapshot())
      LOG_FATAL("A diff cannot serve as a prologue");

    Snapshot<V> Snap;
    Snap.KInfo = std::make_shared<KernelInfo>(Name);

    // Keep one view per allocation until every recorded VA is reserved.
    // Device backing and pinned staging may then be allocated anywhere else.
    std::vector<std::pair<Blob *, llvm::StringRef>> Payloads;
    Reader->readInto(
        Snap,
        [&](BlobHeader H) -> Blob & {
          return Snap.DeviceMemory.try_emplace(H.DevAddr, H).first->second;
        },
        [&](Blob &B, size_t, llvm::StringRef Data) {
          Payloads.emplace_back(&B, Data);
        });
    
    auto Pro = std::make_unique<PrologueState<V>>(std::move(Snap));
    Pro->mapRecordedAddresses();
    for (auto &[B, Data] : Payloads) {
      B->allocateDeviceData();
      upload(B->getDeviceData(), Data);
      copyDevice(B->getBlobAddr(), B->getDeviceData(), Data.size());
    }

    return Pro;
  }

  std::unique_ptr<EpilogueState<V>> epilogue(const char *Name, const char *File,
                                             const char *BaseFile,
                                             const PrologueState<V> *Pro) {
    auto Reader = Registry::open(File);
    Snapshot<V> Snap;
    if (Reader->requiresBaseSnapshot() && Pro) {
      // clone the prologue d-to-d to be faster
      // then construct diff on device
      Snap = clone(*Pro);
    } else {
      Snap.KInfo = std::make_shared<KernelInfo>(Name);
      if (Reader->requiresBaseSnapshot()) {
        if (!BaseFile)
          LOG_FATAL("A diff epilogue requires a base prologue");

        auto Base = Registry::open(BaseFile);
        if (Base->requiresBaseSnapshot())
          LOG_FATAL("A diff cannot serve as a prologue");
        readDevice(*Base, Snap);
      }
    }

    readDevice(*Reader, Snap);
    return std::make_unique<EpilogueState<V>>(std::move(Snap));
  }

  // CUDA can return from a device-to-device copy before it completes.
  void finish() const { check(RT::DeviceStreamSynchronize(nullptr)); }
};
} // namespace mneme::python
