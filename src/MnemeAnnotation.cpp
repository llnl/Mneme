#include "MnemeAnnotationRuntime.hpp"
#include "mneme/MnemeLogger.hpp"

namespace mneme {

void annotate(const void *ptr, Metadata md) {
  detail::annotate_impl(ptr, std::move(md));
}

void annotate(const void *ptr, std::size_t bytes, Metadata md) {
  detail::annotate_region_impl(ptr, bytes, std::move(md));
}

namespace detail {

void annotate_impl(const void *ptr, Metadata md) {
  if (!ptr) {
    LOG_WARN("annotate called with null pointer");
    return;
  }

  LOG_DEBUG(
      "request ptr={} builtin={} threshold={} threshold_kind={} norm={} tag={}",
      ptr, static_cast<unsigned>(md.builtin), md.threshold,
      static_cast<unsigned>(md.threshold_kind), static_cast<unsigned>(md.norm),
      md.tag.value_or("no_tag"));

  if (mneme_set_metadata_for_ptr) {
    mneme_set_metadata_for_ptr(ptr, std::move(md));
    return;
  }

  LOG_WARN("no recorder hook installed for ptr={} tag={}; annotation ignored",
           ptr, md.tag.value_or("no_tag"));
}

void annotate_region_impl(const void *ptr, std::size_t bytes, Metadata md) {
  if (!ptr) {
    LOG_WARN("region annotate called with null pointer");
    return;
  }

  if (bytes == 0) {
    LOG_WARN("region annotate called with zero bytes ptr={} tag={}", ptr,
             md.tag.value_or("no_tag"));
    return;
  }

  LOG_DEBUG("region request ptr={} bytes={} builtin={} threshold={} "
            "threshold_kind={} norm={} tag={}",
            ptr, bytes, static_cast<unsigned>(md.builtin), md.threshold,
            static_cast<unsigned>(md.threshold_kind),
            static_cast<unsigned>(md.norm), md.tag.value_or("no_tag"));

  if (mneme_set_metadata_for_region) {
    mneme_set_metadata_for_region(ptr, bytes, std::move(md));
    return;
  }

  LOG_WARN("no recorder region hook installed for ptr={} bytes={} tag={}; "
           "annotation ignored",
           ptr, bytes, md.tag.value_or("no_tag"));
}

bool get_annotation(const void *ptr, Metadata &md) {
  if (!ptr)
    return false;

  if (!mneme_get_metadata_for_ptr)
    return false;

  return mneme_get_metadata_for_ptr(ptr, &md);
}

void erase_annotation(const void *ptr) {
  if (!ptr)
    return;

  if (mneme_erase_metadata_for_ptr)
    mneme_erase_metadata_for_ptr(ptr);
}

} // namespace detail
} // namespace mneme
