#include "MnemeAnnotationRuntime.hpp"

namespace mneme {

void annotate(const void *ptr, Metadata md) {
  detail::annotate_impl(ptr, std::move(md));
}

void annotate(const void *ptr, std::size_t bytes, Metadata md) {
  detail::annotate_region_impl(ptr, bytes, std::move(md));
}

namespace detail {

void annotate_impl(const void *ptr, Metadata md) {
  if (mneme_set_metadata_for_ptr)
    mneme_set_metadata_for_ptr(ptr, std::move(md));
}

void annotate_region_impl(const void *ptr, std::size_t bytes, Metadata md) {
  if (mneme_set_metadata_for_region)
    mneme_set_metadata_for_region(ptr, bytes, std::move(md));
}

bool get_annotation(const void *ptr, Metadata &md) {
  if (!mneme_get_metadata_for_ptr)
    return false;

  return mneme_get_metadata_for_ptr(ptr, &md);
}

void erase_annotation(const void *ptr) {
  if (mneme_erase_metadata_for_ptr)
    mneme_erase_metadata_for_ptr(ptr);
}

} // namespace detail
} // namespace mneme
