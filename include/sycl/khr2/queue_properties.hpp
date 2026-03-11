#ifndef SYCL_KHR2_QUEUE_PROPERTIES_HPP
#define SYCL_KHR2_QUEUE_PROPERTIES_HPP

#include "properties.hpp"

namespace sycl::khr2 {

// Stub queue tag (no real SYCL runtime).
struct queue_tag {};

// --- enable_profiling (runtime boolean property, self-keyed) ---

struct enable_profiling : detail::runtime_property<enable_profiling> {
  constexpr enable_profiling(bool v = true) : value{v} {}
  bool value;
};

template <>
struct is_property_key_for<enable_profiling, queue_tag> : std::true_type {};

// --- in_order (runtime boolean property, self-keyed) ---

struct in_order : detail::runtime_property<in_order> {
  constexpr in_order(bool v = true) : value{v} {}
  bool value;
};

template <>
struct is_property_key_for<in_order, queue_tag> : std::true_type {};

} // namespace sycl::khr2

#endif // SYCL_KHR2_QUEUE_PROPERTIES_HPP
