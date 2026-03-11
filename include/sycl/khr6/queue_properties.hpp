#ifndef SYCL_KHR6_QUEUE_PROPERTIES_HPP
#define SYCL_KHR6_QUEUE_PROPERTIES_HPP

#include "properties.hpp"

namespace sycl::khr6 {

/// Stub queue tag (no real SYCL runtime).
struct queue_tag {};

/// Runtime boolean property: enable_profiling.
struct enable_profiling : detail::runtime_property<enable_profiling> {
  constexpr enable_profiling(bool v = true) : value{v} {}
  bool value;
};

template <>
struct is_property_key_for<enable_profiling, queue_tag> : std::true_type {};

/// Runtime boolean property: in_order.
struct in_order : detail::runtime_property<in_order> {
  constexpr in_order(bool v = true) : value{v} {}
  bool value;
};

template <>
struct is_property_key_for<in_order, queue_tag> : std::true_type {};

} // namespace sycl::khr6

#endif // SYCL_KHR6_QUEUE_PROPERTIES_HPP
