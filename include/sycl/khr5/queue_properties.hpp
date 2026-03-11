#ifndef SYCL_KHR5_QUEUE_PROPERTIES_HPP
#define SYCL_KHR5_QUEUE_PROPERTIES_HPP

#include "properties.hpp"

namespace sycl::khr5 {

struct queue_tag {};

struct enable_profiling : detail::runtime_property<enable_profiling> {
  constexpr enable_profiling(bool v = true) : value{v} {}
  bool value;
};

template <>
struct is_property_key_for<enable_profiling, queue_tag> : std::true_type {};

struct in_order : detail::runtime_property<in_order> {
  constexpr in_order(bool v = true) : value{v} {}
  bool value;
};

template <>
struct is_property_key_for<in_order, queue_tag> : std::true_type {};

} // namespace sycl::khr5

#endif // SYCL_KHR5_QUEUE_PROPERTIES_HPP
