#ifndef SYCL_KHR5_DETAIL_PROPERTY_BASES_HPP
#define SYCL_KHR5_DETAIL_PROPERTY_BASES_HPP

/// v5 design: parameter pack for compile-time properties.
///
///   template<int... Values> struct prop2;
///   template<> struct prop2<> { ... };       // empty pack = key
///   template<int V> struct prop2<V> { ... }; // single value = property
///
/// No wrappers, no booleans, no sentinels.  Pure C++ template specialization.
/// The key is prop2<> — structurally distinct from prop2<0>.
///
/// Same three tag bases as v2.  Same __detail_key_t convention.

namespace sycl::khr5::detail {

struct runtime_property_base {};
struct constant_property_base {};
struct hybrid_property_base {};

template <typename Derived>
struct runtime_property : runtime_property_base {
  using __detail_key_t = Derived;
};

} // namespace sycl::khr5::detail

#endif // SYCL_KHR5_DETAIL_PROPERTY_BASES_HPP
