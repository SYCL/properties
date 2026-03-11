#ifndef SYCL_KHR6_DETAIL_PROPERTY_BASES_HPP
#define SYCL_KHR6_DETAIL_PROPERTY_BASES_HPP

#include <type_traits>

/// v6 design: auto NTTP with a no_value sentinel TYPE.
///
///   struct no_value {};
///   template<auto V = no_value{}>
///   struct prop : constant_property_base { ... };
///
/// - prop<42>: V = int{42}, a property with value 42
/// - prop<0>:  V = int{0}, a property with value 0 (NOT reserved!)
/// - prop<>:   V = no_value{}, the key (sentinel is a different TYPE)
///
/// has_value is static constexpr, derived from decltype(V).
/// No wrapper, no runtime boolean, no reserved values.

namespace sycl::khr6::detail {

/// Sentinel type: default NTTP value for compile-time properties.
/// A property instantiated with no_value{} is the key form.
struct no_value {};

/// Tag bases for trait detection.
struct runtime_property_base {};
struct constant_property_base {};
struct hybrid_property_base {};

/// CRTP base for runtime properties (self-keyed).
template <typename Derived>
struct runtime_property : runtime_property_base {
  using __detail_key_t = Derived;
};

/// Is V the sentinel?
template <auto V>
inline constexpr bool is_sentinel_v =
    std::is_same_v<decltype(V), no_value>;

} // namespace sycl::khr6::detail

#endif // SYCL_KHR6_DETAIL_PROPERTY_BASES_HPP
