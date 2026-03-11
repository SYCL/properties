#ifndef SYCL_KHR4_DETAIL_PROPERTY_BASES_HPP
#define SYCL_KHR4_DETAIL_PROPERTY_BASES_HPP

/// v4 design: use ct_value<T> as the NTTP wrapper for compile-time properties.
///
/// ct_value<T> has two states:
///   - sentinel (default-constructed): no value, used as the key form
///   - valued (constructed from T): holds a real value
///
/// This means prop<> (sentinel) is ALWAYS distinct from prop<0> (value 0),
/// eliminating the sentinel-reservation caveat of v2.
///
/// Thanks to ct_value<T>'s implicit constructor from T, users write
/// prop<42> instead of prop<ct_value<int>{42}>.

namespace sycl::khr4::detail {

// Wrapper for compile-time property values used as NTTPs.
// Default-constructed = sentinel (key form), constructed from T = real value.
// Structural type (all members public, literal) so it can be used as NTTP.
template <typename T>
struct ct_value {
  T value{};
  bool has_value{false};

  constexpr ct_value() = default;
  constexpr ct_value(T v) : value{v}, has_value{true} {}
  constexpr operator T() const { return value; }
};

// Tag bases for trait detection.
struct runtime_property_base {};
struct constant_property_base {};
struct hybrid_property_base {};

// Convenience CRTP base for a runtime property (is its own key).
template <typename Derived>
struct runtime_property : runtime_property_base {
  using __detail_key_t = Derived;
};

} // namespace sycl::khr4::detail

#endif // SYCL_KHR4_DETAIL_PROPERTY_BASES_HPP
