#ifndef SYCL_KHR2_DETAIL_PROPERTY_BASES_HPP
#define SYCL_KHR2_DETAIL_PROPERTY_BASES_HPP

/// Alternative design: no separate key types.
///
/// The key for a compile-time or hybrid property is the default instantiation
/// of the property template itself (e.g. prop2<> instead of prop2_key).
/// This eliminates constant_property_key_base and hybrid_property_key_base.
///
/// Only three tag bases remain:
///   runtime_property_base   -- runtime-only property (self-keyed by type)
///   constant_property_base  -- compile-time property (keyed by default instantiation)
///   hybrid_property_base    -- mixed property (keyed by default instantiation)

namespace sycl::khr2::detail {

// Tag bases for trait detection via std::is_base_of_v.

struct runtime_property_base {};
struct constant_property_base {};
struct hybrid_property_base {};

// Convenience CRTP base for a runtime property (is its own key).
template <typename Derived>
struct runtime_property : runtime_property_base {
  using __detail_key_t = Derived;
};

} // namespace sycl::khr2::detail

#endif // SYCL_KHR2_DETAIL_PROPERTY_BASES_HPP
