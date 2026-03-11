#ifndef SYCL_KHR3_DETAIL_PROPERTY_BASES_HPP
#define SYCL_KHR3_DETAIL_PROPERTY_BASES_HPP

/// v3 design: use template_key<Template> as the key for compile-time and
/// hybrid properties.  No sentinel values, no separate key types, no reserved
/// default template parameters.
///
/// Only three tag bases (same as v2):
///   runtime_property_base   -- runtime-only property (self-keyed by type)
///   constant_property_base  -- compile-time property
///   hybrid_property_base    -- mixed property
///
/// Plus a generic key wrapper:
///   template_key<Template>  -- key for any templated property

namespace sycl::khr3::detail {

// Tag bases for trait detection.
struct runtime_property_base {};
struct constant_property_base {};
struct hybrid_property_base {};

// Generic key wrapper: takes the property *template* (not an instantiation)
// as a template template parameter.  All instantiations of the same template
// produce the same key type.
//
// Example:
//   template<int V> struct prop2 { using __detail_key_t = template_key<prop2>; };
//   // prop2<0>::__detail_key_t == prop2<42>::__detail_key_t == template_key<prop2>
//
// This relies on C++20 P0522R0: template<auto...> matches template<int>, etc.
template <template <auto...> class>
struct template_key {};

// Variant for properties with typename parameters.
template <template <typename...> class>
struct type_template_key {};

// Convenience CRTP base for a runtime property (is its own key).
template <typename Derived>
struct runtime_property : runtime_property_base {
  using __detail_key_t = Derived;
};

} // namespace sycl::khr3::detail

#endif // SYCL_KHR3_DETAIL_PROPERTY_BASES_HPP
