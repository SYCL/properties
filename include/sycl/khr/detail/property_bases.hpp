#ifndef SYCL_KHR_DETAIL_PROPERTY_BASES_HPP
#define SYCL_KHR_DETAIL_PROPERTY_BASES_HPP

namespace sycl::khr::detail {

// Tag bases for trait detection via std::is_base_of_v.

// Runtime-only property (is its own key).
struct runtime_property_base {};

// Compile-time-only property (value encoded in the type).
struct constant_property_base {};

// Key type for a compile-time property.
struct constant_property_key_base {};

// Property with both compile-time and runtime values.
struct hybrid_property_base {};

// Key type for a hybrid property.
struct hybrid_property_key_base {};

// Convenience CRTP base for a runtime property that is its own key.
template <typename Derived>
struct runtime_property : runtime_property_base {
  using __detail_key_t = Derived;
};

// Convenience CRTP base for a hybrid property.
// KeyType is the separate key class for this hybrid property.
template <typename Derived, typename KeyType>
struct hybrid_property : hybrid_property_base {
  using __detail_key_t = KeyType;
};

} // namespace sycl::khr::detail

#endif // SYCL_KHR_DETAIL_PROPERTY_BASES_HPP
