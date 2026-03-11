#ifndef SYCL_KHR3_PROPERTIES_HPP
#define SYCL_KHR3_PROPERTIES_HPP

#include <tuple>
#include <type_traits>

#include "detail/property_bases.hpp"
#include "detail/property_meta.hpp"

namespace sycl::khr3 {

// Re-export template_key and type_template_key for user convenience.
using detail::template_key;
using detail::type_template_key;

// Forward declaration.
template <typename... EncodedProperties>
class properties;

// ============================================================================
// Traits
// ============================================================================

// --- is_property ---

template <typename T>
struct is_property
    : std::bool_constant<
          std::is_base_of_v<detail::runtime_property_base, T> ||
          std::is_base_of_v<detail::constant_property_base, T> ||
          std::is_base_of_v<detail::hybrid_property_base, T>> {};

template <typename T>
inline constexpr bool is_property_v = is_property<T>::value;

// --- is_property_key ---
//
// A type T is a property key if:
//   - T is a runtime property (self-keyed), OR
//   - T is a template_key<...> or type_template_key<...>.
//
// Unlike v2, non-default instantiations like prop2<0> are never keys.
// The key is always template_key<prop2>, which is not a property at all.

template <typename T>
struct is_property_key
    : std::bool_constant<
          std::is_base_of_v<detail::runtime_property_base, T> ||
          detail::is_template_key_v<T>> {};

template <typename T>
inline constexpr bool is_property_key_v = is_property_key<T>::value;

// --- is_property_key_compile_time ---
//
// True when T is a template_key<...> whose associated property inherits from
// constant_property_base.  For simplicity, we check whether the key is a
// template_key (not a type_template_key for hybrid) AND we rely on the
// convention that constant properties use template_key.
//
// A more precise check would require mapping key -> property, but for the
// common case we can use a trait that property authors specialize.
// Default: template_key is compile-time, type_template_key is compile-time,
// runtime properties are not.  Hybrid properties override via specialization.

template <typename T, typename = void>
struct is_property_key_compile_time : std::false_type {};

// template_key<TT> is compile-time by default.
template <template <auto...> class TT>
struct is_property_key_compile_time<template_key<TT>> : std::true_type {};

// type_template_key<TT> is compile-time by default.
template <template <typename...> class TT>
struct is_property_key_compile_time<type_template_key<TT>> : std::true_type {};

template <typename T>
inline constexpr bool is_property_key_compile_time_v =
    is_property_key_compile_time<T>::value;

// --- is_property_key_for (default: false, specialized per property) ---

template <typename T, typename Class>
struct is_property_key_for : std::false_type {};

template <typename T, typename Class>
inline constexpr bool is_property_key_for_v =
    is_property_key_for<T, Class>::value;

// --- is_property_for (delegates via __detail_key_t) ---

template <typename T, typename Class, typename = void>
struct is_property_for : std::false_type {};

template <typename T, typename Class>
struct is_property_for<T, Class, std::enable_if_t<is_property_v<T>>>
    : is_property_key_for<typename T::__detail_key_t, Class> {};

template <typename T, typename Class>
inline constexpr bool is_property_for_v = is_property_for<T, Class>::value;

// --- is_property_list_for ---

template <typename T, typename Class>
struct is_property_list_for : std::false_type {};

template <typename T, typename Class>
inline constexpr bool is_property_list_for_v =
    is_property_list_for<T, Class>::value;

// ============================================================================
// properties class
// ============================================================================

template <typename... EncodedProperties>
class properties {
private:
  using stored_t = detail::filter_runtime_properties_t<EncodedProperties...>;
  [[no_unique_address]] stored_t stored_;

  template <typename P>
  static constexpr auto wrap(P&& p) {
    if constexpr (detail::is_compile_time_property_v<std::remove_cvref_t<P>>) {
      return std::tuple<>{};
    } else {
      return std::tuple<std::remove_cvref_t<P>>{std::forward<P>(p)};
    }
  }

public:
  template <typename... Properties>
    requires(is_property_v<Properties> && ...)
  constexpr properties(Properties... props)
      : stored_{std::tuple_cat(wrap(std::move(props))...)} {
    static_assert(!detail::has_duplicate_keys_v<Properties...>,
                  "Properties list must not contain duplicate keys");
  }

  constexpr properties()
    requires(sizeof...(EncodedProperties) == 0)
      : stored_{} {}

  template <typename PropertyKey>
    requires(is_property_key_v<PropertyKey>)
  static constexpr bool has_property() {
    return (std::is_same_v<PropertyKey,
                           typename EncodedProperties::__detail_key_t> ||
            ...);
  }

  // (1) Static get_property for compile-time properties.
  template <typename PropertyKey>
    requires(is_property_key_compile_time_v<PropertyKey> &&
             has_property<PropertyKey>())
  static constexpr auto get_property() {
    using Property =
        detail::find_property_t<PropertyKey, EncodedProperties...>;
    return Property{};
  }

  // (2) Non-static get_property for runtime/hybrid properties.
  template <typename PropertyKey>
    requires(!is_property_key_compile_time_v<PropertyKey> &&
             is_property_key_v<PropertyKey> && has_property<PropertyKey>())
  constexpr auto get_property() const {
    using Property =
        detail::find_property_t<PropertyKey, EncodedProperties...>;
    return std::get<Property>(stored_);
  }
};

// CTAD deduction guide.
template <typename... Properties>
properties(Properties...) -> properties<Properties...>;

// empty_properties_t
using empty_properties_t = decltype(properties{});

// Partial specialization of is_property_list_for for properties<...>.
template <typename... Props, typename Class>
struct is_property_list_for<properties<Props...>, Class>
    : std::bool_constant<(is_property_for_v<Props, Class> && ...)> {};

template <typename Class>
struct is_property_list_for<properties<>, Class> : std::true_type {};

} // namespace sycl::khr3

#endif // SYCL_KHR3_PROPERTIES_HPP
