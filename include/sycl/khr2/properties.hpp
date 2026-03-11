#ifndef SYCL_KHR2_PROPERTIES_HPP
#define SYCL_KHR2_PROPERTIES_HPP

#include <tuple>
#include <type_traits>

#include "detail/property_bases.hpp"
#include "detail/property_meta.hpp"

namespace sycl::khr2 {

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
//   - T is a runtime property (it is its own key), OR
//   - T is a compile-time or hybrid property AND T is the default
//     instantiation of its template (i.e. T == T::__detail_key_t).
//
// This eliminates separate key base classes entirely.  The "key" for
// prop2<V> is prop2<> (the default instantiation), detected by checking
// std::is_same_v<T, typename T::__detail_key_t>.

template <typename T, typename = void>
struct is_property_key : std::false_type {};

// Runtime property: always its own key.
template <typename T>
struct is_property_key<
    T, std::enable_if_t<std::is_base_of_v<detail::runtime_property_base, T>>>
    : std::true_type {};

// Compile-time property: only a key when T is the default instantiation.
template <typename T>
struct is_property_key<
    T, std::enable_if_t<std::is_base_of_v<detail::constant_property_base, T> &&
                         std::is_same_v<T, typename T::__detail_key_t>>>
    : std::true_type {};

// Hybrid property: only a key when T is the default instantiation.
template <typename T>
struct is_property_key<
    T, std::enable_if_t<std::is_base_of_v<detail::hybrid_property_base, T> &&
                         std::is_same_v<T, typename T::__detail_key_t>>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_property_key_v = is_property_key<T>::value;

// --- is_property_key_compile_time ---
//
// True when T is the key of a compile-time-only property, i.e. T inherits
// from constant_property_base AND T is the default instantiation.

template <typename T, typename = void>
struct is_property_key_compile_time : std::false_type {};

template <typename T>
struct is_property_key_compile_time<
    T, std::enable_if_t<std::is_base_of_v<detail::constant_property_base, T> &&
                         std::is_same_v<T, typename T::__detail_key_t>>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_property_key_compile_time_v =
    is_property_key_compile_time<T>::value;

// --- is_property_key_for (default: false, specialized per property) ---

template <typename T, typename Class>
struct is_property_key_for : std::false_type {};

template <typename T, typename Class>
inline constexpr bool is_property_key_for_v =
    is_property_key_for<T, Class>::value;

// --- is_property_for (delegates to is_property_key_for via __detail_key_t) ---

template <typename T, typename Class, typename = void>
struct is_property_for : std::false_type {};

template <typename T, typename Class>
struct is_property_for<T, Class, std::enable_if_t<is_property_v<T>>>
    : is_property_key_for<typename T::__detail_key_t, Class> {};

template <typename T, typename Class>
inline constexpr bool is_property_for_v = is_property_for<T, Class>::value;

// --- is_property_list_for (specialized on properties<...>) ---

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

  // (2) Non-static get_property for runtime properties.
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

} // namespace sycl::khr2

#endif // SYCL_KHR2_PROPERTIES_HPP
