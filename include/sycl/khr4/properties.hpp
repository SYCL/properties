#ifndef SYCL_KHR4_PROPERTIES_HPP
#define SYCL_KHR4_PROPERTIES_HPP

#include <tuple>
#include <type_traits>

#include "detail/property_bases.hpp"
#include "detail/property_meta.hpp"

namespace sycl::khr4 {

// Re-export ct_value for property authors.
using detail::ct_value;

// Forward declaration.
template <typename... EncodedProperties>
class properties;

// ============================================================================
// Traits
// ============================================================================

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
// v4: A type T is a property key if:
//   - T is a runtime property (self-keyed), OR
//   - T is a compile-time/hybrid property AND T == T::__detail_key_t
//     (the sentinel instantiation, where ct_value has no value).
//
// Same mechanism as v2 but now prop<> != prop<0> because ct_value
// distinguishes sentinel from value-0.

template <typename T, typename = void>
struct is_property_key : std::false_type {};

template <typename T>
struct is_property_key<
    T, std::enable_if_t<std::is_base_of_v<detail::runtime_property_base, T>>>
    : std::true_type {};

template <typename T>
struct is_property_key<
    T, std::enable_if_t<std::is_base_of_v<detail::constant_property_base, T> &&
                         std::is_same_v<T, typename T::__detail_key_t>>>
    : std::true_type {};

template <typename T>
struct is_property_key<
    T, std::enable_if_t<std::is_base_of_v<detail::hybrid_property_base, T> &&
                         std::is_same_v<T, typename T::__detail_key_t>>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_property_key_v = is_property_key<T>::value;

// --- is_property_key_compile_time ---

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

// --- is_property_key_for ---

template <typename T, typename Class>
struct is_property_key_for : std::false_type {};

template <typename T, typename Class>
inline constexpr bool is_property_key_for_v =
    is_property_key_for<T, Class>::value;

// --- is_property_for ---

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

  template <typename PropertyKey>
    requires(is_property_key_compile_time_v<PropertyKey> &&
             has_property<PropertyKey>())
  static constexpr auto get_property() {
    using Property =
        detail::find_property_t<PropertyKey, EncodedProperties...>;
    return Property{};
  }

  template <typename PropertyKey>
    requires(!is_property_key_compile_time_v<PropertyKey> &&
             is_property_key_v<PropertyKey> && has_property<PropertyKey>())
  constexpr auto get_property() const {
    using Property =
        detail::find_property_t<PropertyKey, EncodedProperties...>;
    return std::get<Property>(stored_);
  }
};

template <typename... Properties>
properties(Properties...) -> properties<Properties...>;

using empty_properties_t = decltype(properties{});

template <typename... Props, typename Class>
struct is_property_list_for<properties<Props...>, Class>
    : std::bool_constant<(is_property_for_v<Props, Class> && ...)> {};

template <typename Class>
struct is_property_list_for<properties<>, Class> : std::true_type {};

} // namespace sycl::khr4

#endif // SYCL_KHR4_PROPERTIES_HPP
