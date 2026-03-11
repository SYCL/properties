#ifndef SYCL_KHR6_PROPERTIES_HPP
#define SYCL_KHR6_PROPERTIES_HPP

#include <tuple>
#include <type_traits>

#include "detail/property_bases.hpp"
#include "detail/property_meta.hpp"

namespace sycl::khr6 {

using detail::no_value;

template <typename... EncodedProperties>
class properties;

// ============================================================================
// Traits
// ============================================================================

/// Is T a property type?
template <typename T>
struct is_property
    : std::bool_constant<
          std::is_base_of_v<detail::runtime_property_base, T> ||
          std::is_base_of_v<detail::constant_property_base, T> ||
          std::is_base_of_v<detail::hybrid_property_base, T>> {};

template <typename T>
inline constexpr bool is_property_v =
    is_property<std::remove_cv_t<T>>::value;

/// Is T a property key?
///
/// A type T is a key when T == T::__detail_key_t:
///   - Runtime properties are always their own key.
///   - Compile-time/hybrid properties: only the sentinel instantiation
///     (auto V = no_value{}) satisfies this.
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
inline constexpr bool is_property_key_v =
    is_property_key<std::remove_cv_t<T>>::value;

/// Is T the key of a compile-time-only property?
template <typename T, typename = void>
struct is_property_key_compile_time : std::false_type {};

template <typename T>
struct is_property_key_compile_time<
    T, std::enable_if_t<std::is_base_of_v<detail::constant_property_base, T> &&
                         std::is_same_v<T, typename T::__detail_key_t>>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_property_key_compile_time_v =
    is_property_key_compile_time<std::remove_cv_t<T>>::value;

/// Is T a property key valid for Class?  (Specialized per property.)
template <typename T, typename Class>
struct is_property_key_for : std::false_type {};

template <typename T, typename Class>
inline constexpr bool is_property_key_for_v =
    is_property_key_for<T, Class>::value;

/// Is T a property valid for Class?  (Delegates via __detail_key_t.)
template <typename T, typename Class, typename = void>
struct is_property_for : std::false_type {};

template <typename T, typename Class>
struct is_property_for<T, Class, std::enable_if_t<is_property_v<T>>>
    : is_property_key_for<typename T::__detail_key_t, Class> {};

template <typename T, typename Class>
inline constexpr bool is_property_for_v = is_property_for<T, Class>::value;

/// Is T a properties list whose members are all valid for Class?
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

  // --- has_property (type parameter) ---

  template <typename PropertyKey>
    requires(is_property_key_v<PropertyKey>)
  static constexpr bool has_property() {
    return (std::is_same_v<PropertyKey,
                           typename EncodedProperties::__detail_key_t> ||
            ...);
  }

  // --- has_property (NTTP: accepts a constexpr variable as key) ---

  template <auto Key>
    requires(is_property_key_v<std::remove_cvref_t<decltype(Key)>>)
  static constexpr bool has_property() {
    return has_property<std::remove_cvref_t<decltype(Key)>>();
  }

  // --- get_property (type parameter, compile-time) ---

  template <typename PropertyKey>
    requires(is_property_key_compile_time_v<PropertyKey> &&
             has_property<PropertyKey>())
  static constexpr auto get_property() {
    using Property =
        detail::find_property_t<PropertyKey, EncodedProperties...>;
    return Property{};
  }

  // --- get_property (type parameter, runtime/hybrid) ---

  template <typename PropertyKey>
    requires(!is_property_key_compile_time_v<PropertyKey> &&
             is_property_key_v<PropertyKey> && has_property<PropertyKey>())
  constexpr auto get_property() const {
    using Property =
        detail::find_property_t<PropertyKey, EncodedProperties...>;
    return std::get<Property>(stored_);
  }

  // --- get_property (NTTP: accepts a constexpr variable as key) ---

  template <auto Key>
    requires(is_property_key_compile_time_v<
                 std::remove_cvref_t<decltype(Key)>> &&
             has_property<std::remove_cvref_t<decltype(Key)>>())
  static constexpr auto get_property() {
    return get_property<std::remove_cvref_t<decltype(Key)>>();
  }

  template <auto Key>
    requires(
        !is_property_key_compile_time_v<std::remove_cvref_t<decltype(Key)>> &&
        is_property_key_v<std::remove_cvref_t<decltype(Key)>> &&
        has_property<std::remove_cvref_t<decltype(Key)>>())
  constexpr auto get_property() const {
    return get_property<std::remove_cvref_t<decltype(Key)>>();
  }
};

/// CTAD deduction guide.
template <typename... Properties>
properties(Properties...) -> properties<Properties...>;

/// Type of an empty properties list.
using empty_properties_t = decltype(properties{});

/// is_property_list_for partial specialization for properties<...>.
template <typename... Props, typename Class>
struct is_property_list_for<properties<Props...>, Class>
    : std::bool_constant<(is_property_for_v<Props, Class> && ...)> {};

template <typename Class>
struct is_property_list_for<properties<>, Class> : std::true_type {};

} // namespace sycl::khr6

#endif // SYCL_KHR6_PROPERTIES_HPP
