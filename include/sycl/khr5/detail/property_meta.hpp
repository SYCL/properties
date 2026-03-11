#ifndef SYCL_KHR5_DETAIL_PROPERTY_META_HPP
#define SYCL_KHR5_DETAIL_PROPERTY_META_HPP

#include <tuple>
#include <type_traits>

#include "property_bases.hpp"

namespace sycl::khr5::detail {

template <typename PropertyKey, typename... Properties>
struct find_property {
  using type = void;
};

template <typename PropertyKey, typename Property, typename... Rest>
struct find_property<PropertyKey, Property, Rest...> {
  using type = std::conditional_t<
      std::is_same_v<PropertyKey, typename Property::__detail_key_t>, Property,
      typename find_property<PropertyKey, Rest...>::type>;
};

template <typename PropertyKey, typename... Properties>
using find_property_t =
    typename find_property<PropertyKey, Properties...>::type;

template <typename T>
inline constexpr bool is_compile_time_property_v =
    std::is_base_of_v<constant_property_base, T>;

template <typename... Properties>
struct filter_runtime_properties;

template <>
struct filter_runtime_properties<> {
  using type = std::tuple<>;
};

template <typename Property, typename... Rest>
struct filter_runtime_properties<Property, Rest...> {
  using rest_t = typename filter_runtime_properties<Rest...>::type;
  using type = std::conditional_t<is_compile_time_property_v<Property>, rest_t,
                                  decltype(std::tuple_cat(
                                      std::declval<std::tuple<Property>>(),
                                      std::declval<rest_t>()))>;
};

template <typename... Properties>
using filter_runtime_properties_t =
    typename filter_runtime_properties<Properties...>::type;

template <typename Key, typename... Properties>
inline constexpr int count_key_v =
    (std::is_same_v<Key, typename Properties::__detail_key_t> + ...);

template <typename... Properties>
inline constexpr bool has_duplicate_keys_v =
    ((count_key_v<typename Properties::__detail_key_t, Properties...> > 1) ||
     ...);

template <>
inline constexpr bool has_duplicate_keys_v<> = false;

} // namespace sycl::khr5::detail

#endif // SYCL_KHR5_DETAIL_PROPERTY_META_HPP
