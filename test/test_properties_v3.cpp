#include <cassert>
#include <type_traits>

#include <sycl/khr3/properties.hpp>
#include <sycl/khr3/queue_properties.hpp>

using namespace sycl::khr3;

// ============================================================================
// Example compile-time property — using template_key<prop2>
//
// v1: struct prop2_key {};  // separate key type
//     template<int V> struct prop2_impl { using __detail_key_t = prop2_key; };
//
// v2: template<int V = 0> struct prop2 { using __detail_key_t = prop2<>; };
//     // caveat: prop2<0> == prop2<> is reserved as key
//
// v3: template<int V> struct prop2 { using __detail_key_t = template_key<prop2>; };
//     // NO sentinel, NO separate key type, prop2<0> is a valid property!
// ============================================================================

namespace sycl::khr3 {

template <int Value>
struct prop2 : detail::constant_property_base {
  using __detail_key_t = template_key<prop2>;  // key = the template itself
  static constexpr int value = Value;
};

} // namespace sycl::khr3

// ============================================================================
// Example compile-time property with a type value — using type_template_key
// ============================================================================

namespace sycl::khr3 {

template <typename Value>
struct prop_typed : detail::constant_property_base {
  using __detail_key_t = type_template_key<prop_typed>;
  using value_t = Value;
};

} // namespace sycl::khr3

// ============================================================================
// Example hybrid property
// ============================================================================

namespace sycl::khr3 {

template <int Value1>
struct prop3 : detail::hybrid_property_base {
  using __detail_key_t = template_key<prop3>;
  constexpr prop3(int v2 = 0) : value2(v2) {}
  static constexpr int value1 = Value1;
  int value2;
};

// Hybrid key is NOT compile-time-only (has runtime state).
// Override the default (template_key is compile-time by default).
template <>
struct is_property_key_compile_time<template_key<prop3>> : std::false_type {};

} // namespace sycl::khr3

// ============================================================================
// Trait tests
// ============================================================================

// is_property
static_assert(is_property_v<enable_profiling>);
static_assert(is_property_v<in_order>);
static_assert(is_property_v<prop2<0>>);
static_assert(is_property_v<prop2<42>>);
static_assert(is_property_v<prop_typed<int>>);
static_assert(is_property_v<prop3<10>>);
static_assert(!is_property_v<int>);
static_assert(!is_property_v<template_key<prop2>>);  // key is NOT a property

// is_property_key
static_assert(is_property_key_v<enable_profiling>);  // runtime = self-keyed
static_assert(is_property_key_v<in_order>);
static_assert(is_property_key_v<template_key<prop2>>);       // v3 key
static_assert(is_property_key_v<type_template_key<prop_typed>>);
static_assert(is_property_key_v<template_key<prop3>>);
static_assert(!is_property_key_v<prop2<0>>);   // prop2<0> is NOT a key
static_assert(!is_property_key_v<prop2<42>>);  // prop2<42> is NOT a key
static_assert(!is_property_key_v<int>);

// is_property_key_compile_time
static_assert(is_property_key_compile_time_v<template_key<prop2>>);
static_assert(is_property_key_compile_time_v<type_template_key<prop_typed>>);
static_assert(!is_property_key_compile_time_v<template_key<prop3>>);   // hybrid override
static_assert(!is_property_key_compile_time_v<enable_profiling>);       // runtime

// is_property_for / is_property_key_for
static_assert(is_property_for_v<enable_profiling, queue_tag>);
static_assert(is_property_for_v<in_order, queue_tag>);
static_assert(!is_property_for_v<enable_profiling, int>);

// is_property_list_for
static_assert(is_property_list_for_v<properties<enable_profiling, in_order>, queue_tag>);
static_assert(is_property_list_for_v<empty_properties_t, queue_tag>);
static_assert(!is_property_list_for_v<int, queue_tag>);

// ============================================================================
// empty_properties_t
// ============================================================================

static_assert(std::is_same_v<empty_properties_t, properties<>>);

// ============================================================================
// properties construction and has_property
//
// v3 query syntax:  has_property<template_key<prop2>>()
// ============================================================================

// Empty.
constexpr properties empty{};
static_assert(!decltype(empty)::has_property<enable_profiling>());

// Runtime-only.
constexpr properties rt{enable_profiling{true}, in_order{false}};
static_assert(decltype(rt)::has_property<enable_profiling>());
static_assert(decltype(rt)::has_property<in_order>());
static_assert(!decltype(rt)::has_property<template_key<prop2>>());

// Compile-time only.
constexpr properties ct{prop2<42>{}};
static_assert(decltype(ct)::has_property<template_key<prop2>>());
static_assert(!decltype(ct)::has_property<enable_profiling>());

// THE KEY TEST: prop2<0> is a valid property, NOT confused with the key.
constexpr properties ct_zero{prop2<0>{}};
static_assert(decltype(ct_zero)::has_property<template_key<prop2>>());
static_assert(decltype(ct_zero)::get_property<template_key<prop2>>().value == 0);

// Mixed list.
constexpr properties mixed{enable_profiling{true}, prop2<7>{}};
static_assert(decltype(mixed)::has_property<enable_profiling>());
static_assert(decltype(mixed)::has_property<template_key<prop2>>());

// Type-valued property.
constexpr properties typed{prop_typed<float>{}};
static_assert(decltype(typed)::has_property<type_template_key<prop_typed>>());
static_assert(std::is_same_v<
    decltype(decltype(typed)::get_property<type_template_key<prop_typed>>())::value_t,
    float>);

// ============================================================================
// get_property
// ============================================================================

// Static get_property for compile-time properties.
static_assert(decltype(ct)::get_property<template_key<prop2>>().value == 42);
static_assert(decltype(mixed)::get_property<template_key<prop2>>().value == 7);

// Constexpr runtime get_property.
static_assert(rt.get_property<enable_profiling>().value == true);
static_assert(rt.get_property<in_order>().value == false);
static_assert(mixed.get_property<enable_profiling>().value == true);

// ============================================================================
// Hybrid property
// ============================================================================

constexpr properties hybrid_list{prop3<10>{20}};
static_assert(decltype(hybrid_list)::has_property<template_key<prop3>>());
static_assert(hybrid_list.get_property<template_key<prop3>>().value1 == 10);
static_assert(hybrid_list.get_property<template_key<prop3>>().value2 == 20);

// ============================================================================
// Stub queue-like class
// ============================================================================

struct my_queue {
  template <typename PropertyOrList = empty_properties_t>
    requires(is_property_for_v<PropertyOrList, queue_tag> ||
             is_property_list_for_v<PropertyOrList, queue_tag>)
  explicit constexpr my_queue(PropertyOrList props = {}) : props_{} {
    (void)props;
  }

private:
  int props_;
};

// ============================================================================
// Runtime tests
// ============================================================================

int main() {
  // Runtime get_property.
  bool profiling = true;
  properties p1{enable_profiling{profiling}, in_order{false}};
  assert(p1.get_property<enable_profiling>().value == true);
  assert(p1.get_property<in_order>().value == false);

  // Hybrid property with runtime value.
  properties p2{prop3<5>{99}};
  assert(p2.get_property<template_key<prop3>>().value1 == 5);
  assert(p2.get_property<template_key<prop3>>().value2 == 99);

  // prop2<0> works as a real property.
  properties p3{prop2<0>{}};
  assert(p3.get_property<template_key<prop2>>().value == 0);

  // Queue-like construction.
  my_queue q1{};
  my_queue q2{enable_profiling{true}};
  my_queue q3{properties{in_order{true}, enable_profiling{false}}};

  (void)q1;
  (void)q2;
  (void)q3;

  return 0;
}
