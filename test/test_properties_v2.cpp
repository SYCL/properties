#include <cassert>
#include <type_traits>

#include <sycl/khr2/properties.hpp>
#include <sycl/khr2/queue_properties.hpp>

using namespace sycl::khr2;

// ============================================================================
// Example compile-time property — NO SEPARATE KEY TYPE!
//
// v1 required:
//   struct prop2_key : detail::constant_property_key_base {};
//   template<int V> struct prop2_impl : ... { using __detail_key_t = prop2_key; };
//   template<int V> inline constexpr prop2_impl<V> prop2{};
//
// v2 just needs:
//   template<int V = 0> struct prop2 : ... { using __detail_key_t = prop2<>; };
//
// The key is prop2<> (the default instantiation), not a separate type.
// ============================================================================

namespace sycl::khr2 {

template <int Value = 0>
struct prop2 : detail::constant_property_base {
  using __detail_key_t = prop2<>;  // <-- key IS the default instantiation
  static constexpr int value = Value;
};

} // namespace sycl::khr2

// ============================================================================
// Example hybrid property — again, no separate key type.
//
// v1 required:
//   struct prop3_key : detail::hybrid_property_key_base {};
//   template<int V1> struct prop3 : hybrid_property<prop3<V1>, prop3_key> { ... };
//
// v2 just needs:
//   template<int V1 = 0> struct prop3 : hybrid_property_base {
//     using __detail_key_t = prop3<>;
//   };
// ============================================================================

namespace sycl::khr2 {

template <int Value1 = 0>
struct prop3 : detail::hybrid_property_base {
  using __detail_key_t = prop3<>;  // <-- key IS the default instantiation
  constexpr prop3(int v2 = 0) : value2(v2) {}
  static constexpr int value1 = Value1;
  int value2;
};

} // namespace sycl::khr2

// ============================================================================
// Trait tests
// ============================================================================

// is_property
static_assert(is_property_v<enable_profiling>);
static_assert(is_property_v<in_order>);
static_assert(is_property_v<prop2<42>>);
static_assert(is_property_v<prop2<>>);
static_assert(is_property_v<prop3<10>>);
static_assert(is_property_v<prop3<>>);
static_assert(!is_property_v<int>);

// is_property_key — the key insight of v2:
//   prop2<>  IS a key  (default instantiation)
//   prop2<42> is NOT a key (non-default instantiation)
static_assert(is_property_key_v<enable_profiling>);
static_assert(is_property_key_v<in_order>);
static_assert(is_property_key_v<prop2<>>);       // <-- default = key
static_assert(!is_property_key_v<prop2<42>>);     // <-- non-default = not a key
static_assert(is_property_key_v<prop3<>>);        // <-- default = key
static_assert(!is_property_key_v<prop3<10>>);     // <-- non-default = not a key
static_assert(!is_property_key_v<int>);

// is_property_key_compile_time
static_assert(is_property_key_compile_time_v<prop2<>>);   // compile-time prop key
static_assert(!is_property_key_compile_time_v<prop2<42>>); // not a key at all
static_assert(!is_property_key_compile_time_v<enable_profiling>); // runtime
static_assert(!is_property_key_compile_time_v<prop3<>>);   // hybrid, not pure compile-time

// is_property_for / is_property_key_for
static_assert(is_property_for_v<enable_profiling, queue_tag>);
static_assert(is_property_for_v<in_order, queue_tag>);
static_assert(is_property_key_for_v<enable_profiling, queue_tag>);
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
// Compare v1 vs v2 query syntax:
//   v1: plist.has_property<prop2_key>()
//   v2: plist.has_property<prop2<>>()     <-- no separate key type!
// ============================================================================

// Empty.
constexpr properties empty{};
static_assert(!decltype(empty)::has_property<enable_profiling>());

// Runtime-only.
constexpr properties rt{enable_profiling{true}, in_order{false}};
static_assert(decltype(rt)::has_property<enable_profiling>());
static_assert(decltype(rt)::has_property<in_order>());
static_assert(!decltype(rt)::has_property<prop2<>>());

// Compile-time only.
constexpr properties ct{prop2<42>{}};
static_assert(decltype(ct)::has_property<prop2<>>());        // query with prop2<>
static_assert(!decltype(ct)::has_property<enable_profiling>());

// Mixed list.
constexpr properties mixed{enable_profiling{true}, prop2<7>{}};
static_assert(decltype(mixed)::has_property<enable_profiling>());
static_assert(decltype(mixed)::has_property<prop2<>>());

// ============================================================================
// get_property
//
// Compare v1 vs v2:
//   v1: plist.get_property<prop2_key>().value
//   v2: plist.get_property<prop2<>>().value   <-- same template, no key type
// ============================================================================

// Static get_property for compile-time property.
static_assert(decltype(ct)::get_property<prop2<>>().value == 42);
static_assert(decltype(mixed)::get_property<prop2<>>().value == 7);

// Constexpr runtime get_property.
static_assert(rt.get_property<enable_profiling>().value == true);
static_assert(rt.get_property<in_order>().value == false);
static_assert(mixed.get_property<enable_profiling>().value == true);

// ============================================================================
// Hybrid property
// ============================================================================

constexpr properties hybrid_list{prop3<10>{20}};
static_assert(decltype(hybrid_list)::has_property<prop3<>>());
static_assert(hybrid_list.get_property<prop3<>>().value1 == 10);
static_assert(hybrid_list.get_property<prop3<>>().value2 == 20);

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
  assert(p2.get_property<prop3<>>().value1 == 5);
  assert(p2.get_property<prop3<>>().value2 == 99);

  // Queue-like construction.
  my_queue q1{};
  my_queue q2{enable_profiling{true}};
  my_queue q3{properties{in_order{true}, enable_profiling{false}}};

  (void)q1;
  (void)q2;
  (void)q3;

  return 0;
}
