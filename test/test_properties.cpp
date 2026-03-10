#include <cassert>
#include <type_traits>

#include <sycl/khr/properties.hpp>
#include <sycl/khr/queue_properties.hpp>

using namespace sycl::khr;

// ============================================================================
// Example compile-time property (variable template + separate key)
// ============================================================================

namespace sycl::khr {

struct prop2_key : detail::constant_property_key_base {};

template <int Value>
struct prop2_impl : detail::constant_property_base {
  using __detail_key_t = prop2_key;
  static constexpr int value = Value;
};

template <int Value>
inline constexpr prop2_impl<Value> prop2{};

} // namespace sycl::khr

// ============================================================================
// Example hybrid property (compile-time + runtime values)
// ============================================================================

namespace sycl::khr {

struct prop3_key : detail::hybrid_property_key_base {};

template <int Value1>
struct prop3 : detail::hybrid_property<prop3<Value1>, prop3_key> {
  constexpr prop3(int v2) : value2(v2) {}
  static constexpr int value1 = Value1;
  int value2;
};

} // namespace sycl::khr

// ============================================================================
// Trait tests
// ============================================================================

// is_property
static_assert(is_property_v<enable_profiling>);
static_assert(is_property_v<in_order>);
static_assert(is_property_v<prop2_impl<42>>);
static_assert(is_property_v<prop3<10>>);
static_assert(!is_property_v<int>);
static_assert(!is_property_v<queue_tag>);

// is_property_key
static_assert(is_property_key_v<enable_profiling>); // runtime prop is its own key
static_assert(is_property_key_v<in_order>);
static_assert(is_property_key_v<prop2_key>);
static_assert(is_property_key_v<prop3_key>);
static_assert(!is_property_key_v<prop2_impl<42>>); // the property itself is not a key
static_assert(!is_property_key_v<int>);

// is_property_key_compile_time
static_assert(is_property_key_compile_time_v<prop2_key>);
static_assert(!is_property_key_compile_time_v<enable_profiling>);
static_assert(!is_property_key_compile_time_v<prop3_key>); // hybrid key is not compile-time only

// is_property_for / is_property_key_for
static_assert(is_property_for_v<enable_profiling, queue_tag>);
static_assert(is_property_for_v<in_order, queue_tag>);
static_assert(is_property_key_for_v<enable_profiling, queue_tag>);
static_assert(is_property_key_for_v<in_order, queue_tag>);
static_assert(!is_property_for_v<enable_profiling, int>);

// is_property_list_for
static_assert(is_property_list_for_v<properties<enable_profiling, in_order>, queue_tag>);
static_assert(is_property_list_for_v<empty_properties_t, queue_tag>);
static_assert(is_property_list_for_v<properties<enable_profiling>, queue_tag>);
static_assert(!is_property_list_for_v<int, queue_tag>);

// ============================================================================
// empty_properties_t
// ============================================================================

static_assert(std::is_same_v<empty_properties_t, properties<>>);

// ============================================================================
// properties construction and has_property
// ============================================================================

// Empty.
constexpr properties empty{};
static_assert(!decltype(empty)::has_property<enable_profiling>());

// Runtime-only.
constexpr properties rt{enable_profiling{true}, in_order{false}};
static_assert(decltype(rt)::has_property<enable_profiling>());
static_assert(decltype(rt)::has_property<in_order>());
static_assert(!decltype(rt)::has_property<prop2_key>());

// Compile-time only.
constexpr properties ct{prop2<42>};
static_assert(decltype(ct)::has_property<prop2_key>());
static_assert(!decltype(ct)::has_property<enable_profiling>());

// Mixed list (runtime + compile-time properties).
constexpr properties mixed{enable_profiling{true}, prop2<7>};
static_assert(decltype(mixed)::has_property<enable_profiling>());
static_assert(decltype(mixed)::has_property<prop2_key>());

// ============================================================================
// get_property
// ============================================================================

// Static get_property for compile-time property.
static_assert(decltype(ct)::get_property<prop2_key>().value == 42);
static_assert(decltype(mixed)::get_property<prop2_key>().value == 7);

// Constexpr runtime get_property.
static_assert(rt.get_property<enable_profiling>().value == true);
static_assert(rt.get_property<in_order>().value == false);
static_assert(mixed.get_property<enable_profiling>().value == true);

// ============================================================================
// Hybrid property
// ============================================================================

constexpr properties hybrid_list{prop3<10>{20}};
static_assert(decltype(hybrid_list)::has_property<prop3_key>());
// Hybrid uses non-static get_property (has runtime state).
static_assert(hybrid_list.get_property<prop3_key>().value1 == 10);
static_assert(hybrid_list.get_property<prop3_key>().value2 == 20);

// ============================================================================
// Stub queue-like class using property constraints
// ============================================================================

struct my_queue {
  template <typename PropertyOrList = empty_properties_t>
    requires(is_property_for_v<PropertyOrList, queue_tag> ||
             is_property_list_for_v<PropertyOrList, queue_tag>)
  explicit constexpr my_queue(PropertyOrList props = {}) : props_{} {
    (void)props;
  }

private:
  int props_; // placeholder
};

// ============================================================================
// Runtime tests
// ============================================================================

int main() {
  // Runtime get_property with non-constexpr values.
  bool profiling = true;
  properties p1{enable_profiling{profiling}, in_order{false}};
  assert(p1.get_property<enable_profiling>().value == true);
  assert(p1.get_property<in_order>().value == false);

  // Hybrid property with runtime value.
  properties p2{prop3<5>{99}};
  assert(p2.get_property<prop3_key>().value1 == 5);
  assert(p2.get_property<prop3_key>().value2 == 99);

  // Queue-like construction.
  my_queue q1{};
  my_queue q2{enable_profiling{true}};
  my_queue q3{properties{in_order{true}, enable_profiling{false}}};

  (void)q1;
  (void)q2;
  (void)q3;

  return 0;
}
