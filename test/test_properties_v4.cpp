#include <cassert>
#include <type_traits>

#include <sycl/khr4/properties.hpp>
#include <sycl/khr4/queue_properties.hpp>

using namespace sycl::khr4;

// ============================================================================
// Example compile-time property with ct_value wrapper
//
// v2: template<int V = 0> struct prop2 { using __detail_key_t = prop2<>; };
//     // CAVEAT: prop2<0> == prop2<> — value 0 is reserved as sentinel
//
// v4: template<ct_value<int> V = {}>
//     struct prop2 { using __detail_key_t = prop2<>; };
//     // prop<> is sentinel (has_value=false), prop<0> holds value 0
//     // NO value reservation!
//
// Thanks to ct_value's implicit constructor from int, users still write
// prop2<42> — the int is converted to ct_value<int>{42, true}.
// ============================================================================

namespace sycl::khr4 {

template <ct_value<int> V = {}>
struct prop2 : detail::constant_property_base {
  using __detail_key_t = prop2<>;  // sentinel = key
  static constexpr int value = V;  // implicit conversion to int
};

} // namespace sycl::khr4

// ============================================================================
// Compile-time property with bool value — v2 would waste false as sentinel!
// ============================================================================

namespace sycl::khr4 {

template <ct_value<bool> V = {}>
struct enable_thing : detail::constant_property_base {
  using __detail_key_t = enable_thing<>;
  static constexpr bool value = V;
};

} // namespace sycl::khr4

// ============================================================================
// Hybrid property
// ============================================================================

namespace sycl::khr4 {

template <ct_value<int> V1 = {}>
struct prop3 : detail::hybrid_property_base {
  using __detail_key_t = prop3<>;
  constexpr prop3(int v2 = 0) : value2(v2) {}
  static constexpr int value1 = V1;
  int value2;
};

} // namespace sycl::khr4

// ============================================================================
// Trait tests
// ============================================================================

// is_property
static_assert(is_property_v<enable_profiling>);
static_assert(is_property_v<in_order>);
static_assert(is_property_v<prop2<0>>);
static_assert(is_property_v<prop2<42>>);
static_assert(is_property_v<prop2<>>);       // sentinel is also a property type
static_assert(is_property_v<enable_thing<true>>);
static_assert(is_property_v<enable_thing<false>>);
static_assert(is_property_v<prop3<10>>);
static_assert(!is_property_v<int>);

// is_property_key
static_assert(is_property_key_v<enable_profiling>);
static_assert(is_property_key_v<in_order>);
static_assert(is_property_key_v<prop2<>>);         // sentinel = key
static_assert(!is_property_key_v<prop2<0>>);        // value 0 is NOT a key!
static_assert(!is_property_key_v<prop2<42>>);
static_assert(is_property_key_v<enable_thing<>>);
static_assert(!is_property_key_v<enable_thing<false>>);  // false is NOT sentinel!
static_assert(!is_property_key_v<enable_thing<true>>);
static_assert(is_property_key_v<prop3<>>);
static_assert(!is_property_key_v<prop3<10>>);

// is_property_key_compile_time
static_assert(is_property_key_compile_time_v<prop2<>>);
static_assert(is_property_key_compile_time_v<enable_thing<>>);
static_assert(!is_property_key_compile_time_v<prop3<>>);   // hybrid
static_assert(!is_property_key_compile_time_v<enable_profiling>);

// is_property_for / is_property_key_for
static_assert(is_property_for_v<enable_profiling, queue_tag>);
static_assert(is_property_for_v<in_order, queue_tag>);
static_assert(!is_property_for_v<enable_profiling, int>);

// is_property_list_for
static_assert(is_property_list_for_v<properties<enable_profiling, in_order>, queue_tag>);
static_assert(is_property_list_for_v<empty_properties_t, queue_tag>);

// ============================================================================
// empty_properties_t
// ============================================================================

static_assert(std::is_same_v<empty_properties_t, properties<>>);

// ============================================================================
// THE KEY TESTS: prop2<0> is valid and distinct from key prop2<>
// ============================================================================

// prop2<> is the key (sentinel, has_value=false)
// prop2<0> is a property with value 0 (has_value=true)
// They are DIFFERENT types.
static_assert(!std::is_same_v<prop2<>, prop2<0>>);

constexpr properties with_zero{prop2<0>{}};
static_assert(decltype(with_zero)::has_property<prop2<>>());
static_assert(decltype(with_zero)::get_property<prop2<>>().value == 0);

constexpr properties with_42{prop2<42>{}};
static_assert(decltype(with_42)::get_property<prop2<>>().value == 42);

// enable_thing<false> is a real property, not confused with the key
static_assert(!std::is_same_v<enable_thing<>, enable_thing<false>>);

constexpr properties with_false{enable_thing<false>{}};
static_assert(decltype(with_false)::has_property<enable_thing<>>());
static_assert(decltype(with_false)::get_property<enable_thing<>>().value == false);

constexpr properties with_true{enable_thing<true>{}};
static_assert(decltype(with_true)::get_property<enable_thing<>>().value == true);

// ============================================================================
// properties construction and has_property
// ============================================================================

constexpr properties empty{};
static_assert(!decltype(empty)::has_property<enable_profiling>());

constexpr properties rt{enable_profiling{true}, in_order{false}};
static_assert(decltype(rt)::has_property<enable_profiling>());
static_assert(decltype(rt)::has_property<in_order>());
static_assert(!decltype(rt)::has_property<prop2<>>());

constexpr properties mixed{enable_profiling{true}, prop2<7>{}};
static_assert(decltype(mixed)::has_property<enable_profiling>());
static_assert(decltype(mixed)::has_property<prop2<>>());

// ============================================================================
// get_property
// ============================================================================

static_assert(decltype(with_42)::get_property<prop2<>>().value == 42);
static_assert(decltype(mixed)::get_property<prop2<>>().value == 7);
static_assert(rt.get_property<enable_profiling>().value == true);
static_assert(rt.get_property<in_order>().value == false);

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
  bool profiling = true;
  properties p1{enable_profiling{profiling}, in_order{false}};
  assert(p1.get_property<enable_profiling>().value == true);
  assert(p1.get_property<in_order>().value == false);

  properties p2{prop3<5>{99}};
  assert(p2.get_property<prop3<>>().value1 == 5);
  assert(p2.get_property<prop3<>>().value2 == 99);

  // prop2<0> works as a real property value.
  properties p3{prop2<0>{}};
  assert(p3.get_property<prop2<>>().value == 0);

  my_queue q1{};
  my_queue q2{enable_profiling{true}};
  my_queue q3{properties{in_order{true}, enable_profiling{false}}};
  (void)q1;
  (void)q2;
  (void)q3;

  return 0;
}
