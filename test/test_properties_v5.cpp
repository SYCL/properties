#include <cassert>
#include <type_traits>

#include <sycl/khr5/properties.hpp>
#include <sycl/khr5/queue_properties.hpp>

using namespace sycl::khr5;

// ============================================================================
// Compile-time property with parameter pack
//
// v2: template<int V = 0> struct prop2 { ... };
//     // prop2<> == prop2<0> — value 0 reserved as sentinel
//
// v4: template<ct_value<int> V = {}> struct prop2 { ... };
//     // No reserved value, but uses a bool has_value discriminant
//
// v5: template<int... Values> struct prop2;
//     template<> struct prop2<> { ... };       // empty pack = key
//     template<int V> struct prop2<V> { ... }; // single value = property
//     // No wrapper, no boolean, no sentinel. Pure C++ template specialization.
// ============================================================================

namespace sycl::khr5 {

template <int... Values>
struct prop2;

// Key form: empty pack.
template <>
struct prop2<> : detail::constant_property_base {
  using __detail_key_t = prop2<>;
};

// Value form: single element.
template <int V>
struct prop2<V> : detail::constant_property_base {
  using __detail_key_t = prop2<>;
  static constexpr int value = V;
};

} // namespace sycl::khr5

// ============================================================================
// Bool property — both false and true are usable values
// ============================================================================

namespace sycl::khr5 {

template <bool... Values>
struct enable_thing;

template <>
struct enable_thing<> : detail::constant_property_base {
  using __detail_key_t = enable_thing<>;
};

template <bool V>
struct enable_thing<V> : detail::constant_property_base {
  using __detail_key_t = enable_thing<>;
  static constexpr bool value = V;
};

} // namespace sycl::khr5

// ============================================================================
// Hybrid property
// ============================================================================

namespace sycl::khr5 {

template <int... Values>
struct prop3;

template <>
struct prop3<> : detail::hybrid_property_base {
  using __detail_key_t = prop3<>;
  constexpr prop3(int v2 = 0) : value2(v2) {}
  int value2;
};

template <int V1>
struct prop3<V1> : detail::hybrid_property_base {
  using __detail_key_t = prop3<>;
  constexpr prop3(int v2 = 0) : value2(v2) {}
  static constexpr int value1 = V1;
  int value2;
};

} // namespace sycl::khr5

// ============================================================================
// Type-valued compile-time property (uses typename... pack)
// ============================================================================

namespace sycl::khr5 {

template <typename... Values>
struct prop_typed;

template <>
struct prop_typed<> : detail::constant_property_base {
  using __detail_key_t = prop_typed<>;
};

template <typename V>
struct prop_typed<V> : detail::constant_property_base {
  using __detail_key_t = prop_typed<>;
  using value_t = V;
};

} // namespace sycl::khr5

// ============================================================================
// Trait tests
// ============================================================================

// is_property
static_assert(is_property_v<enable_profiling>);
static_assert(is_property_v<in_order>);
static_assert(is_property_v<prop2<0>>);
static_assert(is_property_v<prop2<42>>);
static_assert(is_property_v<prop2<>>);
static_assert(is_property_v<enable_thing<true>>);
static_assert(is_property_v<enable_thing<false>>);
static_assert(is_property_v<enable_thing<>>);
static_assert(is_property_v<prop3<10>>);
static_assert(is_property_v<prop_typed<float>>);
static_assert(!is_property_v<int>);

// is_property_key
static_assert(is_property_key_v<enable_profiling>);
static_assert(is_property_key_v<in_order>);
static_assert(is_property_key_v<prop2<>>);          // empty pack = key
static_assert(!is_property_key_v<prop2<0>>);         // NOT a key
static_assert(!is_property_key_v<prop2<42>>);
static_assert(is_property_key_v<enable_thing<>>);
static_assert(!is_property_key_v<enable_thing<false>>);
static_assert(!is_property_key_v<enable_thing<true>>);
static_assert(is_property_key_v<prop3<>>);
static_assert(!is_property_key_v<prop3<10>>);
static_assert(is_property_key_v<prop_typed<>>);
static_assert(!is_property_key_v<prop_typed<float>>);

// is_property_key_compile_time
static_assert(is_property_key_compile_time_v<prop2<>>);
static_assert(is_property_key_compile_time_v<enable_thing<>>);
static_assert(is_property_key_compile_time_v<prop_typed<>>);
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
// THE KEY TESTS: prop2<0> is distinct from prop2<> — no reserved values
// ============================================================================

// Structurally different types — empty pack vs single-element pack.
static_assert(!std::is_same_v<prop2<>, prop2<0>>);
static_assert(!std::is_same_v<prop2<>, prop2<1>>);
static_assert(!std::is_same_v<enable_thing<>, enable_thing<false>>);
static_assert(!std::is_same_v<enable_thing<>, enable_thing<true>>);

// prop2<0> as a real property.
constexpr properties with_zero{prop2<0>{}};
static_assert(decltype(with_zero)::has_property<prop2<>>());
static_assert(decltype(with_zero)::get_property<prop2<>>().value == 0);

// enable_thing<false> as a real property.
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

constexpr properties ct{prop2<42>{}};
static_assert(decltype(ct)::has_property<prop2<>>());
static_assert(!decltype(ct)::has_property<enable_profiling>());

constexpr properties mixed{enable_profiling{true}, prop2<7>{}};
static_assert(decltype(mixed)::has_property<enable_profiling>());
static_assert(decltype(mixed)::has_property<prop2<>>());

// Type-valued property.
constexpr properties typed{prop_typed<float>{}};
static_assert(decltype(typed)::has_property<prop_typed<>>());
static_assert(std::is_same_v<
    decltype(decltype(typed)::get_property<prop_typed<>>())::value_t, float>);

// ============================================================================
// get_property
// ============================================================================

static_assert(decltype(ct)::get_property<prop2<>>().value == 42);
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
// Stub queue
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
