#include <cassert>
#include <type_traits>

#include <sycl/khr6/properties.hpp>
#include <sycl/khr6/queue_properties.hpp>

using namespace sycl::khr6;

// ============================================================================
// Compile-time property: auto NTTP + no_value sentinel TYPE.
//
// prop2_t is the internal struct (users don't need to name it).
// prop2<42> is the public variable template.
// prop2<> is the key — passed as an NTTP to has_property/get_property.
// ============================================================================

namespace sycl::khr6 {

template <auto V = no_value{}>
struct prop2_t : detail::constant_property_base {
  using __detail_key_t = prop2_t<>;
  static constexpr bool has_value = !detail::is_sentinel_v<V>;
  static constexpr auto value = V;
};

template <auto V = no_value{}>
inline constexpr prop2_t<V> prop2{};

} // namespace sycl::khr6

// ============================================================================
// Bool compile-time property — false and true both usable.
// ============================================================================

namespace sycl::khr6 {

template <auto V = no_value{}>
struct enable_thing_t : detail::constant_property_base {
  using __detail_key_t = enable_thing_t<>;
  static constexpr bool has_value = !detail::is_sentinel_v<V>;
  static constexpr auto value = V;
};

template <auto V = no_value{}>
inline constexpr enable_thing_t<V> enable_thing{};

} // namespace sycl::khr6

// ============================================================================
// Hybrid property (compile-time auto NTTP + runtime constructor arg).
// ============================================================================

namespace sycl::khr6 {

template <auto V1 = no_value{}>
struct prop3 : detail::hybrid_property_base {
  using __detail_key_t = prop3<>;
  constexpr prop3(int v2 = 0) : value2(v2) {}
  static constexpr auto value1 = V1;
  int value2;
};

} // namespace sycl::khr6

// ============================================================================
// Type-valued compile-time property (typename parameter).
// ============================================================================

namespace sycl::khr6 {

template <typename V = void>
struct prop_typed : detail::constant_property_base {
  using __detail_key_t = prop_typed<>;
  using value_t = V;
};

} // namespace sycl::khr6

// ============================================================================
// Trait tests
// ============================================================================

static_assert(is_property_v<enable_profiling>);
static_assert(is_property_v<in_order>);
static_assert(is_property_v<decltype(prop2<42>)>);
static_assert(is_property_v<decltype(prop2<0>)>);
static_assert(is_property_v<decltype(prop2<>)>);
static_assert(is_property_v<decltype(enable_thing<true>)>);
static_assert(is_property_v<decltype(enable_thing<false>)>);
static_assert(is_property_v<decltype(enable_thing<>)>);
static_assert(is_property_v<prop3<10>>);
static_assert(is_property_v<prop_typed<float>>);
static_assert(!is_property_v<int>);

// is_property_key — the sentinel variable's type is the key
static_assert(is_property_key_v<decltype(prop2<>)>);
static_assert(!is_property_key_v<decltype(prop2<0>)>);
static_assert(!is_property_key_v<decltype(prop2<42>)>);
static_assert(is_property_key_v<decltype(enable_thing<>)>);
static_assert(!is_property_key_v<decltype(enable_thing<false>)>);
static_assert(is_property_key_v<enable_profiling>);
static_assert(is_property_key_v<in_order>);
static_assert(!is_property_key_v<int>);

// is_property_key_compile_time
static_assert(is_property_key_compile_time_v<decltype(prop2<>)>);
static_assert(is_property_key_compile_time_v<decltype(enable_thing<>)>);
static_assert(!is_property_key_compile_time_v<prop3<>>);
static_assert(!is_property_key_compile_time_v<enable_profiling>);

// has_value is static constexpr
static_assert(decltype(prop2<42>)::has_value);
static_assert(decltype(prop2<0>)::has_value);
static_assert(!decltype(prop2<>)::has_value);
static_assert(decltype(enable_thing<false>)::has_value);
static_assert(!decltype(enable_thing<>)::has_value);

// is_property_for / is_property_key_for
static_assert(is_property_for_v<enable_profiling, queue_tag>);
static_assert(is_property_for_v<in_order, queue_tag>);
static_assert(!is_property_for_v<enable_profiling, int>);

// is_property_list_for
static_assert(
    is_property_list_for_v<properties<enable_profiling, in_order>, queue_tag>);
static_assert(is_property_list_for_v<empty_properties_t, queue_tag>);
static_assert(!is_property_list_for_v<int, queue_tag>);

// ============================================================================
// empty_properties_t
// ============================================================================

static_assert(std::is_same_v<empty_properties_t, properties<>>);

// ============================================================================
// KEY TESTS: prop2<0> is distinct from key prop2<>
// ============================================================================

static_assert(!std::is_same_v<decltype(prop2<0>), decltype(prop2<>)>);
static_assert(!std::is_same_v<decltype(prop2<42>), decltype(prop2<>)>);
static_assert(
    !std::is_same_v<decltype(enable_thing<false>), decltype(enable_thing<>)>);

// ============================================================================
// properties construction + has_property
//
// Users write:
//   prop2<42>              to construct a property (variable template)
//   has_property<prop2<>>  to query (NTTP overload)
//   get_property<prop2<>>  to retrieve (NTTP overload)
// ============================================================================

constexpr properties empty{};
static_assert(!decltype(empty)::has_property<enable_profiling>());

constexpr properties rt{enable_profiling{true}, in_order{false}};
static_assert(decltype(rt)::has_property<enable_profiling>());
static_assert(decltype(rt)::has_property<in_order>());
static_assert(!decltype(rt)::has_property<prop2<>>());

constexpr properties ct{prop2<42>};
static_assert(decltype(ct)::has_property<prop2<>>());
static_assert(!decltype(ct)::has_property<enable_profiling>());

constexpr properties ct_zero{prop2<0>};
static_assert(decltype(ct_zero)::has_property<prop2<>>());

constexpr properties mixed{enable_profiling{true}, prop2<7>};
static_assert(decltype(mixed)::has_property<enable_profiling>());
static_assert(decltype(mixed)::has_property<prop2<>>());

constexpr properties with_false{enable_thing<false>};
static_assert(decltype(with_false)::has_property<enable_thing<>>());

constexpr properties with_true{enable_thing<true>};
static_assert(decltype(with_true)::has_property<enable_thing<>>());

// Type-valued property (uses type parameter, not NTTP).
constexpr properties typed{prop_typed<float>{}};
static_assert(decltype(typed)::has_property<prop_typed<>>());
static_assert(std::is_same_v<
    decltype(decltype(typed)::get_property<prop_typed<>>())::value_t, float>);

// ============================================================================
// get_property — NTTP overload: get_property<prop2<>>()
// ============================================================================

static_assert(decltype(ct)::get_property<prop2<>>().value == 42);
static_assert(decltype(ct_zero)::get_property<prop2<>>().value == 0);
static_assert(decltype(mixed)::get_property<prop2<>>().value == 7);
static_assert(
    decltype(with_false)::get_property<enable_thing<>>().value == false);
static_assert(
    decltype(with_true)::get_property<enable_thing<>>().value == true);

// Type overload still works for runtime properties.
static_assert(rt.get_property<enable_profiling>().value == true);
static_assert(rt.get_property<in_order>().value == false);
static_assert(mixed.get_property<enable_profiling>().value == true);

// ============================================================================
// Hybrid property
// ============================================================================

constexpr properties hybrid_list{prop3<10>{20}};
static_assert(decltype(hybrid_list)::has_property<prop3<>>());
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
  assert(p2.get_property<prop3<>>().value2 == 99);

  properties p3{prop2<0>};
  assert(p3.get_property<prop2<>>().value == 0);

  my_queue q1{};
  my_queue q2{enable_profiling{true}};
  my_queue q3{properties{in_order{true}, enable_profiling{false}}};
  (void)q1;
  (void)q2;
  (void)q3;

  return 0;
}
