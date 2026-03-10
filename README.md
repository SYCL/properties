# sycl_khr_properties — Minimal C++20 Implementation

A standalone, header-only reference implementation of the SYCL compile-time
properties extension proposed in
[KhronosGroup/SYCL-Docs#980](https://github.com/KhronosGroup/SYCL-Docs/pull/980).

This KHR introduces a `properties` class that supersedes `sycl::property_list`,
enabling compile-time property propagation so that:
- APIs can detect property errors at compilation time rather than runtime.
- Properties can influence codegen (e.g. kernel specialization).

## Property taxonomy

The extension defines three property kinds, distinguished by how their values
are conveyed:

| Kind | Value carrier | Key | Storage in `properties` |
|------|--------------|-----|------------------------|
| **Runtime** | Constructor args → member variables | The property type itself | Stored in internal tuple |
| **Compile-time** | Template parameters → `static constexpr` members | Separate `_key` type | Not stored (stateless, reconstructed) |
| **Hybrid** | Template params + constructor args | Separate `_key` type | Stored in internal tuple |

### Defining properties

```cpp
// Runtime property — struct with constexpr constructor.
struct in_order : detail::runtime_property<in_order> {
  constexpr in_order(bool v = true) : value{v} {}
  bool value;
};

// Compile-time property — variable template + key.
struct prop2_key : detail::constant_property_key_base {};

template <int V>
struct prop2_impl : detail::constant_property_base {
  using __detail_key_t = prop2_key;
  static constexpr int value = V;
};

template <int V>
inline constexpr prop2_impl<V> prop2{};

// Hybrid property — class template with both.
struct prop3_key : detail::hybrid_property_key_base {};

template <int V1>
struct prop3 : detail::hybrid_property<prop3<V1>, prop3_key> {
  constexpr prop3(int v2) : value2(v2) {}
  static constexpr int value1 = V1;
  int value2;
};
```

### Using properties

```cpp
using namespace sycl::khr;

// CTAD deduces EncodedProperties from constructor arguments.
constexpr properties plist{in_order{true}, prop2<42>};

// Existence check — always constexpr.
static_assert(decltype(plist)::has_property<in_order>());
static_assert(decltype(plist)::has_property<prop2_key>());

// Compile-time get (static) — value embedded in the type.
static_assert(decltype(plist)::get_property<prop2_key>().value == 42);

// Runtime get (non-static) — value stored in the properties object.
static_assert(plist.get_property<in_order>().value == true);
```

## Trait machinery

Six traits classify properties, keys, and property lists:

| Trait | Detected via |
|-------|-------------|
| `is_property<T>` | `is_base_of` against `{runtime,constant,hybrid}_property_base` |
| `is_property_key<T>` | `is_base_of` against `runtime_property_base`, `{constant,hybrid}_property_key_base` |
| `is_property_key_compile_time<T>` | `is_base_of<constant_property_key_base, T>` |
| `is_property_for<T, Class>` | Delegates to `is_property_key_for<T::__detail_key_t, Class>` |
| `is_property_key_for<T, Class>` | Explicit specialization per property × class pair |
| `is_property_list_for<T, Class>` | Partial specialization on `properties<...>`, folds `is_property_for` |

Only `is_property_key_for` needs a per-property specialization; all other traits
are derived automatically from the tag-base inheritance.

## Architecture

```
include/sycl/khr/
  detail/
    property_bases.hpp   — Tag bases: runtime, constant, hybrid (+ CRTP helpers)
    property_meta.hpp    — find_property, filter_runtime_properties, has_duplicate_keys
  properties.hpp         — properties<...> class, CTAD, 6 traits, empty_properties_t
  queue_properties.hpp   — enable_profiling, in_order (concrete queue properties)
```

### Key implementation details

- **Property-to-key mapping**: Every property type exposes
  `using __detail_key_t = <key>;`. Runtime properties are their own key.

- **Storage**: Only properties with runtime state are stored (in a
  `std::tuple`). Compile-time properties are stateless and reconstructed via
  default construction in `get_property()`.

- **`get_property` overload resolution**: Two overloads distinguished by
  `requires` clauses — `static` for `is_property_key_compile_time_v<Key>`,
  non-`static` otherwise.

- **Duplicate key detection**: `static_assert` in the constructor using a fold
  expression that counts key occurrences.

### Known deviations from the spec

- The spec requires `properties` to be trivially copyable when all contained
  runtime properties are trivially copyable. This implementation uses
  `std::tuple` for storage, which is not trivially copyable. A production
  implementation would use a custom aggregate with `[[no_unique_address]]`.

- No real SYCL runtime — `queue_tag` is used as a stub `Class` parameter for
  trait specializations.

## Building

Requires a C++20 compiler (GCC 12+, Clang 15+, MSVC 19.34+).

```bash
cmake -B build
cmake --build build
./build/test_properties
```

## References

- [SYCL-Docs PR #980](https://github.com/KhronosGroup/SYCL-Docs/pull/980) — KHR specification draft
- [intel/llvm#21368](https://github.com/intel/llvm/pull/21368) — Proof-of-concept implementation
